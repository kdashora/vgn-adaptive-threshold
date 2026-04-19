import collections
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn import io, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

MAX_CONSECUTIVE_FAILURES = 2

# UPDATED: State now carries num_objects so detection can use it as a feature
State = collections.namedtuple("State", ["tsdf", "pc", "num_objects"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    rviz=False,
):
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed)
    logger = Logger(logdir, description)

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)

            if pc.is_empty():
                break

            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
                vis.draw_points(np.asarray(pc.points))

            # UPDATED: pass current num_objects into state
            state = State(tsdf, pc, sim.num_objects)

            grasps, scores, timings["planning"], features = grasp_plan_fn(state)

            if len(grasps) == 0:
                break

            if rviz:
                vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            grasp, score = grasps[0], scores[0]
            if rviz:
                vis.draw_grasp(grasp, score, sim.gripper.finger_depth)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)

            logger.log_grasp(round_id, state, timings, grasp, score, label, features, sim.num_objects)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path   = self.logdir / "rounds.csv"
        self.grasps_csv_path   = self.logdir / "grasps.csv"
        self.features_csv_path = self.logdir / "features.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id", "scene_id",
                "qx", "qy", "qz", "qw",
                "x", "y", "z",
                "width", "score", "label",
                "integration_time", "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

        if not self.features_csv_path.exists():
            feature_columns = [
                "round_id", "scene_id", "object_count",
                "qual_mean", "qual_std", "qual_max",
                "qual_p50", "qual_p75", "qual_p90",
                "num_above_050", "num_above_065", "num_above_075",
                "num_above_085", "num_above_095",
                "tsdf_surface", "tsdf_occupied",
                "label",
            ]
            io.create_csv(self.features_csv_path, feature_columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label, features, object_count):
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label_int = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id, scene_id,
            qx, qy, qz, qw,
            x, y, z,
            width, score, label_int,
            timings["integration"], timings["planning"],
        )

        io.append_csv(
            self.features_csv_path,
            round_id, scene_id, object_count,
            features["qual_mean"],    features["qual_std"],     features["qual_max"],
            features["qual_p50"],     features["qual_p75"],     features["qual_p90"],
            features["num_above_050"], features["num_above_065"], features["num_above_075"],
            features["num_above_085"], features["num_above_095"],
            features["tsdf_surface"], features["tsdf_occupied"],
            label_int,
        )


class Data(object):
    def __init__(self, logdir):
        self.logdir  = logdir
        self.rounds  = pd.read_csv(logdir / "rounds.csv")
        self.grasps  = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))
        return scene_data["points"], grasp, score, label
