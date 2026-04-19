import time
import numpy as np
from scipy import ndimage
import torch
from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network

THRESHOLD_FLOOR       = 0.50
THRESHOLD_STEP        = 0.05
MAX_FALLBACK_ATTEMPTS = 8

# Statistical threshold modes
STAT_MODES = ["mean_std", "percentile", "scaled_max"]


class VGN(object):
    def __init__(self, model_path, rviz=False, threshold=0.90,
                 adaptive=False, predictor_path=None, scene_type=None,
                 stat_threshold=None, stat_k=1.0):
        """
        Args:
            stat_threshold: one of 'mean_std', 'percentile', 'scaled_max'
                            If set, uses statistical threshold instead of MLP
            stat_k:         tuning constant for statistical threshold
                            mean_std:    threshold = mean + k * std  (default k=1.0)
                            percentile:  threshold = k-th percentile (default k=75)
                            scaled_max:  threshold = k * max         (default k=0.8)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz
        self.threshold = threshold
        self.adaptive = adaptive
        self.scene_type = scene_type
        self.stat_threshold = stat_threshold
        self.stat_k = stat_k

        if self.adaptive:
            from vgn.threshold_predictor import ThresholdPredictor
            assert predictor_path is not None, "Must provide --predictor path for adaptive mode"
            self.predictor = ThresholdPredictor(predictor_path)
            print("[VGN] Adaptive threshold mode enabled — predictor loaded from %s" % predictor_path)
        elif self.stat_threshold:
            assert stat_threshold in STAT_MODES, "stat_threshold must be one of %s" % STAT_MODES
            print("[VGN] Statistical threshold mode: %s  k=%.2f" % (stat_threshold, stat_k))
            self.predictor = None
        else:
            self.predictor = None

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_grid()
        voxel_size = state.tsdf.voxel_size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)

        object_count = getattr(state, 'num_objects', 5)
        features = extract_features(qual_vol, tsdf_vol, object_count=object_count)

        # pick threshold
        if self.adaptive and self.predictor is not None:
            threshold = self.predictor.predict(features, scene_type=self.scene_type)
        elif self.stat_threshold:
            threshold = statistical_threshold(
                features, mode=self.stat_threshold, k=self.stat_k
            )
        else:
            threshold = self.threshold

        # clamp to valid range
        threshold = float(np.clip(threshold, THRESHOLD_FLOOR, 0.99))

        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=threshold)

        # fallback: lower threshold if no grasps found
        attempts = 0
        current_threshold = threshold
        while len(grasps) == 0 and current_threshold > THRESHOLD_FLOOR and attempts < MAX_FALLBACK_ATTEMPTS:
            current_threshold = round(current_threshold - THRESHOLD_STEP, 2)
            current_threshold = max(current_threshold, THRESHOLD_FLOOR)
            grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=current_threshold)
            attempts += 1

        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)
        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        if self.rviz:
            vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

        return grasps, scores, toc, features


def statistical_threshold(features, mode="mean_std", k=1.0):
    """
    Compute a threshold directly from the scene's quality score distribution.
    No learned model needed — adapts to each scene automatically.

    Modes:
        mean_std:    threshold = qual_mean + k * qual_std
                     High-confidence scenes get higher threshold.
                     k=1.0 is one std above mean.

        percentile:  threshold = k-th percentile of nonzero qual scores
                     k=75 means only top 25% of voxels are candidates.

        scaled_max:  threshold = k * qual_max
                     Always relative to the best grasp in the scene.
                     k=0.8 means within 20% of the best score.
    """
    if mode == "mean_std":
        t = features["qual_mean"] + k * features["qual_std"]
    elif mode == "percentile":
        # use stored percentile features — interpolate k
        # k=50 → qual_p50, k=75 → qual_p75, k=90 → qual_p90
        if k <= 50:
            t = features["qual_p50"]
        elif k <= 75:
            frac = (k - 50) / 25.0
            t = features["qual_p50"] + frac * (features["qual_p75"] - features["qual_p50"])
        else:
            frac = (k - 75) / 15.0
            t = features["qual_p75"] + frac * (features["qual_p90"] - features["qual_p75"])
    elif mode == "scaled_max":
        t = k * features["qual_max"]
    else:
        t = 0.90  # fallback to default

    return float(np.clip(t, THRESHOLD_FLOOR, 0.99))


def extract_features(qual_vol, tsdf_vol, object_count=5):
    """Extract scene-level features from qual_vol before thresholding."""
    q = qual_vol.flatten()
    q_nonzero = q[q > 0]

    qual_mean     = float(np.mean(q_nonzero))            if len(q_nonzero) > 0 else 0.0
    qual_std      = float(np.std(q_nonzero))             if len(q_nonzero) > 0 else 0.0
    qual_max      = float(np.max(q_nonzero))             if len(q_nonzero) > 0 else 0.0
    qual_p50      = float(np.percentile(q_nonzero, 50))  if len(q_nonzero) > 0 else 0.0
    qual_p75      = float(np.percentile(q_nonzero, 75))  if len(q_nonzero) > 0 else 0.0
    qual_p90      = float(np.percentile(q_nonzero, 90))  if len(q_nonzero) > 0 else 0.0

    num_above_050 = int(np.sum(q > 0.50))
    num_above_065 = int(np.sum(q > 0.65))
    num_above_075 = int(np.sum(q > 0.75))
    num_above_085 = int(np.sum(q > 0.85))
    num_above_095 = int(np.sum(q > 0.95))

    tsdf = tsdf_vol.squeeze()
    tsdf_surface  = float(np.mean(np.logical_and(tsdf > 1e-3, tsdf < 0.5)))
    tsdf_occupied = float(np.mean(tsdf < 0.5))

    return {
        "qual_mean":      qual_mean,
        "qual_std":       qual_std,
        "qual_max":       qual_max,
        "qual_p50":       qual_p50,
        "qual_p75":       qual_p75,
        "qual_p90":       qual_p90,
        "num_above_050":  num_above_050,
        "num_above_065":  num_above_065,
        "num_above_075":  num_above_075,
        "num_above_085":  num_above_085,
        "num_above_095":  num_above_095,
        "tsdf_surface":   tsdf_surface,
        "tsdf_occupied":  tsdf_occupied,
        "object_count":   float(object_count),
    }


def predict(tsdf_vol, net, device):
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)
    qual_vol  = qual_vol.cpu().squeeze().numpy()
    rot_vol   = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )
    outside_voxels = tsdf_vol > 0.5
    inside_voxels  = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels   = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    qual_vol[qual_vol < threshold] = 0.0
    max_vol  = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask     = np.where(qual_vol, 1.0, 0.0)
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori   = Rotation.from_quat(rot_vol[:, i, j, k])
    pos   = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
