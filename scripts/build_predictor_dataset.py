"""
build_predictor_dataset.py

Combines sweep data across all object counts (3, 5, 7, 10) and both scenes.
This gives the predictor rich signal to learn scene-adaptive thresholds.

Output: data/predictor/dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

SWEEP_RUNS = [
    # Apr 15 — pile 5obj (original sweep)
    ("data/experiments/26-04-15-12-14-03", 0.65, "pile",   5),
    ("data/experiments/26-04-15-12-21-44", 0.75, "pile",   5),
    ("data/experiments/26-04-15-12-29-36", 0.85, "pile",   5),
    ("data/experiments/26-04-15-12-37-29", 0.95, "pile",   5),

    # Apr 15 — packed 5obj (original sweep)
    ("data/experiments/26-04-15-12-18-23", 0.65, "packed", 5),
    ("data/experiments/26-04-15-12-26-08", 0.75, "packed", 5),
    ("data/experiments/26-04-15-12-34-03", 0.85, "packed", 5),
    ("data/experiments/26-04-15-12-41-20", 0.95, "packed", 5),

    # Apr 18 — pile 10obj
    ("data/experiments/26-04-18-11-25-03", 0.65, "pile",   10),
    ("data/experiments/26-04-18-11-43-43", 0.75, "pile",   10),
    ("data/experiments/26-04-18-12-02-58", 0.85, "pile",   10),
    ("data/experiments/26-04-18-12-22-29", 0.95, "pile",   10),

    # Apr 18 — pile 3obj
    ("data/experiments/26-04-18-14-25-45", 0.65, "pile",   3),
    ("data/experiments/26-04-18-14-30-09", 0.75, "pile",   3),
    ("data/experiments/26-04-18-14-34-30", 0.85, "pile",   3),
    ("data/experiments/26-04-18-14-38-57", 0.95, "pile",   3),

    # Apr 18 — pile 7obj
    ("data/experiments/26-04-18-14-43-24", 0.65, "pile",   7),
    ("data/experiments/26-04-18-14-55-13", 0.75, "pile",   7),
    ("data/experiments/26-04-18-15-06-49", 0.85, "pile",   7),
    ("data/experiments/26-04-18-15-18-38", 0.95, "pile",   7),

    # Apr 18 — packed
    ("data/experiments/26-04-18-15-30-32", 0.65, "packed", 5),
    ("data/experiments/26-04-18-15-36-43", 0.75, "packed", 5),
    ("data/experiments/26-04-18-15-42-53", 0.85, "packed", 5),
    ("data/experiments/26-04-18-15-49-05", 0.95, "packed", 5),
]

FEATURE_COLS = [
    "qual_mean", "qual_std", "qual_max",
    "qual_p50", "qual_p75", "qual_p90",
    "num_above_050", "num_above_065", "num_above_075",
    "num_above_085", "num_above_095",
    "tsdf_surface", "tsdf_occupied",
    "threshold",
]

def main():
    output_dir = Path("data/predictor")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for run_path, threshold, scene_type, obj_count in SWEEP_RUNS:
        run_dir = Path(run_path)
        if not run_dir.exists():
            print("MISSING: %s — skipping" % run_path)
            continue

        features_df = pd.read_csv(run_dir / "features.csv")
        rounds_df   = pd.read_csv(run_dir / "rounds.csv")
        grasps_df   = pd.read_csv(run_dir / "grasps.csv")

        cleared = (
            grasps_df.groupby("round_id")["label"]
            .sum().reset_index()
            .rename(columns={"label": "cleared_count"})
        )
        rounds_merged = rounds_df.merge(cleared, on="round_id", how="left").fillna(0)
        rounds_merged["pct_cleared"] = (
            rounds_merged["cleared_count"] / rounds_merged["object_count"]
        )

        sr_per_round = (
            grasps_df.groupby("round_id")["label"]
            .mean().reset_index()
            .rename(columns={"label": "success_rate"})
        )
        rounds_merged = rounds_merged.merge(sr_per_round, on="round_id", how="left").fillna(0)

        first_features = (
            features_df.sort_index()
            .groupby("round_id").first()
            .reset_index()
        )

        merged = rounds_merged.merge(
            first_features[["round_id"] + [c for c in FEATURE_COLS if c != "threshold"]],
            on="round_id", how="inner"
        )

        for _, row in merged.iterrows():
            entry = {col: row[col] for col in FEATURE_COLS if col != "threshold"}
            entry["threshold"]    = threshold
            entry["scene_type"]   = scene_type
            entry["object_count"] = obj_count
            entry["pct_cleared"]  = row["pct_cleared"]
            entry["success_rate"] = row["success_rate"]
            entry["reward"]       = 0.3 * row["success_rate"] + 0.7 * row["pct_cleared"]
            all_rows.append(entry)

        print("%-40s  thresh=%.2f  scene=%-6s  obj=%2d  rounds=%d" % (
            run_dir.name, threshold, scene_type, obj_count, len(merged)
        ))

    dataset = pd.DataFrame(all_rows)
    out_path = output_dir / "dataset.csv"
    dataset.to_csv(out_path, index=False)

    print("\nDataset saved to %s" % out_path)
    print("Total rounds: %d" % len(dataset))
    print("\nReward by scene + object_count + threshold:")
    print(dataset.groupby(["scene_type", "object_count", "threshold"])["reward"]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
