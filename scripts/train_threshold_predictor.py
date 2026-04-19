"""
train_threshold_predictor.py

Restored best configuration:
- Shared MLP (64→32→16)
- PC-weighted reward: 0.3·SR + 0.7·PC

Output: data/predictor/threshold_predictor.pkl
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

CANDIDATE_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

FEATURE_COLS = [
    "qual_mean", "qual_std", "qual_max",
    "qual_p50", "qual_p75", "qual_p90",
    "num_above_050", "num_above_065", "num_above_075",
    "num_above_085", "num_above_095",
    "tsdf_surface", "tsdf_occupied",
    "object_count",
    "threshold",
]


def main():
    data_path = Path("data/predictor/dataset.csv")
    assert data_path.exists(), "Run build_predictor_dataset.py first"

    df = pd.read_csv(data_path)
    df = df.dropna()  # ADDED: drop rows with NaN features
    print("Dataset: %d rows (after dropping NaN)" % len(df))
    print("Scene types: %s" % df["scene_type"].value_counts().to_dict())

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["reward"].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )

    cv = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")
    print("\n5-fold CV MAE: %.4f ± %.4f" % (-cv.mean(), cv.std()))

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    print("Train MAE: %.4f" % mean_absolute_error(y, y_pred))

    output_dir = Path("data/predictor")
    predictor = {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": FEATURE_COLS,
        "mode":         "shared",
    }
    out_path = output_dir / "threshold_predictor.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(predictor, f)
    print("\nPredictor saved to %s" % out_path)

    print("\n--- Sanity check: predicted best threshold per scene type ---")
    available_scenes = df["scene_type"].unique().tolist()
    for scene_type in available_scenes:
        subset = df[df["scene_type"] == scene_type]
        mean_features = subset[FEATURE_COLS[:-1]].mean().values

        # handle any remaining NaN in mean features
        mean_features = np.nan_to_num(mean_features, nan=0.0)

        best_thresh, best_reward = None, -1
        for t in CANDIDATE_THRESHOLDS:
            x = np.append(mean_features, t).reshape(1, -1).astype(np.float32)
            x_scaled = scaler.transform(x)
            pred_reward = model.predict(x_scaled)[0]
            if pred_reward > best_reward:
                best_reward = pred_reward
                best_thresh = t

        print("  %-6s → best threshold: %.2f  (reward: %.3f)" % (
            scene_type, best_thresh, best_reward
        ))

    print("\n--- Full reward curves ---")
    for scene_type in available_scenes:
        subset = df[df["scene_type"] == scene_type]
        mean_features = np.nan_to_num(subset[FEATURE_COLS[:-1]].mean().values, nan=0.0)
        print("  %s:" % scene_type, end="")
        for t in CANDIDATE_THRESHOLDS:
            x = np.append(mean_features, t).reshape(1, -1).astype(np.float32)
            x_scaled = scaler.transform(x)
            r = model.predict(x_scaled)[0]
            print("  %.2f→%.3f" % (t, r), end="")
        print()


if __name__ == "__main__":
    main()
