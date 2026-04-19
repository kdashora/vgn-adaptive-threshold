"""
threshold_predictor.py

Loads the trained threshold predictor and picks the optimal threshold per scene.
Now includes object_count as a feature for scene-complexity-aware predictions.
"""

import numpy as np
import pickle
from pathlib import Path

CANDIDATE_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


class ThresholdPredictor:
    def __init__(self, model_path):
        model_path = Path(model_path)
        assert model_path.exists(), "Predictor not found at %s" % model_path
        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.mode         = saved.get("mode", "shared")
        self.feature_cols = saved["feature_cols"]
        self.model        = saved["model"]
        self.scaler       = saved["scaler"]
        print("[ThresholdPredictor] Loaded shared MLP predictor")
        print("[ThresholdPredictor] Features: %s" % self.feature_cols)

    def predict(self, features: dict, scene_type: str = None, candidates=None) -> float:
        if candidates is None:
            candidates = CANDIDATE_THRESHOLDS

        base_cols = [c for c in self.feature_cols if c != "threshold"]
        # use .get with 0.0 default so missing features don't crash
        base_vec  = np.array([features.get(c, 0.0) for c in base_cols], dtype=np.float32)
        base_vec  = np.nan_to_num(base_vec, nan=0.0)

        best_thresh, best_reward = candidates[0], -1.0
        for t in candidates:
            x = np.append(base_vec, t).reshape(1, -1).astype(np.float32)
            x_scaled    = self.scaler.transform(x)
            pred_reward = self.model.predict(x_scaled)[0]
            if pred_reward > best_reward:
                best_reward = pred_reward
                best_thresh = t

        return best_thresh
