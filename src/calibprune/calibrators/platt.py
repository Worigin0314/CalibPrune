"""Simple margin-based Platt baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibprune.metrics.calibration import softmax


class PlattScaling:
    name = "platt"

    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=20260616)

    @staticmethod
    def _features(logits: np.ndarray) -> np.ndarray:
        probs = softmax(logits)
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        conf = sorted_probs[:, -1]
        return np.stack([conf, margin], axis=1)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        preds = np.argmax(logits, axis=1)
        y = (preds == labels).astype(int)
        if len(np.unique(y)) < 2:
            return self
        self.model.fit(self._features(logits), y)
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        # Keep class ordering and rescale confidence only. This is a baseline
        # placeholder; paper numbers should prefer probability-space reporting.
        logits = np.asarray(logits, dtype=np.float64)
        if not hasattr(self.model, "classes_"):
            return logits
        correctness_prob = self.model.predict_proba(self._features(logits))[:, -1]
        adjusted = logits.copy()
        top = np.argmax(adjusted, axis=1)
        adjusted[np.arange(adjusted.shape[0]), top] += np.log(np.clip(correctness_prob, 1e-6, 1.0))
        return adjusted

