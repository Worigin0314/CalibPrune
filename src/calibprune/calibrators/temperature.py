"""Temperature scaling baseline."""

from __future__ import annotations

import numpy as np

from calibprune.metrics.calibration import negative_log_likelihood


class TemperatureScaling:
    name = "temperature_scaling"

    def __init__(self, grid: np.ndarray | None = None) -> None:
        self.grid = np.asarray(grid if grid is not None else np.arange(0.5, 3.05, 0.05), dtype=np.float64)
        self.temperature_: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        scores = [negative_log_likelihood(logits / t, labels, probabilities=False) for t in self.grid]
        self.temperature_ = float(self.grid[int(np.argmin(scores))])
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / self.temperature_

