"""Histogram-binning confidence calibrator."""

from __future__ import annotations

import numpy as np

from calibprune.metrics.calibration import softmax


class HistogramBinning:
    name = "histogram_binning"

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self.bin_acc_: np.ndarray = np.zeros(n_bins, dtype=np.float64)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "HistogramBinning":
        probs = softmax(logits)
        conf = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == labels).astype(np.float64)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                mask = (conf >= edges[i]) & (conf <= edges[i + 1])
            else:
                mask = (conf >= edges[i]) & (conf < edges[i + 1])
            self.bin_acc_[i] = float(np.mean(correct[mask])) if np.any(mask) else (i + 0.5) / self.n_bins
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        probs = softmax(logits)
        conf = np.max(probs, axis=1)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        calibrated = probs.copy()
        top = np.argmax(probs, axis=1)
        for row, c in enumerate(conf):
            bin_idx = min(self.n_bins - 1, int(np.searchsorted(edges, c, side="right") - 1))
            target = np.clip(self.bin_acc_[bin_idx], 1e-6, 1.0 - 1e-6)
            old_top = calibrated[row, top[row]]
            if old_top < 1e-12:
                continue
            scale_rest = (1.0 - target) / max(1e-12, 1.0 - old_top)
            calibrated[row, :] *= scale_rest
            calibrated[row, top[row]] = target
        return np.log(np.clip(calibrated, 1e-12, 1.0))

