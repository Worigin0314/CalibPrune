"""Calibrator interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Calibrator(Protocol):
    name: str

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "Calibrator":
        ...

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        ...


class IdentityCalibrator:
    name = "none"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64)

