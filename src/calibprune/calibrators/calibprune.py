"""Pruning-coupled temperature calibration."""

from __future__ import annotations

import numpy as np

from calibprune.metrics.calibration import (
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)
from calibprune.metrics.selective import aurc


class CalibPrune:
    name = "calibprune"

    def __init__(
        self,
        t0_grid: np.ndarray | None = None,
        beta_grid: np.ndarray | None = None,
        objective: str = "ece",
        n_bins: int = 15,
    ) -> None:
        self.t0_grid = np.asarray(t0_grid if t0_grid is not None else np.arange(0.8, 1.51, 0.1), dtype=np.float64)
        self.beta_grid = np.asarray(beta_grid if beta_grid is not None else np.arange(0.0, 2.01, 0.1), dtype=np.float64)
        self.objective = objective
        self.n_bins = n_bins
        self.t0_star: float = 1.0
        self.beta_star: float = 0.0
        self.best_score_: float = float("inf")

    def temperature_at(self, retention: float) -> float:
        return float(self.t0_star + self.beta_star * (1.0 - retention))

    def _score(self, logits: np.ndarray, labels: np.ndarray) -> float:
        if self.objective == "nll":
            return negative_log_likelihood(logits, labels, probabilities=False)
        if self.objective == "ece":
            return expected_calibration_error(logits, labels, n_bins=self.n_bins, probabilities=False)
        raise ValueError(f"Unknown objective: {self.objective}")

    def fit(self, logits_at_ratios: dict[float, np.ndarray], labels: np.ndarray) -> "CalibPrune":
        if not logits_at_ratios:
            raise ValueError("logits_at_ratios cannot be empty.")
        if len(logits_at_ratios) < 2:
            raise ValueError("CalibPrune requires at least two retention ratios to identify t0 and beta.")
        labels = np.asarray(labels, dtype=np.int64)
        best: tuple[float, float, float] | None = None
        for t0 in self.t0_grid:
            for beta in self.beta_grid:
                scores = []
                for retention, logits in logits_at_ratios.items():
                    temp = t0 + beta * (1.0 - float(retention))
                    if temp <= 0:
                        continue
                    scores.append(self._score(np.asarray(logits, dtype=np.float64) / temp, labels))
                if not scores:
                    continue
                score = float(np.mean(scores))
                if best is None or score < best[0]:
                    best = (score, float(t0), float(beta))
        if best is None:
            raise RuntimeError("No valid CalibPrune temperature candidate found.")
        self.best_score_, self.t0_star, self.beta_star = best
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        if retention is None:
            raise ValueError("CalibPrune.transform requires a retention value.")
        return np.asarray(logits, dtype=np.float64) / self.temperature_at(retention)

    def state_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "objective": self.objective,
            "t0_star": self.t0_star,
            "beta_star": self.beta_star,
            "best_score": self.best_score_,
        }


class AdaptiveCalibPrune(CalibPrune):
    """Retention- and confidence-adaptive variant of CalibPrune.

    The base CalibPrune model uses one temperature per retention ratio:

        T(r) = T0 + beta * (1 - r)

    This variant adds a lightweight sample-level sharpness term:

        T_i(r) = T0 + beta * (1 - r) + gamma * z_i

    where z_i is a standardized sharpness feature of the pruned logits.  The
    default uses probability margin and a log-temperature link so temperatures
    are always positive.  Setting gamma = 0 reduces the model to the original
    retention-coupled temperature family.
    """

    name = "adaptive_calibprune"

    def __init__(
        self,
        t0_grid: np.ndarray | None = None,
        beta_grid: np.ndarray | None = None,
        gamma_grid: np.ndarray | None = None,
        objective: str = "ece",
        n_bins: int = 15,
        min_temperature: float = 0.05,
        feature: str = "margin",
        temperature_mode: str = "log",
        gamma_l2: float = 0.05,
        selective_weight: float = 0.0,
        selective_score: str = "max_softmax",
        validation_fraction: float = 0.0,
        validation_seed: int = 20260620,
    ) -> None:
        super().__init__(t0_grid=t0_grid, beta_grid=beta_grid, objective=objective, n_bins=n_bins)
        if temperature_mode not in {"linear", "log"}:
            raise ValueError("temperature_mode must be 'linear' or 'log'.")
        if feature not in {"confidence", "margin", "entropy"}:
            raise ValueError("feature must be 'confidence', 'margin', or 'entropy'.")
        if selective_score not in {"max_softmax", "margin", "entropy"}:
            raise ValueError("selective_score must be 'max_softmax', 'margin', or 'entropy'.")
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1).")
        default_gamma = np.arange(0.0, 1.01, 0.1) if temperature_mode == "linear" else np.arange(-0.5, 0.51, 0.1)
        self.gamma_grid = np.asarray(gamma_grid if gamma_grid is not None else default_gamma, dtype=np.float64)
        self.min_temperature = float(min_temperature)
        self.feature = feature
        self.temperature_mode = temperature_mode
        self.gamma_l2 = float(gamma_l2)
        self.selective_weight = float(selective_weight)
        self.selective_score = selective_score
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.gamma_star: float = 0.0
        self.feature_center_: float = 0.0
        self.feature_scale_: float = 1.0
        self.validation_count_: int = 0
        self.selection_count_: int = 0

    def _raw_feature(self, logits: np.ndarray) -> np.ndarray:
        probs = softmax(np.asarray(logits, dtype=np.float64))
        if self.feature == "confidence":
            return np.max(probs, axis=1)
        if self.feature == "margin":
            if probs.shape[1] < 2:
                return np.zeros(probs.shape[0], dtype=np.float64)
            top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
            return top2[:, 1] - top2[:, 0]
        if self.feature == "entropy":
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
            return -entropy
        raise ValueError(f"Unknown feature: {self.feature}")

    def _standardized_feature(self, logits: np.ndarray) -> np.ndarray:
        return (self._raw_feature(logits) - self.feature_center_) / self.feature_scale_

    def _temperatures_from_feature(self, base: float, feature: np.ndarray, gamma: float) -> np.ndarray:
        if base <= 0:
            return np.full(feature.shape, -1.0, dtype=np.float64)
        if self.temperature_mode == "linear":
            return base + gamma * feature
        if self.temperature_mode == "log":
            return base * np.exp(gamma * feature)
        raise ValueError(f"Unknown temperature mode: {self.temperature_mode}")

    def sample_temperatures(self, logits: np.ndarray, retention: float) -> np.ndarray:
        base = self.temperature_at(retention)
        feature = self._standardized_feature(logits)
        temps = self._temperatures_from_feature(base, feature, self.gamma_star)
        if np.any(temps <= 0):
            raise ValueError("AdaptiveCalibPrune produced non-positive temperatures.")
        return temps

    def _train_selection_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        all_idx = np.arange(n, dtype=np.int64)
        if self.validation_fraction <= 0.0 or n < 4:
            self.validation_count_ = 0
            self.selection_count_ = n
            return all_idx, all_idx
        validation_count = int(round(n * self.validation_fraction))
        validation_count = max(1, min(n - 1, validation_count))
        rng = np.random.default_rng(self.validation_seed)
        shuffled = rng.permutation(all_idx)
        validation_idx = np.sort(shuffled[:validation_count])
        train_idx = np.sort(shuffled[validation_count:])
        self.validation_count_ = int(validation_idx.shape[0])
        self.selection_count_ = int(validation_idx.shape[0])
        return train_idx, validation_idx

    def fit(self, logits_at_ratios: dict[float, np.ndarray], labels: np.ndarray) -> "AdaptiveCalibPrune":
        if not logits_at_ratios:
            raise ValueError("logits_at_ratios cannot be empty.")
        if len(logits_at_ratios) < 2:
            raise ValueError("AdaptiveCalibPrune requires at least two retention ratios to identify t0, beta, and gamma.")
        labels = np.asarray(labels, dtype=np.int64)
        n = int(labels.shape[0])
        train_idx, selection_idx = self._train_selection_indices(n)
        all_features = np.concatenate(
            [
                self._raw_feature(np.asarray(logits, dtype=np.float64)[train_idx])
                for logits in logits_at_ratios.values()
            ]
        )
        self.feature_center_ = float(np.mean(all_features))
        feature_scale = float(np.std(all_features))
        self.feature_scale_ = feature_scale if feature_scale > 1e-12 else 1.0

        best: tuple[float, float, float, float] | None = None
        for t0 in self.t0_grid:
            for beta in self.beta_grid:
                for gamma in self.gamma_grid:
                    scores = []
                    valid = True
                    for retention, logits in logits_at_ratios.items():
                        logits = np.asarray(logits, dtype=np.float64)
                        base = t0 + beta * (1.0 - float(retention))
                        feature = self._standardized_feature(logits)
                        temps = self._temperatures_from_feature(base, feature, float(gamma))
                        if np.any(temps <= self.min_temperature):
                            valid = False
                            break
                        transformed = logits[selection_idx] / temps[selection_idx, None]
                        score_value = self._score(transformed, labels[selection_idx])
                        if self.selective_weight > 0:
                            score_value += self.selective_weight * aurc(
                                transformed,
                                labels[selection_idx],
                                score=self.selective_score,
                            )
                        scores.append(score_value)
                    if not valid or not scores:
                        continue
                    score = float(np.mean(scores) + self.gamma_l2 * float(gamma) ** 2)
                    if best is None or score < best[0]:
                        best = (score, float(t0), float(beta), float(gamma))
        if best is None:
            raise RuntimeError("No valid AdaptiveCalibPrune temperature candidate found.")
        self.best_score_, self.t0_star, self.beta_star, self.gamma_star = best
        return self

    def transform(self, logits: np.ndarray, retention: float | None = None) -> np.ndarray:
        if retention is None:
            raise ValueError("AdaptiveCalibPrune.transform requires a retention value.")
        logits = np.asarray(logits, dtype=np.float64)
        temps = self.sample_temperatures(logits, retention)
        return logits / temps[:, None]

    def state_dict(self) -> dict[str, float | str]:
        state = super().state_dict()
        state.update(
            {
                "name": self.name,
                "gamma_star": self.gamma_star,
                "feature": self.feature,
                "temperature_mode": self.temperature_mode,
                "gamma_l2": self.gamma_l2,
                "selective_weight": self.selective_weight,
                "selective_score": self.selective_score,
                "feature_center": self.feature_center_,
                "feature_scale": self.feature_scale_,
                "min_temperature": self.min_temperature,
                "validation_fraction": self.validation_fraction,
                "validation_seed": self.validation_seed,
                "validation_count": self.validation_count_,
                "selection_count": self.selection_count_,
            }
        )
        return state
