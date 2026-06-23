"""Selective prediction metrics."""

from __future__ import annotations

import numpy as np

from calibprune.metrics.calibration import _as_probabilities, confidence_and_correctness


def selective_confidence(
    values: np.ndarray,
    probabilities: bool | None = None,
    score: str = "max_softmax",
) -> np.ndarray:
    """Return a confidence score for selective prediction.

    Higher values must mean "answer this sample earlier".  ``entropy`` is
    therefore implemented as negative entropy.
    """

    probs = _as_probabilities(values, probabilities)
    if score == "max_softmax":
        return np.max(probs, axis=1)
    if score == "margin":
        if probs.shape[1] < 2:
            return np.ones(probs.shape[0], dtype=np.float64)
        top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]
    if score == "entropy":
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        return -entropy
    raise ValueError("score must be 'max_softmax', 'margin', or 'entropy'.")


def risk_coverage_curve(
    values: np.ndarray,
    labels: np.ndarray,
    probabilities: bool | None = None,
    score: str = "max_softmax",
) -> tuple[np.ndarray, np.ndarray]:
    conf = selective_confidence(values, probabilities=probabilities, score=score)
    _, correct, _ = confidence_and_correctness(values, labels, probabilities)
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    n = len(sorted_correct)
    coverages = np.arange(1, n + 1, dtype=np.float64) / n
    risks = 1.0 - np.cumsum(sorted_correct) / np.arange(1, n + 1)
    return coverages, risks


def aurc(values: np.ndarray, labels: np.ndarray, probabilities: bool | None = None, score: str = "max_softmax") -> float:
    coverages, risks = risk_coverage_curve(values, labels, probabilities, score=score)
    if len(coverages) == 1:
        return float(risks[0])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(risks, coverages))


def selective_accuracy_at_coverages(
    values: np.ndarray,
    labels: np.ndarray,
    coverages: tuple[float, ...] = (0.8, 0.9, 0.95),
    probabilities: bool | None = None,
    score: str = "max_softmax",
) -> dict[str, float]:
    conf = selective_confidence(values, probabilities=probabilities, score=score)
    _, correct, _ = confidence_and_correctness(values, labels, probabilities)
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    n = len(sorted_correct)
    out: dict[str, float] = {}
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out[f"{cov:.2f}"] = float(np.mean(sorted_correct[:k]))
    return out
