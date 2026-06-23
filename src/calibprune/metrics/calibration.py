"""Calibration metrics used by CalibPrune experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReliabilityBins:
    bin_conf: list[float]
    bin_acc: list[float]
    bin_count: list[int]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _as_probabilities(values: np.ndarray, probabilities: bool | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a 2D array with shape [n_samples, n_classes].")
    if probabilities is True:
        return values
    if probabilities is False:
        return softmax(values)
    row_sums = np.sum(values, axis=1)
    is_prob = np.all(values >= -1e-8) and np.all(values <= 1 + 1e-8)
    if is_prob and np.allclose(row_sums, 1.0, atol=1e-5):
        return values
    return softmax(values)


def confidence_and_correctness(
    values: np.ndarray,
    labels: np.ndarray,
    probabilities: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = _as_probabilities(values, probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("values and labels must have the same first dimension.")
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = preds == labels
    return conf, correct.astype(np.float64), preds


def reliability_diagram(
    values: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
    probabilities: bool | None = None,
) -> ReliabilityBins:
    conf, correct, _ = confidence_and_correctness(values, labels, probabilities)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if adaptive:
        order = np.argsort(conf)
        chunks = np.array_split(order, n_bins)
        groups = [chunk for chunk in chunks if len(chunk)]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        groups = []
        for i in range(n_bins):
            left = edges[i]
            right = edges[i + 1]
            if i == n_bins - 1:
                mask = (conf >= left) & (conf <= right)
            else:
                mask = (conf >= left) & (conf < right)
            groups.append(np.flatnonzero(mask))

    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_count: list[int] = []
    for idx in groups:
        if len(idx) == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            bin_count.append(0)
            continue
        bin_conf.append(float(np.mean(conf[idx])))
        bin_acc.append(float(np.mean(correct[idx])))
        bin_count.append(int(len(idx)))
    return ReliabilityBins(bin_conf=bin_conf, bin_acc=bin_acc, bin_count=bin_count)


def expected_calibration_error(
    values: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
    probabilities: bool | None = None,
) -> float:
    bins = reliability_diagram(values, labels, n_bins, adaptive, probabilities)
    n = max(1, int(np.sum(bins.bin_count)))
    total = 0.0
    for conf, acc, count in zip(bins.bin_conf, bins.bin_acc, bins.bin_count):
        total += (count / n) * abs(acc - conf)
    return float(total)


def maximum_calibration_error(
    values: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
    probabilities: bool | None = None,
) -> float:
    bins = reliability_diagram(values, labels, n_bins, adaptive, probabilities)
    errors = [abs(acc - conf) for acc, conf, count in zip(bins.bin_acc, bins.bin_conf, bins.bin_count) if count]
    return float(max(errors) if errors else 0.0)


def brier_score(values: np.ndarray, labels: np.ndarray, probabilities: bool | None = None) -> float:
    probs = _as_probabilities(values, probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(
    values: np.ndarray,
    labels: np.ndarray,
    probabilities: bool | None = None,
) -> float:
    probs = _as_probabilities(values, probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    chosen = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    conf, correct, _ = confidence_and_correctness(logits, labels, probabilities=False)
    rel = reliability_diagram(logits, labels, n_bins=n_bins, probabilities=False)
    return {
        "accuracy": float(np.mean(correct)),
        "ece": expected_calibration_error(logits, labels, n_bins=n_bins, probabilities=False),
        "adaptive_ece": expected_calibration_error(logits, labels, n_bins=n_bins, adaptive=True, probabilities=False),
        "mce": maximum_calibration_error(logits, labels, n_bins=n_bins, probabilities=False),
        "brier": brier_score(logits, labels, probabilities=False),
        "nll": negative_log_likelihood(logits, labels, probabilities=False),
        "reliability": {
            "bin_conf": rel.bin_conf,
            "bin_acc": rel.bin_acc,
            "bin_count": rel.bin_count,
        },
        "max_softmax_mean": float(np.mean(conf)),
        "max_softmax_p95": float(np.quantile(conf, 0.95)),
    }

