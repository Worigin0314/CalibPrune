"""Statistical helpers for paper tables."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def percentile_bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260616,
) -> tuple[float, float]:
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("values must contain at least one observation.")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, values.shape[0], size=values.shape[0])
        stats.append(float(statistic(values[idx])))
    alpha = 1.0 - confidence
    return (
        float(np.quantile(stats, alpha / 2.0)),
        float(np.quantile(stats, 1.0 - alpha / 2.0)),
    )


def bca_bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260616,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap confidence interval.

    This implementation is intended for one-dimensional paired deltas or scalar
    per-sample quantities used in experiment sanity checks. For non-smooth
    metrics such as binned ECE, the paired table script still reports its own
    paired bootstrap procedure.
    """

    values = np.asarray(values)
    if values.ndim == 0 or values.shape[0] < 2:
        raise ValueError("values must contain at least two observations for BCa.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(values))
    boot = np.empty(n_resamples, dtype=np.float64)
    n = values.shape[0]
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(statistic(values[idx]))

    dist = NormalDist()
    eps = 1.0 / (2.0 * n_resamples)
    prop_less = float(np.mean(boot < theta_hat))
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = dist.inv_cdf(prop_less)

    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = float(statistic(np.delete(values, i, axis=0)))
    jack_mean = float(np.mean(jack))
    centered = jack_mean - jack
    denom = 6.0 * float(np.sum(centered**2) ** 1.5)
    acceleration = 0.0 if denom == 0.0 else float(np.sum(centered**3) / denom)

    alpha = 1.0 - confidence

    def adjusted_alpha(raw_alpha: float) -> float:
        z_alpha = dist.inv_cdf(raw_alpha)
        denom_inner = 1.0 - acceleration * (z0 + z_alpha)
        if denom_inner == 0.0:
            return raw_alpha
        return dist.cdf(z0 + (z0 + z_alpha) / denom_inner)

    low_alpha = min(max(adjusted_alpha(alpha / 2.0), 0.0), 1.0)
    high_alpha = min(max(adjusted_alpha(1.0 - alpha / 2.0), 0.0), 1.0)
    return (float(np.quantile(boot, low_alpha)), float(np.quantile(boot, high_alpha)))


def paired_delta(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("paired arrays must have the same shape.")
    return values_a - values_b


def _rank_absolute_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def wilcoxon_signed_rank_test(
    deltas: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | int]:
    """Exact Wilcoxon signed-rank test for paired scalar deltas."""

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    deltas = deltas[deltas != 0.0]
    if deltas.shape[0] == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}

    ranks = _rank_absolute_values(np.abs(deltas))
    positive_sum = float(np.sum(ranks[deltas > 0.0]))
    total = float(np.sum(ranks))
    all_sums = np.array([0.0], dtype=np.float64)
    for rank in ranks:
        all_sums = np.concatenate([all_sums, all_sums + float(rank)])

    if alternative == "less":
        p_value = float(np.mean(all_sums <= positive_sum + 1e-12))
        statistic = positive_sum
    elif alternative == "greater":
        p_value = float(np.mean(all_sums >= positive_sum - 1e-12))
        statistic = positive_sum
    else:
        observed = min(positive_sum, total - positive_sum)
        folded = np.minimum(all_sums, total - all_sums)
        p_value = float(np.mean(folded <= observed + 1e-12))
        statistic = observed

    return {
        "statistic": float(statistic),
        "p_value": min(1.0, max(0.0, p_value)),
        "n": int(deltas.shape[0]),
    }

def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Holm-Bonferroni step-down multiple-comparison correction."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or p_values.shape[0] == 0:
        raise ValueError("p_values must be a non-empty 1D array.")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p_values must lie in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    n = p_values.shape[0]
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=np.float64)
    reject = np.zeros(n, dtype=bool)
    running_max = 0.0
    still_rejecting = True
    for rank, idx in enumerate(order):
        multiplier = n - rank
        running_max = max(running_max, min(1.0, multiplier * float(p_values[idx])))
        adjusted[idx] = running_max
        if still_rejecting and p_values[idx] <= alpha / multiplier:
            reject[idx] = True
        else:
            still_rejecting = False
    return {"adjusted_p_values": adjusted, "reject": reject}

