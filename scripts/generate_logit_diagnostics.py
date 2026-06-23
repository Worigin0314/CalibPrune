"""Logit-scale and confidence-shift diagnostics from saved answer logits.

This script supports the H2 mechanism check: whether pruning shifts the
max-softmax distribution and behaves approximately like a smooth logit-scale
change plus a residual perturbation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.calibration import softmax


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def scalar_text(value: float) -> str:
    return f"{value:g}"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    return np.asarray(payload["logits"], dtype=np.float64), np.asarray(payload["labels"], dtype=np.int64)


def centered(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return logits - np.mean(logits, axis=1, keepdims=True)


def confidence(logits: np.ndarray) -> np.ndarray:
    return np.max(softmax(logits), axis=1)


def entropy(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def margin(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    if probs.shape[1] < 2:
        return np.zeros(probs.shape[0], dtype=np.float64)
    top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def logit_l2(logits: np.ndarray) -> np.ndarray:
    return np.linalg.norm(centered(logits), axis=1)


def pooled_alpha(pruned: np.ndarray, unpruned: np.ndarray) -> float:
    p = centered(pruned).reshape(-1)
    u = centered(unpruned).reshape(-1)
    denom = float(np.dot(u, u))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(p, u) / denom)


def residual_relative_norm(pruned: np.ndarray, unpruned: np.ndarray) -> float:
    alpha = pooled_alpha(pruned, unpruned)
    if not np.isfinite(alpha):
        return float("nan")
    p = centered(pruned)
    u = centered(unpruned)
    residual = p - alpha * u
    return float(np.linalg.norm(residual) / max(np.linalg.norm(p), 1e-12))


def metric_value(pruned: np.ndarray, unpruned: np.ndarray, metric: str) -> float:
    if metric == "max_softmax_mean_shift":
        return float(np.mean(confidence(pruned) - confidence(unpruned)))
    if metric == "max_softmax_median_shift":
        return float(np.median(confidence(pruned) - confidence(unpruned)))
    if metric == "max_softmax_q95_shift":
        return float(np.quantile(confidence(pruned), 0.95) - np.quantile(confidence(unpruned), 0.95))
    if metric == "entropy_mean_shift":
        return float(np.mean(entropy(pruned) - entropy(unpruned)))
    if metric == "margin_mean_shift":
        return float(np.mean(margin(pruned) - margin(unpruned)))
    if metric == "logit_l2_mean_shift":
        return float(np.mean(logit_l2(pruned) - logit_l2(unpruned)))
    if metric == "logit_l2_ratio_mean_minus_1":
        return float(np.mean(logit_l2(pruned) / np.clip(logit_l2(unpruned), 1e-12, None)) - 1.0)
    if metric == "alpha_pooled_minus_1":
        return float(pooled_alpha(pruned, unpruned) - 1.0)
    if metric == "scale_residual_rel_norm":
        return residual_relative_norm(pruned, unpruned)
    raise ValueError(f"Unknown metric: {metric}")


METRICS = [
    "max_softmax_mean_shift",
    "max_softmax_median_shift",
    "max_softmax_q95_shift",
    "entropy_mean_shift",
    "margin_mean_shift",
    "logit_l2_mean_shift",
    "logit_l2_ratio_mean_minus_1",
    "alpha_pooled_minus_1",
    "scale_residual_rel_norm",
]


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def paired_bootstrap_ci(
    pruned: np.ndarray,
    unpruned: np.ndarray,
    metric: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if pruned.shape != unpruned.shape:
        raise ValueError("Paired logits must have identical shape.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    rng = np.random.default_rng(seed)
    n = pruned.shape[0]
    estimates = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.integers(0, n, size=n)
        estimates[idx] = metric_value(pruned[sample], unpruned[sample], metric)
    alpha = 1.0 - confidence_level
    return (float(np.quantile(estimates, alpha / 2.0)), float(np.quantile(estimates, 1.0 - alpha / 2.0)))


def build_rows(
    *,
    logits_dir: Path,
    model: str,
    dataset: str,
    n_test: int,
    seeds: list[int],
    pruners: list[str],
    retentions: list[float],
    n_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pruner in pruners:
        for retention in retentions:
            pooled_pruned: list[np.ndarray] = []
            pooled_unpruned: list[np.ndarray] = []
            seed_metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
            used_seeds: list[int] = []
            for seed in seeds:
                none_path = logits_dir / f"{model}-{dataset}-test-none-r1-n{n_test}-seed{seed}.npz"
                pruned_path = logits_dir / f"{model}-{dataset}-test-{pruner}-r{scalar_text(retention)}-n{n_test}-seed{seed}.npz"
                unpruned_logits, unpruned_labels = load_npz(none_path)
                pruned_logits, pruned_labels = load_npz(pruned_path)
                if not np.array_equal(unpruned_labels, pruned_labels):
                    raise ValueError(f"Labels/order mismatch for {pruned_path.name} vs {none_path.name}.")
                if unpruned_logits.shape != pruned_logits.shape:
                    raise ValueError(f"Shape mismatch for {pruned_path.name} vs {none_path.name}.")
                pooled_unpruned.append(unpruned_logits)
                pooled_pruned.append(pruned_logits)
                used_seeds.append(seed)
                for metric in METRICS:
                    seed_metric_values[metric].append(metric_value(pruned_logits, unpruned_logits, metric))

            pruned_all = np.vstack(pooled_pruned)
            unpruned_all = np.vstack(pooled_unpruned)
            for metric in METRICS:
                estimate = metric_value(pruned_all, unpruned_all, metric)
                ci_low, ci_high = paired_bootstrap_ci(
                    pruned_all,
                    unpruned_all,
                    metric,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed,
                )
                seed_values = np.asarray(seed_metric_values[metric], dtype=np.float64)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "pruner": pruner,
                        "retention": retention,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_samples": int(pruned_all.shape[0]),
                        "n_seeds": int(len(used_seeds)),
                        "seeds": ",".join(str(item) for item in used_seeds),
                        "n_resamples": int(n_resamples),
                        "confidence": float(confidence_level),
                        "seed_delta_mean": float(np.mean(seed_values)),
                        "seed_delta_std": float(np.std(seed_values, ddof=1)) if seed_values.size > 1 else 0.0,
                        "seed_delta_min": float(np.min(seed_values)),
                        "seed_delta_max": float(np.max(seed_values)),
                        "one_minus_retention": float(1.0 - retention),
                        "interpretation": interpretation(metric),
                    }
                )
    return rows


def interpretation(metric: str) -> str:
    if metric.startswith("max_softmax") or metric.startswith("margin"):
        return "positive means pruning sharpens confidence"
    if metric.startswith("entropy"):
        return "negative means pruning sharpens confidence"
    if metric.startswith("logit_l2") or metric.startswith("alpha"):
        return "positive means pruned logits have larger scale"
    if metric == "scale_residual_rel_norm":
        return "smaller means scalar-scale model explains pruning better"
    return "paired pruned minus unpruned diagnostic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--seeds", type=parse_int_csv, required=True)
    parser.add_argument("--pruners", type=parse_csv, required=True)
    parser.add_argument("--retentions", type=parse_float_csv, required=True)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    rows = build_rows(
        logits_dir=PROJECT_ROOT / args.logits_dir,
        model=args.model,
        dataset=args.dataset,
        n_test=args.n_test,
        seeds=args.seeds,
        pruners=args.pruners,
        retentions=args.retentions,
        n_resamples=args.n_resamples,
        confidence_level=args.confidence,
        bootstrap_seed=args.seed,
    )
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = PROJECT_ROOT / args.output_md if args.output_md else csv_path.with_suffix(".md")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(df)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()