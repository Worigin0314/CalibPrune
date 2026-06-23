"""Paired bootstrap statistics for the current real-model experiment gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.calibration import confidence_and_correctness, expected_calibration_error, softmax
from calibprune.metrics.selective import aurc
from calibprune.metrics.statistics import bca_bootstrap_ci, holm_bonferroni, wilcoxon_signed_rank_test
from scripts.generate_gate_figures import ConditionData, parse_seeds, pooled_conditions


MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class PairedComparison:
    name: str
    left: ConditionData
    right: ConditionData
    interpretation: str
    seed_pairs: tuple[tuple[ConditionData, ConditionData], ...] = field(default_factory=tuple)


def parse_retention_values(value: str) -> list[float]:
    retentions = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not retentions:
        raise argparse.ArgumentTypeError("At least one retention is required.")
    return retentions


def retention_text(value: float) -> str:
    return f"{value:g}"


def accuracy_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    _, correct, _ = confidence_and_correctness(logits, labels, probabilities=False)
    return float(np.mean(correct))


def max_softmax_mean_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    conf, _, _ = confidence_and_correctness(logits, labels, probabilities=False)
    return float(np.mean(conf))


METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy_metric,
    "ece": lambda logits, labels: expected_calibration_error(logits, labels, probabilities=False),
    "adaptive_ece": lambda logits, labels: expected_calibration_error(logits, labels, adaptive=True, probabilities=False),
    "aurc": lambda logits, labels: aurc(logits, labels, probabilities=False),
    "max_softmax_mean": max_softmax_mean_metric,
}


def paired_metric_delta(
    left: ConditionData,
    right: ConditionData,
    metric_fn: MetricFn,
    indices: np.ndarray | None = None,
) -> float:
    if not np.array_equal(left.labels, right.labels):
        raise ValueError("Paired conditions must share identical labels and ordering.")
    if left.logits.shape[0] != right.logits.shape[0]:
        raise ValueError("Paired conditions must share the same sample count.")
    if indices is None:
        return float(metric_fn(left.logits, left.labels) - metric_fn(right.logits, right.labels))
    return float(
        metric_fn(left.logits[indices], left.labels[indices])
        - metric_fn(right.logits[indices], right.labels[indices])
    )


def paired_bootstrap_ci(
    left: ConditionData,
    right: ConditionData,
    metric_fn: MetricFn,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")
    n = left.logits.shape[0]
    if n == 0:
        raise ValueError("Cannot bootstrap an empty comparison.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates.append(paired_metric_delta(left, right, metric_fn, indices=idx))
    alpha = 1.0 - confidence
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def make_comparisons(conditions: list[ConditionData]) -> list[PairedComparison]:
    by_label = {condition.label: condition for condition in conditions}
    unpruned = by_label["Unpruned raw"]
    fastv = next(condition for condition in conditions if condition.label.endswith(" raw") and condition.label.startswith("FastV"))
    calibprune = next(condition for condition in conditions if "CalibPrune" in condition.label)
    return [
        PairedComparison("FastV raw - unpruned raw", fastv, unpruned, "positive_ece_means_drift"),
        PairedComparison("CalibPrune - FastV raw", calibprune, fastv, "negative_ece_means_improvement"),
        PairedComparison("CalibPrune - unpruned raw", calibprune, unpruned, "negative_ece_means_below_unpruned"),
    ]


def read_result_payloads(result_globs: list[str]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for pattern in result_globs:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            payload["_path"] = str(path)
            payloads.append(payload)
    return payloads


def find_result_payload(
    payloads: list[dict[str, object]],
    *,
    dataset: str,
    pruner: str,
    calibrator: str,
    retention: float,
    seed: int,
) -> dict[str, object]:
    matches = [
        payload
        for payload in payloads
        if payload.get("dataset") == dataset
        and payload.get("pruner") == pruner
        and payload.get("calibrator") == calibrator
        and int(payload.get("seed", -1)) == seed
        and abs(float(payload.get("retention", -1.0)) - retention) <= 1e-9
    ]
    if not matches:
        raise RuntimeError(
            f"Missing {calibrator} result for dataset={dataset}, pruner={pruner}, "
            f"retention={retention:g}, seed={seed}."
        )
    return matches[-1]


def _adaptive_feature(logits: np.ndarray, feature: str) -> np.ndarray:
    probs = softmax(np.asarray(logits, dtype=np.float64))
    if feature == "confidence":
        return np.max(probs, axis=1)
    if feature == "margin":
        if probs.shape[1] < 2:
            return np.zeros(probs.shape[0], dtype=np.float64)
        top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]
    if feature == "entropy":
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        return -entropy
    raise ValueError(f"Unknown adaptive feature: {feature}")


def transform_with_result_state(logits: np.ndarray, retention: float, payload: dict[str, object]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    state = payload.get("state")
    if not isinstance(state, dict):
        raise RuntimeError(f"Result payload lacks state: {payload.get('_path')}")
    calibrator = payload.get("calibrator")
    if calibrator == "temperature_scaling":
        temperature = float(state["temperature"])
        if temperature <= 0:
            raise ValueError(f"Temperature scaling state must be positive, got {temperature}.")
        return logits / temperature
    if calibrator == "calibprune":
        temperature = float(state["t0_star"]) + float(state["beta_star"]) * (1.0 - float(retention))
        if temperature <= 0:
            raise ValueError(f"CalibPrune temperature must be positive, got {temperature}.")
        return logits / temperature
    if calibrator == "adaptive_calibprune":
        base = float(state["t0_star"]) + float(state["beta_star"]) * (1.0 - float(retention))
        if base <= 0:
            raise ValueError(f"AdaptiveCalibPrune base temperature must be positive, got {base}.")
        feature = _adaptive_feature(logits, str(state.get("feature", "margin")))
        scale = float(state.get("feature_scale", 1.0))
        if scale <= 0:
            scale = 1.0
        z = (feature - float(state.get("feature_center", 0.0))) / scale
        gamma = float(state.get("gamma_star", 0.0))
        mode = str(state.get("temperature_mode", "linear"))
        if mode == "linear":
            temperatures = base + gamma * z
        elif mode == "log":
            temperatures = base * np.exp(gamma * z)
        else:
            raise ValueError(f"Unknown adaptive temperature mode: {mode}")
        if np.any(temperatures <= 0):
            raise ValueError("AdaptiveCalibPrune produced non-positive temperatures.")
        return logits / temperatures[:, None]
    raise ValueError(f"Unsupported calibrator state: {calibrator}")


def condition_for_seed(
    *,
    logits_dir: Path,
    payloads: list[dict[str, object]],
    model: str,
    dataset: str,
    n_test: int,
    seed: int,
    pruner: str,
    retention: float,
    calibrator: str,
) -> ConditionData:
    path = logits_dir / f"{model}-{dataset}-test-{pruner}-r{retention_text(retention)}-n{n_test}-seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    if calibrator != "none":
        payload = find_result_payload(
            payloads,
            dataset=dataset,
            pruner=pruner,
            calibrator=calibrator,
            retention=retention,
            seed=seed,
        )
        logits = transform_with_result_state(logits, retention, payload)
    label = f"{pruner} r={retention:g} {calibrator}"
    return ConditionData(label, logits, labels)


def pooled_condition_for_calibrator(
    *,
    logits_dir: Path,
    payloads: list[dict[str, object]],
    model: str,
    dataset: str,
    n_test: int,
    seeds: list[int],
    pruner: str,
    retention: float,
    calibrator: str,
) -> tuple[ConditionData, tuple[ConditionData, ...]]:
    per_seed = tuple(
        condition_for_seed(
            logits_dir=logits_dir,
            payloads=payloads,
            model=model,
            dataset=dataset,
            n_test=n_test,
            seed=seed,
            pruner=pruner,
            retention=retention,
            calibrator=calibrator,
        )
        for seed in seeds
    )
    logits = np.vstack([condition.logits for condition in per_seed])
    labels = np.concatenate([condition.labels for condition in per_seed])
    return ConditionData(per_seed[0].label, logits, labels), per_seed


def make_adaptive_vs_temperature_comparisons(
    *,
    logits_dir: Path,
    result_globs: list[str],
    model: str,
    dataset: str,
    n_test: int,
    seeds: list[int],
    pruner: str,
    retentions: list[float],
) -> list[PairedComparison]:
    payloads = read_result_payloads(result_globs)
    comparisons: list[PairedComparison] = []
    for retention in retentions:
        adaptive, adaptive_seeds = pooled_condition_for_calibrator(
            logits_dir=logits_dir,
            payloads=payloads,
            model=model,
            dataset=dataset,
            n_test=n_test,
            seeds=seeds,
            pruner=pruner,
            retention=retention,
            calibrator="adaptive_calibprune",
        )
        temperature, temperature_seeds = pooled_condition_for_calibrator(
            logits_dir=logits_dir,
            payloads=payloads,
            model=model,
            dataset=dataset,
            n_test=n_test,
            seeds=seeds,
            pruner=pruner,
            retention=retention,
            calibrator="temperature_scaling",
        )
        comparisons.append(
            PairedComparison(
                f"AdaptiveCalibPrune - temperature scaling ({pruner} r={retention:g})",
                adaptive,
                temperature,
                "negative_ece_means_adaptive_improvement",
                tuple(zip(adaptive_seeds, temperature_seeds)),
            )
        )
    return comparisons


def seed_level_deltas(comparison: PairedComparison, metric_fn: MetricFn) -> np.ndarray:
    return np.asarray(
        [paired_metric_delta(left, right, metric_fn) for left, right in comparison.seed_pairs],
        dtype=np.float64,
    )


def compute_rows(
    comparisons: list[PairedComparison],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        for metric_name, metric_fn in METRICS.items():
            estimate = paired_metric_delta(comparison.left, comparison.right, metric_fn)
            ci_low, ci_high = paired_bootstrap_ci(
                comparison.left,
                comparison.right,
                metric_fn,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
            seed_deltas = seed_level_deltas(comparison, metric_fn) if comparison.seed_pairs else np.array([])
            if seed_deltas.size:
                wilcoxon = wilcoxon_signed_rank_test(seed_deltas)
                wilcoxon_p = float(wilcoxon["p_value"])
                wilcoxon_stat = float(wilcoxon["statistic"])
                wilcoxon_n = int(wilcoxon["n"])
                seed_delta_mean = float(np.mean(seed_deltas))
                if seed_deltas.size >= 2:
                    seed_bca_low, seed_bca_high = bca_bootstrap_ci(
                        seed_deltas,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=seed,
                    )
                else:
                    seed_bca_low = float("nan")
                    seed_bca_high = float("nan")
            else:
                wilcoxon_p = float("nan")
                wilcoxon_stat = float("nan")
                wilcoxon_n = 0
                seed_delta_mean = float("nan")
                seed_bca_low = float("nan")
                seed_bca_high = float("nan")
            rows.append(
                {
                    "comparison": comparison.name,
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_samples": int(comparison.left.labels.shape[0]),
                    "n_resamples": int(n_resamples),
                    "confidence": float(confidence),
                    "seed_delta_mean": seed_delta_mean,
                    "seed_delta_bca_ci_low": seed_bca_low,
                    "seed_delta_bca_ci_high": seed_bca_high,
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p": wilcoxon_p,
                    "wilcoxon_n": wilcoxon_n,
                    "interpretation": comparison.interpretation,
                }
            )

    finite = np.asarray([row["wilcoxon_p"] for row in rows if np.isfinite(float(row["wilcoxon_p"]))], dtype=np.float64)
    holm_iter = iter([])
    if finite.size:
        holm = holm_bonferroni(finite, alpha=alpha)
        holm_iter = iter(zip(holm["adjusted_p_values"], holm["reject"]))
    for row in rows:
        if np.isfinite(float(row["wilcoxon_p"])):
            adjusted, reject = next(holm_iter)
            row["wilcoxon_holm_p"] = float(adjusted)
            row["wilcoxon_holm_reject"] = bool(reject)
        else:
            row["wilcoxon_holm_p"] = float("nan")
            row["wilcoxon_holm_reject"] = False
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--result-glob", action="append", default=None)
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("20260616,20260617,20260618"))
    parser.add_argument("--retention", type=float, default=0.5)
    parser.add_argument("--retentions", type=parse_retention_values, default=None)
    parser.add_argument("--pruner", default="fastv")
    parser.add_argument("--comparison-mode", choices=["default", "adaptive_vs_temperature"], default="default")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--output-csv", default="results/tables/pope_lite_llava_128_512_paired_bootstrap_ci.csv")
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    result_globs = args.result_glob or ["results/raw/pope_lite_llava_128_512*/*.json"]

    if args.comparison_mode == "adaptive_vs_temperature":
        comparisons = make_adaptive_vs_temperature_comparisons(
            logits_dir=PROJECT_ROOT / args.logits_dir,
            result_globs=result_globs,
            model=args.model,
            dataset=args.dataset,
            n_test=args.n_test,
            seeds=args.seeds,
            pruner=args.pruner,
            retentions=args.retentions or [args.retention],
        )
    else:
        conditions = pooled_conditions(
            logits_dir=PROJECT_ROOT / args.logits_dir,
            result_globs=result_globs,
            model=args.model,
            dataset=args.dataset,
            n_test=args.n_test,
            seeds=args.seeds,
            retention=args.retention,
        )
        comparisons = make_comparisons(conditions)

    rows = compute_rows(
        comparisons,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        seed=args.seed,
        alpha=args.alpha,
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



