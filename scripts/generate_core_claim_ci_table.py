"""Paper-facing bootstrap CIs for core calibration-drift claims.

The manuscript uses this table to avoid scattering headline statistical
intervals across several diagnostic outputs. Every row is recomputed from saved
answer-level logits and, for calibrated conditions, saved calibrator states.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.calibration import confidence_and_correctness, expected_calibration_error
from calibprune.metrics.selective import aurc
from scripts.generate_gate_statistics import read_result_payloads, transform_with_result_state
from scripts.generate_logit_diagnostics import entropy, residual_relative_norm


@dataclass(frozen=True)
class Condition:
    label: str
    logits: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    setting: str
    left: Condition
    right: Condition
    metric_name: str
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
    direction: str
    interpretation: str


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def scalar_text(value: float) -> str:
    return f"{value:g}"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    return np.asarray(payload["logits"], dtype=np.float64), np.asarray(payload["labels"], dtype=np.int64)


def raw_condition(*, logits_dir: Path, model: str, dataset: str, n_test: int, seeds: list[int], pruner: str, retention: float) -> Condition:
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for seed in seeds:
        path = logits_dir / f"{model}-{dataset}-test-{pruner}-r{scalar_text(retention)}-n{n_test}-seed{seed}.npz"
        logits, labels = load_npz(path)
        logits_parts.append(logits)
        label_parts.append(labels)
    return Condition(
        label=f"{model}/{dataset}/{pruner}/r{scalar_text(retention)} raw",
        logits=np.vstack(logits_parts),
        labels=np.concatenate(label_parts),
    )


def calibrated_condition(
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
    run_id_contains: str | None = None,
) -> Condition:
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for seed in seeds:
        raw_path = logits_dir / f"{model}-{dataset}-test-{pruner}-r{scalar_text(retention)}-n{n_test}-seed{seed}.npz"
        logits, labels = load_npz(raw_path)
        payload = find_payload(
            payloads,
            dataset=dataset,
            pruner=pruner,
            retention=retention,
            calibrator=calibrator,
            seed=seed,
            run_id_contains=run_id_contains,
        )
        logits_parts.append(transform_with_result_state(logits, retention, payload))
        label_parts.append(labels)
    return Condition(
        label=f"{model}/{dataset}/{pruner}/r{scalar_text(retention)} {calibrator}",
        logits=np.vstack(logits_parts),
        labels=np.concatenate(label_parts),
    )


def find_payload(
    payloads: list[dict[str, object]],
    *,
    dataset: str,
    pruner: str,
    retention: float,
    calibrator: str,
    seed: int,
    run_id_contains: str | None = None,
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    for payload in payloads:
        if payload.get("dataset") != dataset:
            continue
        if payload.get("pruner") != pruner:
            continue
        if payload.get("calibrator") != calibrator:
            continue
        if int(payload.get("seed", -1)) != seed:
            continue
        if abs(float(payload.get("retention", -1.0)) - retention) > 1e-9:
            continue
        if run_id_contains and run_id_contains not in str(payload.get("run_id", "")):
            continue
        matches.append(payload)
    if not matches:
        detail = f"dataset={dataset}, pruner={pruner}, retention={retention:g}, calibrator={calibrator}, seed={seed}"
        if run_id_contains:
            detail += f", run_id contains {run_id_contains!r}"
        raise RuntimeError(f"Missing calibrator payload for {detail}.")
    return matches[-1]


def assert_paired(left: Condition, right: Condition) -> None:
    if left.logits.shape != right.logits.shape:
        raise ValueError(f"Shape mismatch: {left.label} {left.logits.shape} vs {right.label} {right.logits.shape}")
    if not np.array_equal(left.labels, right.labels):
        raise ValueError(f"Label/order mismatch: {left.label} vs {right.label}")


def accuracy_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    _, correct, _ = confidence_and_correctness(logits, labels, probabilities=False)
    return float(np.mean(correct))


def ece_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(expected_calibration_error(logits, labels, probabilities=False))


def aurc_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(aurc(logits, labels, probabilities=False))


def max_softmax_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    conf, _, _ = confidence_and_correctness(logits, labels, probabilities=False)
    return float(np.mean(conf))


def entropy_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(entropy(logits)))


def scalar_residual_metric(logits: np.ndarray, labels: np.ndarray) -> float:
    # This metric is only meaningful as left-vs-right; the dummy definition keeps
    # the table machinery uniform, and paired_delta handles it specially.
    del labels
    return float(np.mean(logits))


def paired_delta(left: Condition, right: Condition, metric_name: str, metric_fn: Callable[[np.ndarray, np.ndarray], float], indices: np.ndarray | None = None) -> float:
    assert_paired(left, right)
    l_logits = left.logits if indices is None else left.logits[indices]
    r_logits = right.logits if indices is None else right.logits[indices]
    labels = left.labels if indices is None else left.labels[indices]
    if metric_name == "scalar_fit_residual":
        return float(residual_relative_norm(l_logits, r_logits))
    return float(metric_fn(l_logits, labels) - metric_fn(r_logits, labels))


def bootstrap_ci(left: Condition, right: Condition, metric_name: str, metric_fn: Callable[[np.ndarray, np.ndarray], float], *, n_resamples: int, confidence: float, seed: int) -> tuple[float, float]:
    assert_paired(left, right)
    rng = np.random.default_rng(seed)
    n = left.logits.shape[0]
    estimates = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.integers(0, n, size=n)
        estimates[idx] = paired_delta(left, right, metric_name, metric_fn, sample)
    alpha = 1.0 - confidence
    return float(np.quantile(estimates, alpha / 2.0)), float(np.quantile(estimates, 1.0 - alpha / 2.0))


def format_interval(value: float, low: float, high: float) -> str:
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def build_claims(*, logits_dir: Path, payloads: list[dict[str, object]]) -> list[ClaimSpec]:
    llava_seeds = [20260616, 20260617, 20260618]
    qwen_seeds = [20260616, 20260617, 20260618, 20260619]

    llava_none_512 = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="pope", n_test=512, seeds=llava_seeds, pruner="none", retention=1.0)
    llava_fastv_512 = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="pope", n_test=512, seeds=llava_seeds, pruner="fastv", retention=0.5)
    qwen_none_512 = raw_condition(logits_dir=logits_dir, model="qwen2vl_2b", dataset="pope", n_test=512, seeds=qwen_seeds, pruner="none", retention=1.0)
    qwen_fastv_512 = raw_condition(logits_dir=logits_dir, model="qwen2vl_2b", dataset="pope", n_test=512, seeds=qwen_seeds, pruner="fastv", retention=0.5)

    llava_none_705 = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="pope", n_test=705, seeds=llava_seeds, pruner="none", retention=1.0)
    sparse_705 = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="pope", n_test=705, seeds=llava_seeds, pruner="sparsevlm", retention=0.5)
    vtw_705 = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="pope", n_test=705, seeds=llava_seeds, pruner="vtw", retention=0.5)

    llava_fastv_ts_512 = calibrated_condition(
        logits_dir=logits_dir,
        payloads=payloads,
        model="llava15_7b_4bit",
        dataset="pope",
        n_test=512,
        seeds=llava_seeds,
        pruner="fastv",
        retention=0.5,
        calibrator="temperature_scaling",
    )

    aok_raw = raw_condition(logits_dir=logits_dir, model="llava15_7b_4bit", dataset="aokvqa", n_test=1145, seeds=[20260616], pruner="fastv", retention=0.5)
    aok_adaptive = calibrated_condition(
        logits_dir=logits_dir,
        payloads=payloads,
        model="llava15_7b_4bit",
        dataset="aokvqa",
        n_test=1145,
        seeds=[20260616],
        pruner="fastv",
        retention=0.5,
        calibrator="adaptive_calibprune",
        run_id_contains="adaptive_log_margin_r05",
    )

    return [
        ClaimSpec(
            "llava_fastv_confidence_sharpening",
            "LLaVA/POPE FastV r=0.5, 3 seeds",
            llava_fastv_512,
            llava_none_512,
            "max_softmax_mean_delta",
            max_softmax_metric,
            "positive = confidence sharpening",
            "FastV increases max-softmax confidence on LLaVA.",
        ),
        ClaimSpec(
            "llava_fastv_entropy_drop",
            "LLaVA/POPE FastV r=0.5, 3 seeds",
            llava_fastv_512,
            llava_none_512,
            "entropy_mean_delta",
            entropy_metric,
            "negative = confidence sharpening",
            "Entropy decreases under the same LLaVA FastV gate.",
        ),
        ClaimSpec(
            "llava_fastv_scalar_residual",
            "LLaVA/POPE FastV r=0.5, 3 seeds",
            llava_fastv_512,
            llava_none_512,
            "scalar_fit_residual",
            scalar_residual_metric,
            "lower = more scale-like",
            "Residual quantifies non-scalar perturbation after the best pooled scale fit.",
        ),
        ClaimSpec(
            "qwen_fastv_confidence_flattening",
            "Qwen2-VL/POPE FastV r=0.5, 4 seeds",
            qwen_fastv_512,
            qwen_none_512,
            "max_softmax_mean_delta",
            max_softmax_metric,
            "negative = confidence flattening",
            "Qwen2-VL reverses the LLaVA confidence-sharpening direction.",
        ),
        ClaimSpec(
            "qwen_fastv_scalar_residual",
            "Qwen2-VL/POPE FastV r=0.5, 4 seeds",
            qwen_fastv_512,
            qwen_none_512,
            "scalar_fit_residual",
            scalar_residual_metric,
            "lower = more scale-like",
            "Qwen2-VL FastV has a large non-scalar residual.",
        ),
        ClaimSpec(
            "sparsevlm_aurc_improvement",
            "LLaVA/POPE layer-8 SparseVLM r=0.5, 3 seeds",
            sparse_705,
            llava_none_705,
            "aurc_delta",
            aurc_metric,
            "negative = improved selective reliability",
            "SparseVLM improves AURC despite inconclusive ECE movement.",
        ),
        ClaimSpec(
            "vtw_aurc_improvement",
            "LLaVA/POPE layer-8 VTW r=0.5, 3 seeds",
            vtw_705,
            llava_none_705,
            "aurc_delta",
            aurc_metric,
            "negative = improved selective reliability",
            "VTW improves AURC despite inconclusive ECE movement.",
        ),
        ClaimSpec(
            "ts_ece_improvement",
            "LLaVA/POPE FastV r=0.5 + temperature scaling, 3 seeds",
            llava_fastv_ts_512,
            llava_fastv_512,
            "ece_delta",
            ece_metric,
            "negative = better calibration",
            "Temperature scaling lowers ECE for pruned FastV logits.",
        ),
        ClaimSpec(
            "ts_aurc_worsening",
            "LLaVA/POPE FastV r=0.5 + temperature scaling, 3 seeds",
            llava_fastv_ts_512,
            llava_fastv_512,
            "aurc_delta",
            aurc_metric,
            "positive = worse selective reliability",
            "The same temperature scaling transform worsens AURC.",
        ),
        ClaimSpec(
            "aokvqa_adaptive_failure",
            "LLaVA/A-OKVQA FastV r=0.5 + unguarded AdaptiveCalibPrune, full validation",
            aok_adaptive,
            aok_raw,
            "ece_delta",
            ece_metric,
            "positive = worse calibration",
            "Unguarded adaptive calibration overfits on the A-OKVQA gate.",
        ),
    ]


def compute_rows(claims: list[ClaimSpec], *, n_resamples: int, confidence: float, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, claim in enumerate(claims):
        estimate = paired_delta(claim.left, claim.right, claim.metric_name, claim.metric_fn)
        ci_low, ci_high = bootstrap_ci(
            claim.left,
            claim.right,
            claim.metric_name,
            claim.metric_fn,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + offset,
        )
        rows.append(
            {
                "claim_id": claim.claim_id,
                "setting": claim.setting,
                "metric": claim.metric_name,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "estimate_ci": format_interval(estimate, ci_low, ci_high),
                "n_samples": int(claim.left.labels.shape[0]),
                "confidence": float(confidence),
                "n_resamples": int(n_resamples),
                "direction": claim.direction,
                "interpretation": claim.interpretation,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--result-glob", action="append", default=None)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-csv", default="results/tables/core_claim_bootstrap_ci.csv")
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    result_globs = args.result_glob or [
        "results/raw/pope_lite_llava_128_512*/*.json",
        "results/raw/aokvqa_full_validation_llava_fastv_r05_calibrated/*.json",
    ]
    payloads = read_result_payloads(result_globs)
    claims = build_claims(logits_dir=PROJECT_ROOT / args.logits_dir, payloads=payloads)
    rows = compute_rows(claims, n_resamples=args.n_resamples, confidence=args.confidence, seed=args.seed)
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = PROJECT_ROOT / args.output_md if args.output_md else csv_path.with_suffix(".md")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(df)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
