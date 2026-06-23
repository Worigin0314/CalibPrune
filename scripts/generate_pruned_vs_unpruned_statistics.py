"""Paired statistics for raw pruned logits versus unpruned logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.statistics import bca_bootstrap_ci, holm_bonferroni, wilcoxon_signed_rank_test
from scripts.generate_gate_figures import ConditionData, load_logits_and_labels, parse_seeds
from scripts.generate_gate_statistics import METRICS, PairedComparison, paired_bootstrap_ci, paired_metric_delta


def retention_text(value: float) -> str:
    return f"{value:g}"


def condition_for_paths(paths: list[Path], label: str) -> ConditionData:
    logits, labels = load_logits_and_labels(paths)
    return ConditionData(label, logits, labels)


def seed_condition(path: Path, label: str) -> ConditionData:
    logits, labels = load_logits_and_labels([path])
    return ConditionData(label, logits, labels)


def build_comparison(
    *,
    logits_dir: Path,
    model: str,
    dataset: str,
    n_test: int,
    seeds: list[int],
    pruner: str,
    retention: float,
) -> PairedComparison:
    none_paths = [logits_dir / f"{model}-{dataset}-test-none-r1-n{n_test}-seed{seed}.npz" for seed in seeds]
    pruned_paths = [
        logits_dir / f"{model}-{dataset}-test-{pruner}-r{retention_text(retention)}-n{n_test}-seed{seed}.npz"
        for seed in seeds
    ]
    unpruned = condition_for_paths(none_paths, "unpruned raw")
    pruned = condition_for_paths(pruned_paths, f"{pruner} r={retention:g} raw")
    seed_pairs = tuple(
        (
            seed_condition(pruned_path, f"{pruner} seed {seed}"),
            seed_condition(none_path, f"none seed {seed}"),
        )
        for seed, pruned_path, none_path in zip(seeds, pruned_paths, none_paths)
    )
    return PairedComparison(
        f"{pruner} r={retention:g} raw - unpruned raw",
        pruned,
        unpruned,
        "positive_ece_means_pruning_increases_drift",
        seed_pairs,
    )


def compute_rows(comparison: PairedComparison, *, n_resamples: int, confidence: float, seed: int, alpha: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
        seed_deltas = np.asarray(
            [paired_metric_delta(left, right, metric_fn) for left, right in comparison.seed_pairs],
            dtype=np.float64,
        )
        wilcoxon = wilcoxon_signed_rank_test(seed_deltas)
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
                "seed_delta_mean": float(np.mean(seed_deltas)),
                "seed_delta_bca_ci_low": seed_bca_low,
                "seed_delta_bca_ci_high": seed_bca_high,
                "wilcoxon_statistic": float(wilcoxon["statistic"]),
                "wilcoxon_p": float(wilcoxon["p_value"]),
                "wilcoxon_n": int(wilcoxon["n"]),
                "interpretation": comparison.interpretation,
            }
        )
    p_values = np.asarray([row["wilcoxon_p"] for row in rows], dtype=np.float64)
    holm = holm_bonferroni(p_values, alpha=alpha)
    for row, adjusted, reject in zip(rows, holm["adjusted_p_values"], holm["reject"]):
        row["wilcoxon_holm_p"] = float(adjusted)
        row["wilcoxon_holm_reject"] = bool(reject)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--pruner", default="fastv")
    parser.add_argument("--retention", type=float, default=0.5)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    comparison = build_comparison(
        logits_dir=PROJECT_ROOT / args.logits_dir,
        model=args.model,
        dataset=args.dataset,
        n_test=args.n_test,
        seeds=args.seeds,
        pruner=args.pruner,
        retention=args.retention,
    )
    rows = compute_rows(
        comparison,
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
