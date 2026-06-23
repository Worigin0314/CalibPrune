"""Reliability and risk-coverage triptych for raw pruning plus a calibrator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.calibration import confidence_and_correctness, metrics_from_logits, reliability_diagram
from calibprune.metrics.selective import aurc, risk_coverage_curve
from scripts.generate_gate_figures import save_figure
from scripts.generate_gate_statistics import read_result_payloads, transform_with_result_state


@dataclass(frozen=True)
class Condition:
    label: str
    logits: np.ndarray
    labels: np.ndarray


def parse_csv_int(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scalar_text(value: float) -> str:
    return f"{value:g}"


def load_logits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    return np.asarray(payload["logits"], dtype=np.float64), np.asarray(payload["labels"], dtype=np.int64)


def find_calibrator_payload(
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
            f"Missing {calibrator} result for dataset={dataset}, pruner={pruner}, retention={retention:g}, seed={seed}."
        )
    return matches[-1]


def pooled_conditions(
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
) -> list[Condition]:
    none_parts: list[np.ndarray] = []
    pruned_parts: list[np.ndarray] = []
    calibrated_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for seed in seeds:
        none_path = logits_dir / f"{model}-{dataset}-test-none-r1-n{n_test}-seed{seed}.npz"
        pruned_path = logits_dir / f"{model}-{dataset}-test-{pruner}-r{scalar_text(retention)}-n{n_test}-seed{seed}.npz"
        none_logits, none_labels = load_logits(none_path)
        pruned_logits, pruned_labels = load_logits(pruned_path)
        if not np.array_equal(none_labels, pruned_labels):
            raise ValueError(f"Label/order mismatch for seed {seed}.")
        payload = find_calibrator_payload(
            payloads,
            dataset=dataset,
            pruner=pruner,
            calibrator=calibrator,
            retention=retention,
            seed=seed,
        )
        calibrated = transform_with_result_state(pruned_logits, retention, payload)
        none_parts.append(none_logits)
        pruned_parts.append(pruned_logits)
        calibrated_parts.append(calibrated)
        label_parts.append(none_labels)
    labels = np.concatenate(label_parts)
    return [
        Condition("Unpruned raw", np.vstack(none_parts), labels),
        Condition(f"{pruner} r={retention:g} raw", np.vstack(pruned_parts), labels),
        Condition(f"{pruner} r={retention:g} + {calibrator}", np.vstack(calibrated_parts), labels),
    ]


def plot_reliability_triptych(conditions: list[Condition], output_dir: Path, stem: str) -> None:
    fig, axes = plt.subplots(1, len(conditions), figsize=(10.5, 3.3), sharex=True, sharey=True)
    n_bins = 15
    centers = (np.arange(n_bins) + 0.5) / n_bins
    width = 0.88 / n_bins
    for ax, condition in zip(axes, conditions):
        rel = reliability_diagram(condition.logits, condition.labels, n_bins=n_bins, probabilities=False)
        counts = np.asarray(rel.bin_count)
        ax.bar(centers, rel.bin_acc, width=width, color="#72B7B2", alpha=0.78, label="Accuracy")
        nonempty = counts > 0
        ax.plot(centers[nonempty], np.asarray(rel.bin_conf)[nonempty], "o-", color="#E45756", linewidth=1.8, label="Confidence")
        ax.plot([0, 1], [0, 1], "--", color="#8C8C8C", linewidth=1.0)
        ax.set_title(condition.label)
        ax.set_xlabel("Confidence bin")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Accuracy / confidence")
    axes[-1].legend(loc="lower right", fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{stem}_reliability",
        caption="Reliability triptych for unpruned, pruned raw, and calibrated pruned predictions.",
    )


def plot_risk_coverage(conditions: list[Condition], output_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for condition in conditions:
        coverage, risk = risk_coverage_curve(condition.logits, condition.labels, probabilities=False)
        ax.plot(coverage, risk, linewidth=2.0, label=condition.label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Risk-coverage curves")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{stem}_risk_coverage",
        caption="Risk-coverage curves for unpruned, pruned raw, and calibrated pruned predictions.",
    )


def write_summary(conditions: list[Condition], path: Path) -> None:
    rows = []
    for condition in conditions:
        metrics = metrics_from_logits(condition.logits, condition.labels)
        conf, _, _ = confidence_and_correctness(condition.logits, condition.labels, probabilities=False)
        rows.append(
            {
                "condition": condition.label,
                "n_samples": int(condition.labels.shape[0]),
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
                "adaptive_ece": metrics["adaptive_ece"],
                "aurc": aurc(condition.logits, condition.labels, probabilities=False),
                "max_softmax_mean": float(np.mean(conf)),
                "max_softmax_p95": float(np.quantile(conf, 0.95)),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--result-glob", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--seeds", type=parse_csv_int, required=True)
    parser.add_argument("--pruner", required=True)
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--calibrator", default="temperature_scaling")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    payloads = read_result_payloads(args.result_glob)
    conditions = pooled_conditions(
        logits_dir=PROJECT_ROOT / args.logits_dir,
        payloads=payloads,
        model=args.model,
        dataset=args.dataset,
        n_test=args.n_test,
        seeds=args.seeds,
        pruner=args.pruner,
        retention=args.retention,
        calibrator=args.calibrator,
    )
    output_dir = PROJECT_ROOT / args.output_dir
    plot_reliability_triptych(conditions, output_dir, args.stem)
    plot_risk_coverage(conditions, output_dir, args.stem)
    write_summary(conditions, PROJECT_ROOT / args.summary_csv)
    written = sorted(str(path.relative_to(PROJECT_ROOT)) for path in output_dir.glob(f"{args.stem}_*"))
    print(json.dumps({"figures": written, "summary_csv": args.summary_csv}, indent=2))


if __name__ == "__main__":
    main()