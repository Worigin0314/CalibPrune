"""Generate reproducible figures for the current real-model experiment gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.calibration import (
    confidence_and_correctness,
    metrics_from_logits,
    reliability_diagram,
)
from calibprune.metrics.selective import aurc, risk_coverage_curve


@dataclass(frozen=True)
class ConditionData:
    label: str
    logits: np.ndarray
    labels: np.ndarray


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_logits_and_labels(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in paths:
        payload = load_npz(path)
        logits.append(np.asarray(payload["logits"], dtype=np.float64))
        labels.append(np.asarray(payload["labels"], dtype=np.int64))
    return np.vstack(logits), np.concatenate(labels)


def apply_calibprune_state(logits: np.ndarray, retention: float, state: dict[str, float]) -> np.ndarray:
    temperature = float(state["t0_star"]) + float(state["beta_star"]) * (1.0 - float(retention))
    if temperature <= 0:
        raise ValueError(f"CalibPrune temperature must be positive, got {temperature}.")
    return np.asarray(logits, dtype=np.float64) / temperature


def find_calibprune_states(result_globs: list[str], *, dataset: str, retention: float) -> dict[int, dict[str, float]]:
    states: dict[int, dict[str, float]] = {}
    for pattern in result_globs:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("dataset") != dataset:
                continue
            if payload.get("calibrator") != "calibprune":
                continue
            if payload.get("pruner") != "fastv":
                continue
            if abs(float(payload.get("retention", -1.0)) - retention) > 1e-9:
                continue
            states[int(payload["seed"])] = payload["state"]
    return states


def pooled_conditions(
    *,
    logits_dir: Path,
    result_globs: list[str],
    model: str,
    dataset: str,
    n_test: int,
    seeds: list[int],
    retention: float,
) -> list[ConditionData]:
    none_paths = [
        logits_dir / f"{model}-{dataset}-test-none-r1-n{n_test}-seed{seed}.npz"
        for seed in seeds
    ]
    fastv_paths = [
        logits_dir / f"{model}-{dataset}-test-fastv-r{retention:g}-n{n_test}-seed{seed}.npz"
        for seed in seeds
    ]
    none_logits, none_labels = load_logits_and_labels(none_paths)
    fastv_logits, fastv_labels = load_logits_and_labels(fastv_paths)
    if not np.array_equal(none_labels, fastv_labels):
        # Different seed splits are concatenated in the same order per condition,
        # so label equality should hold for a paired pooled comparison.
        raise RuntimeError("Pooled none and FastV labels are not aligned.")

    states = find_calibprune_states(result_globs, dataset=dataset, retention=retention)
    missing = sorted(set(seeds).difference(states))
    if missing:
        raise RuntimeError(f"Missing CalibPrune states for seeds: {missing}")
    calibrated_parts: list[np.ndarray] = []
    for seed, path in zip(seeds, fastv_paths):
        payload = load_npz(path)
        calibrated_parts.append(
            apply_calibprune_state(np.asarray(payload["logits"], dtype=np.float64), retention, states[seed])
        )
    calibrated_logits = np.vstack(calibrated_parts)
    return [
        ConditionData("Unpruned raw", none_logits, none_labels),
        ConditionData(f"FastV r={retention:g} raw", fastv_logits, fastv_labels),
        ConditionData(f"FastV r={retention:g} + CalibPrune", calibrated_logits, fastv_labels),
    ]


def write_tex_snippet(path: Path, *, caption: str) -> None:
    tex_path = path.with_suffix(".tex")
    pdf_name = path.with_suffix(".pdf").name
    tex_path.write_text(
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{pdf_name}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{path.stem}}}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )


def gate_prefix(dataset: str, n_cal: int, n_test: int) -> str:
    return f"{dataset}_{n_cal}_{n_test}"


def dataset_label(dataset: str) -> str:
    labels = {
        "pope": "POPE",
        "scienceqa": "ScienceQA",
        "mmbench_cn": "MMBench-CN",
    }
    return labels.get(dataset, dataset)


def seed_label(seed_count: int) -> str:
    if seed_count == 1:
        return "single-seed"
    return f"{seed_count}-seed"


def pooled_label(seed_count: int) -> str:
    if seed_count == 1:
        return "single seed"
    return f"pooled {seed_count} seeds"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, caption: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    write_tex_snippet(output_dir / stem, caption=caption)
    plt.close(fig)


def plot_ece_retention(
    aggregate_csv: Path,
    output_dir: Path,
    *,
    dataset: str,
    n_cal: int,
    n_test: int,
    seed_count: int,
) -> None:
    prefix = gate_prefix(dataset, n_cal, n_test)
    label = dataset_label(dataset)
    df = pd.read_csv(aggregate_csv)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {
        "none": "#4C78A8",
        "temperature_scaling": "#F58518",
        "calibprune": "#54A24B",
    }
    labels = {
        "none": "FastV raw",
        "temperature_scaling": "FastV + temperature scaling",
        "calibprune": "FastV + CalibPrune",
    }
    fastv = df[(df["subset_name"] == "test") & (df["pruner"] == "fastv")]
    for calibrator in ["none", "temperature_scaling", "calibprune"]:
        group = fastv[fastv["calibrator"] == calibrator].sort_values("retention")
        x = group["retention"].to_numpy(dtype=float)
        y = group["ece_mean"].to_numpy(dtype=float)
        yerr = group["ece_std"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors[calibrator], label=labels[calibrator])
        ax.fill_between(x, y - yerr, y + yerr, color=colors[calibrator], alpha=0.14, linewidth=0)

    none_raw = df[(df["subset_name"] == "test") & (df["pruner"] == "none") & (df["calibrator"] == "none")]
    if not none_raw.empty:
        baseline = float(none_raw.iloc[0]["ece_mean"])
        ax.axhline(baseline, color="#4C78A8", linestyle="--", linewidth=1.3, label="Unpruned raw")
    ax.set_xlabel("Visual token retention")
    ax.set_ylabel("ECE")
    ax.set_title(f"{label} {n_cal}/{n_test} {seed_label(seed_count)} gate")
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{prefix}_ece_retention",
        caption=f"ECE across FastV retention ratios on the {label} {n_cal}/{n_test} {seed_label(seed_count)} gate.",
    )


def plot_confidence_histogram(
    conditions: list[ConditionData],
    output_dir: Path,
    *,
    dataset: str,
    n_cal: int,
    n_test: int,
    seed_count: int,
) -> None:
    prefix = gate_prefix(dataset, n_cal, n_test)
    label = dataset_label(dataset)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bins = np.linspace(0.45, 1.0, 24)
    for condition in conditions:
        conf, _, _ = confidence_and_correctness(condition.logits, condition.labels, probabilities=False)
        ax.hist(conf, bins=bins, histtype="step", linewidth=2.0, density=True, label=condition.label)
    ax.set_xlabel("Max softmax confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} confidence distribution, {pooled_label(seed_count)}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{prefix}_confidence_hist_r05",
        caption=f"Max-softmax confidence distributions before and after FastV pruning on {label} at retention 0.5 ({pooled_label(seed_count)}).",
    )


def plot_reliability(
    conditions: list[ConditionData],
    output_dir: Path,
    *,
    dataset: str,
    n_cal: int,
    n_test: int,
    seed_count: int,
) -> None:
    prefix = gate_prefix(dataset, n_cal, n_test)
    label = dataset_label(dataset)
    fig, axes = plt.subplots(1, len(conditions), figsize=(10.5, 3.3), sharex=True, sharey=True)
    n_bins = 15
    centers = (np.arange(n_bins) + 0.5) / n_bins
    width = 0.88 / n_bins
    for ax, condition in zip(axes, conditions):
        rel = reliability_diagram(condition.logits, condition.labels, n_bins=n_bins, probabilities=False)
        counts = np.asarray(rel.bin_count)
        ax.bar(centers, rel.bin_acc, width=width, color="#72B7B2", alpha=0.78, label="Accuracy")
        nonempty = counts > 0
        sizes = 12 + 5 * np.sqrt(counts[nonempty])
        ax.scatter(np.asarray(rel.bin_conf)[nonempty], np.asarray(rel.bin_acc)[nonempty], s=sizes, color="#E45756", label="Observed bins")
        ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(condition.label, fontsize=9)
        ax.set_xlabel("Confidence")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=7, loc="lower right")
    save_figure(
        fig,
        output_dir,
        f"{prefix}_reliability_r05",
        caption=f"Reliability diagrams for unpruned, FastV-pruned, and CalibPrune-corrected {label} predictions ({pooled_label(seed_count)}).",
    )


def plot_risk_coverage(
    conditions: list[ConditionData],
    output_dir: Path,
    *,
    dataset: str,
    n_cal: int,
    n_test: int,
    seed_count: int,
) -> None:
    prefix = gate_prefix(dataset, n_cal, n_test)
    label = dataset_label(dataset)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for condition in conditions:
        coverage, risk = risk_coverage_curve(condition.logits, condition.labels, probabilities=False)
        ax.plot(coverage, risk, linewidth=2.0, label=condition.label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(f"{label} risk-coverage curves, {pooled_label(seed_count)}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{prefix}_risk_coverage_r05",
        caption=f"Risk-coverage curves for the {label} {n_cal}/{n_test} {seed_label(seed_count)} gate.",
    )


def write_condition_summary(conditions: list[ConditionData], output_csv: Path) -> None:
    rows = []
    for condition in conditions:
        metrics = metrics_from_logits(condition.logits, condition.labels)
        rows.append(
            {
                "condition": condition.label,
                "n_samples": int(condition.labels.shape[0]),
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
                "adaptive_ece": metrics["adaptive_ece"],
                "aurc": aurc(condition.logits, condition.labels),
                "max_softmax_mean": metrics["max_softmax_mean"],
                "max_softmax_p95": metrics["max_softmax_p95"],
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", default="results/tables/pope_lite_llava_128_512_fastv_multiseed_summary.csv")
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--result-glob", action="append", default=["results/raw/pope_lite_llava_128_512*/*.json"])
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--n-cal", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("20260616,20260617,20260618"))
    parser.add_argument("--retention", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--summary-csv", default="results/tables/pope_lite_llava_128_512_pooled_figure_metrics.csv")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    aggregate_csv = PROJECT_ROOT / args.aggregate_csv
    logits_dir = PROJECT_ROOT / args.logits_dir
    conditions = pooled_conditions(
        logits_dir=logits_dir,
        result_globs=args.result_glob,
        model=args.model,
        dataset=args.dataset,
        n_test=args.n_test,
        seeds=args.seeds,
        retention=args.retention,
    )

    plot_ece_retention(
        aggregate_csv,
        output_dir,
        dataset=args.dataset,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed_count=len(args.seeds),
    )
    plot_confidence_histogram(
        conditions,
        output_dir,
        dataset=args.dataset,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed_count=len(args.seeds),
    )
    plot_reliability(
        conditions,
        output_dir,
        dataset=args.dataset,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed_count=len(args.seeds),
    )
    plot_risk_coverage(
        conditions,
        output_dir,
        dataset=args.dataset,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed_count=len(args.seeds),
    )
    write_condition_summary(conditions, PROJECT_ROOT / args.summary_csv)

    prefix = gate_prefix(args.dataset, args.n_cal, args.n_test)
    written = sorted(str(path.relative_to(PROJECT_ROOT)) for path in output_dir.glob(f"{prefix}_*"))
    print(json.dumps({"figures": written, "summary_csv": args.summary_csv}, indent=2))


if __name__ == "__main__":
    main()
