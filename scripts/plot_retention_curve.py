"""Plot the LLaVA/POPE FastV retention curve used in the manuscript.

The figure combines the 3-seed metric summary with paired logit-diagnostic
outputs. It is derived from saved logits; no model inference is performed here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "tables" / "pope_lite_llava_128_512_fastv_multiseed_summary.csv"
DIAG_CSV = ROOT / "results" / "tables" / "logit_sharpening_llava_pope_fastv_512_3seed.csv"
OUT_CSV = ROOT / "results" / "tables" / "retention_curve_llava_pope_fastv.csv"
OUT_MD = ROOT / "results" / "tables" / "retention_curve_llava_pope_fastv.md"
OUT_PDF = ROOT / "paper" / "figures" / "retention_curve_llava_pope_fastv.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "retention_curve_llava_pope_fastv.png"


def read_summary() -> dict[float, dict[str, float]]:
    rows: dict[float, dict[str, float]] = {}
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["calibrator"] != "none":
                continue
            if row["pruner"] not in {"none", "fastv"}:
                continue
            r = float(row["retention"])
            rows[r] = {
                "accuracy": float(row["accuracy_mean"]),
                "ece": float(row["ece_mean"]),
                "aurc": float(row["aurc_mean"]),
            }
    return rows


def read_diagnostics() -> dict[float, dict[str, float]]:
    rows: dict[float, dict[str, float]] = {1.0: {"max_softmax_shift": 0.0, "scalar_residual": 0.0}}
    with DIAG_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row["metric"]
            if metric not in {"max_softmax_mean_shift", "scale_residual_rel_norm"}:
                continue
            r = float(row["retention"])
            slot = rows.setdefault(r, {})
            if metric == "max_softmax_mean_shift":
                slot["max_softmax_shift"] = float(row["estimate"])
            else:
                slot["scalar_residual"] = float(row["estimate"])
    return rows


def write_tables(rows: list[dict[str, float]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["retention", "accuracy", "ece", "aurc", "max_softmax_shift", "scalar_residual"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.6f}" for k in fieldnames})
    lines = [
        "Derived from the 3-seed LLaVA/POPE FastV summary and saved-logit diagnostics.",
        "",
        "| r | Acc | ECE | AURC | Max-softmax shift | Scalar residual |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row['retention']:.2f} | {row['accuracy']:.4f} | {row['ece']:.4f} | "
            f"{row['aurc']:.4f} | {row['max_softmax_shift']:+.4f} | {row['scalar_residual']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, float]]) -> None:
    x = [row["retention"] for row in rows]
    series = [
        ("Accuracy", "accuracy", "#0072B2"),
        ("ECE", "ece", "#D55E00"),
        ("AURC", "aurc", "#009E73"),
        ("Max-softmax shift", "max_softmax_shift", "#CC79A7"),
        ("Scalar residual", "scalar_residual", "#E69F00"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(12.8, 2.7), sharex=True)
    for ax, (title, key, color) in zip(axes, series):
        y = [row[key] for row in rows]
        ax.plot(x, y, marker="o", color=color, linewidth=1.8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, color="#DDDDDD", linewidth=0.6)
        ax.set_xlabel("retention r", fontsize=8)
        ax.tick_params(labelsize=8)
        if key in {"max_softmax_shift"}:
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_ylabel("metric value", fontsize=8)
    fig.suptitle("Retention-dependent reliability drift: LLaVA/POPE FastV", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)


def main() -> None:
    summary = read_summary()
    diagnostics = read_diagnostics()
    rows = []
    for retention in [1.0, 0.75, 0.5, 0.25]:
        merged = {"retention": retention}
        merged.update(summary[retention])
        merged.update(diagnostics[retention])
        rows.append(merged)
    write_tables(rows)
    plot(rows)


if __name__ == "__main__":
    main()
