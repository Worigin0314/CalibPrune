"""Plot scalar residual against observed reliability drift.

The manuscript uses this as a diagnostic sanity check rather than a formal
predictive claim: residual should warn when scalar temperature explanations are
mechanistically unsafe, while ECE and AURC remain the direct reliability metrics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


CELLS = [
    ("LLaVA", "POPE", "FastV r=.75", 0.139255, 0.0066667, -0.0010403),
    ("LLaVA", "POPE", "FastV r=.5", 0.266669, 0.0208545, 0.0087573),
    ("LLaVA", "POPE", "FastV r=.25", 0.423521, 0.0503784, 0.0341484),
    ("LLaVA", "POPE", "SparseVLM r=.5", 0.0503699, 0.0050, -0.0027),
    ("LLaVA", "POPE", "VTW r=.5", 0.0540707, 0.0015, -0.0028),
    ("LLaVA", "A-OKVQA", "FastV r=.5", 0.1718739, -0.00393735, -0.0016008),
    ("Qwen2-VL", "ScienceQA", "FastV r=.75", 1.986e-9, 0.0739503, 0.0238928),
    ("Qwen2-VL", "ScienceQA", "FastV r=.5", 1.926e-9, 0.0718456, 0.0259591),
    ("Qwen2-VL", "ScienceQA", "FastV r=.25", 1.845e-9, 0.0562922, 0.0194389),
    ("Qwen2-VL", "A-OKVQA", "FastV r=.75 full", 0.5808, 0.0713392, -0.0634623),
    ("Qwen2-VL", "A-OKVQA", "FastV r=.5 full", 0.56036165, 0.06306129, -0.06066789),
    ("Qwen2-VL", "A-OKVQA", "FastV r=.25 full", 0.5092, 0.0453863, -0.0568494),
    ("Qwen2-VL", "POPE", "FastV r=.5", 0.8657, -0.0340, -0.0048),
]


def rank(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][1]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    norm_x = math.sqrt(sum((value - mean_x) ** 2 for value in x))
    norm_y = math.sqrt(sum((value - mean_y) ** 2 for value in y))
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (norm_x * norm_y)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank(x), rank(y))


def write_outputs() -> None:
    out_dir = ROOT / "results" / "tables"
    fig_dir = ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    residual = [row[3] for row in CELLS]
    abs_ece = [abs(row[4]) for row in CELLS]
    abs_aurc = [abs(row[5]) for row in CELLS]
    signed_ece = [row[4] for row in CELLS]
    signed_aurc = [row[5] for row in CELLS]

    summary_rows = [
        ("signed_delta_ece", pearson(residual, signed_ece), spearman(residual, signed_ece), len(CELLS)),
        ("signed_delta_aurc", pearson(residual, signed_aurc), spearman(residual, signed_aurc), len(CELLS)),
        ("abs_delta_ece", pearson(residual, abs_ece), spearman(residual, abs_ece), len(CELLS)),
        ("abs_delta_aurc", pearson(residual, abs_aurc), spearman(residual, abs_aurc), len(CELLS)),
    ]

    csv_path = out_dir / "residual_reliability_correlation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "pearson", "spearman", "n"])
        for name, pearson_value, spearman_value, n in summary_rows:
            writer.writerow([name, f"{pearson_value:.4f}", f"{spearman_value:.4f}", n])

    md_path = out_dir / "residual_reliability_correlation.md"
    md_lines = [
        "| Comparison | Pearson | Spearman | n |",
        "|:--|--:|--:|--:|",
    ]
    for name, pearson_value, spearman_value, n in summary_rows:
        md_lines.append(f"| {name} | {pearson_value:.3f} | {spearman_value:.3f} | {n} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    cell_csv = out_dir / "residual_reliability_cells.csv"
    with cell_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backbone", "dataset", "condition", "residual", "delta_ece", "delta_aurc"])
        writer.writerows(CELLS)

    colors = ["#4C78A8" if row[0] == "LLaVA" else "#F58518" for row in CELLS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    axes[0].scatter(residual, abs_ece, c=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("Scalar residual")
    axes[0].set_ylabel("|Delta ECE|")
    axes[0].set_title(f"ECE drift, rho={spearman(residual, abs_ece):.2f}")
    axes[1].scatter(residual, abs_aurc, c=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Scalar residual")
    axes[1].set_ylabel("|Delta AURC|")
    axes[1].set_title(f"AURC drift, rho={spearman(residual, abs_aurc):.2f}")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_reliability_correlation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    write_outputs()
