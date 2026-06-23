"""Plotting utilities for CalibPrune tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ece_by_retention(df: pd.DataFrame, output: str | Path) -> None:
    required = {"dataset", "pruner", "retention", "ece"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for (dataset, pruner), group in df.groupby(["dataset", "pruner"]):
        group = group.sort_values("retention")
        ax.plot(group["retention"], group["ece"], marker="o", label=f"{dataset}/{pruner}")
    ax.set_xlabel("Retention ratio")
    ax.set_ylabel("ECE (lower is better)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

