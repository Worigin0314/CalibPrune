"""Apply Holm correction across a declared statistical family of CSV tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.metrics.statistics import holm_bonferroni


def parse_label(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = path.stem
    return stem.replace("_adaptive_vs_temperature_paired_stats", "").replace("_", " ")


def parse_pruner_retention(comparison: str) -> tuple[str, float | None]:
    match = re.search(r"\((?P<pruner>[A-Za-z0-9_\-]+) r=(?P<retention>[0-9.]+)\)", comparison)
    if not match:
        return "", None
    return match.group("pruner"), float(match.group("retention"))


def read_family_rows(csv_paths: list[Path], labels: list[str | None], metric: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if "metric" not in df.columns or "wilcoxon_p" not in df.columns:
            raise ValueError(f"{path} is not a paired statistics CSV.")
        df = df[df["metric"] == metric].copy()
        if df.empty:
            raise ValueError(f"{path} has no rows for metric={metric!r}.")
        dataset_label = parse_label(path, label)
        pruner_retention = df["comparison"].map(parse_pruner_retention)
        df.insert(0, "family_dataset", dataset_label)
        df.insert(1, "family_metric", metric)
        df.insert(2, "family_pruner", [item[0] for item in pruner_retention])
        df.insert(3, "family_retention", [item[1] for item in pruner_retention])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def apply_family_holm(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    p_values = pd.to_numeric(df["wilcoxon_p"], errors="coerce")
    finite_values = p_values.to_numpy(dtype=float)
    finite_mask = p_values.notna() & np.isfinite(finite_values)
    df["family_holm_p"] = np.nan
    df["family_holm_reject"] = False
    if finite_mask.any():
        holm = holm_bonferroni(p_values[finite_mask].to_numpy(dtype=float), alpha=alpha)
        df.loc[finite_mask, "family_holm_p"] = holm["adjusted_p_values"]
        df.loc[finite_mask, "family_holm_reject"] = holm["reject"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--metric", default="ece")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    csv_paths = [PROJECT_ROOT / path for path in args.input_csv]
    labels = args.label or []
    if labels and len(labels) != len(csv_paths):
        raise ValueError("--label must be provided once per --input-csv when used.")
    if not labels:
        labels = [None] * len(csv_paths)

    df = read_family_rows(csv_paths, labels, args.metric)
    df = apply_family_holm(df, args.alpha)
    keep = [
        "family_dataset",
        "family_metric",
        "family_pruner",
        "family_retention",
        "comparison",
        "estimate",
        "ci_low",
        "ci_high",
        "n_samples",
        "seed_delta_mean",
        "seed_delta_bca_ci_low",
        "seed_delta_bca_ci_high",
        "wilcoxon_p",
        "wilcoxon_holm_p",
        "family_holm_p",
        "family_holm_reject",
    ]
    out = df[keep]
    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    md_path = PROJECT_ROOT / args.output_md if args.output_md else csv_path.with_suffix(".md")
    md_path.write_text(out.to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(out)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
