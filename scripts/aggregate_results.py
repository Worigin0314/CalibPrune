"""Aggregate CalibPrune result JSON files across repeated seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_results import add_baseline_deltas, add_raw_calibrator_deltas, filter_result_rows, load_rows, parse_csv_filter


AGG_METRICS = [
    "accuracy",
    "ece",
    "delta_ece",
    "delta_ece_vs_raw",
    "aurc",
    "aurc_margin",
    "aurc_entropy",
    "delta_aurc_vs_raw",
    "max_softmax_mean",
]


def filter_rows(
    df: pd.DataFrame,
    *,
    subset_name: str | None,
    pruners: set[str] | None,
    exclude_sanity: bool,
    paper_eligible_only: bool,
) -> pd.DataFrame:
    out = df
    if subset_name:
        out = out[out["subset_name"] == subset_name]
    return filter_result_rows(
        out,
        pruners=pruners,
        exclude_sanity=exclude_sanity,
        paper_eligible_only=paper_eligible_only,
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in AGG_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    group_cols = [
        "model",
        "dataset",
        "subset_name",
        "pruner",
        "retention",
        "calibrator",
        "n_samples",
    ]
    grouped = df.groupby(group_cols, dropna=False)
    pieces = [
        grouped["seed"].nunique().rename("n_seeds"),
        grouped["seed"].apply(lambda values: ",".join(str(int(value)) for value in sorted(set(values)))).rename("seeds"),
        grouped["pruner_evidence_type"].first().rename("pruner_evidence_type"),
        grouped["paper_eligible_summary"].all().rename("paper_eligible_all"),
    ]
    for metric in AGG_METRICS:
        pieces.append(grouped[metric].mean().rename(f"{metric}_mean"))
        pieces.append(grouped[metric].std(ddof=0).rename(f"{metric}_std"))
    return pd.concat(pieces, axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--subset-name", default="test")
    parser.add_argument("--pruners", default=None, help="Optional comma-separated pruner allowlist, e.g. none,fastv.")
    parser.add_argument("--exclude-sanity", action="store_true", help="Exclude random/uniform/feature_norm sanity baselines.")
    parser.add_argument(
        "--paper-eligible-only",
        action="store_true",
        help="Keep only rows whose JSON is paper_eligible and whose pruner is not sanity-only.",
    )
    parser.add_argument("--output-csv", default="results/tables/aggregate_summary.csv")
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    df = add_raw_calibrator_deltas(add_baseline_deltas(load_rows(args.input_glob)))
    df = filter_rows(
        df,
        subset_name=args.subset_name,
        pruners=parse_csv_filter(args.pruners),
        exclude_sanity=args.exclude_sanity,
        paper_eligible_only=args.paper_eligible_only,
    )
    if df.empty:
        raise RuntimeError("No rows remained after filtering.")

    out = aggregate(df)
    sort_cols = ["model", "dataset", "subset_name", "pruner", "retention", "calibrator", "n_samples"]
    out = out.sort_values(sort_cols)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)

    md_path = Path(args.output_md) if args.output_md else csv_path.with_suffix(".md")
    display_cols = [
        "model",
        "dataset",
        "subset_name",
        "pruner",
        "pruner_evidence_type",
        "retention",
        "calibrator",
        "n_samples",
        "n_seeds",
        "seeds",
        "paper_eligible_all",
        "accuracy_mean",
        "ece_mean",
        "ece_std",
        "delta_ece_mean",
        "delta_ece_vs_raw_mean",
        "delta_ece_vs_raw_std",
        "aurc_mean",
        "aurc_margin_mean",
        "aurc_entropy_mean",
    ]
    md_path.write_text(out[display_cols].to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(out)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
