"""Summarize CalibPrune JSON result files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


METRICS = ["accuracy", "ece", "adaptive_ece", "aurc", "aurc_margin", "aurc_entropy", "max_softmax_mean"]
SANITY_ONLY_PRUNERS = {"random", "uniform", "feature_norm"}
LITERATURE_HOOK_PRUNERS = {"fastv", "visionzip"}
LITERATURE_PROXY_HOOK_PRUNERS = {"sparsevlm", "pyramiddrop", "vtw"}
EVIDENCE_TYPE_OVERRIDES = {"literature-midlayer-hook"}
NOT_IMPLEMENTED_LITERATURE_PRUNERS: set[str] = set()


def pruner_evidence_type(pruner: object) -> str:
    name = str(pruner).lower()
    if name == "none":
        return "unpruned"
    if name in SANITY_ONLY_PRUNERS:
        return "sanity-only"
    if name in NOT_IMPLEMENTED_LITERATURE_PRUNERS:
        return "not-implemented"
    if name in LITERATURE_PROXY_HOOK_PRUNERS:
        return "literature-proxy-hook"
    if name in LITERATURE_HOOK_PRUNERS:
        return "literature-hook"
    return "unknown"


def annotate_pruner_evidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    default_evidence = df["pruner"].map(pruner_evidence_type)
    if "pruner_evidence_type" in df.columns:
        provided = df["pruner_evidence_type"].where(df["pruner_evidence_type"].isin(EVIDENCE_TYPE_OVERRIDES))
        df["pruner_evidence_type"] = provided.fillna(default_evidence)
    else:
        df["pruner_evidence_type"] = default_evidence
    df["sanity_only"] = df["pruner_evidence_type"] == "sanity-only"
    paper_flag = df.get("paper_eligible", pd.Series(False, index=df.index))
    paper_flag = paper_flag.fillna(False)
    df["paper_eligible_summary"] = (
        paper_flag.astype(bool)
        & ~df["sanity_only"]
        & (df["pruner_evidence_type"] != "not-implemented")
        & (df["pruner_evidence_type"] != "literature-proxy-hook")
    )
    return df


def parse_csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def filter_result_rows(
    df: pd.DataFrame,
    *,
    pruners: set[str] | None = None,
    exclude_sanity: bool = False,
    paper_eligible_only: bool = False,
) -> pd.DataFrame:
    out = df
    if pruners:
        out = out[out["pruner"].isin(pruners)]
    if exclude_sanity:
        out = out[~out["sanity_only"]]
    if paper_eligible_only:
        out = out[out["paper_eligible_summary"]]
    return out


def load_rows(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if "model" not in payload or "pruner" not in payload:
                continue
            row = {"path": path}
            for key in [
                "run_id",
                "model",
                "dataset",
                "subset_name",
                "pruner",
                "retention",
                "calibrator",
                "seed",
                "n_samples",
                "sample_indices_hash",
                "paper_eligible",
                "pruner_evidence_type",
            ] + METRICS:
                row[key] = payload.get(key)
            rows.append(row)
    if not rows:
        raise RuntimeError("No result JSON files matched the requested pattern(s).")
    return annotate_pruner_evidence(pd.DataFrame(rows))


def add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in METRICS:
        df[f"delta_{metric}"] = pd.NA
    keys = ["model", "dataset", "subset_name", "calibrator", "seed", "n_samples"]
    baselines = df[(df["pruner"] == "none") & (df["retention"].astype(float) == 1.0)]
    for _, baseline in baselines.iterrows():
        mask = pd.Series(True, index=df.index)
        for key in keys:
            mask &= df[key] == baseline[key]
        for metric in METRICS:
            baseline_value = pd.to_numeric(pd.Series([baseline[metric]]), errors="coerce").iloc[0]
            if pd.isna(baseline_value):
                continue
            df.loc[mask, f"delta_{metric}"] = pd.to_numeric(df.loc[mask, metric], errors="coerce") - float(baseline_value)
    return df


def add_raw_calibrator_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in METRICS:
        df[f"delta_{metric}_vs_raw"] = pd.NA
    keys = ["model", "dataset", "subset_name", "pruner", "retention", "seed", "n_samples"]
    raw_rows = df[df["calibrator"] == "none"]
    for _, raw in raw_rows.iterrows():
        mask = pd.Series(True, index=df.index)
        for key in keys:
            mask &= df[key] == raw[key]
        for metric in METRICS:
            raw_value = pd.to_numeric(pd.Series([raw[metric]]), errors="coerce").iloc[0]
            if pd.isna(raw_value):
                continue
            df.loc[mask, f"delta_{metric}_vs_raw"] = pd.to_numeric(df.loc[mask, metric], errors="coerce") - float(raw_value)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-csv", default="results/tables/summary.csv")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--pruners", default=None, help="Optional comma-separated pruner allowlist.")
    parser.add_argument("--exclude-sanity", action="store_true", help="Exclude random/uniform/feature_norm sanity baselines.")
    parser.add_argument(
        "--paper-eligible-only",
        action="store_true",
        help="Keep only rows whose JSON is paper_eligible and whose pruner is not sanity-only.",
    )
    args = parser.parse_args()

    df = add_raw_calibrator_deltas(add_baseline_deltas(load_rows(args.input_glob)))
    df = filter_result_rows(
        df,
        pruners=parse_csv_filter(args.pruners),
        exclude_sanity=args.exclude_sanity,
        paper_eligible_only=args.paper_eligible_only,
    )
    if df.empty:
        raise RuntimeError("No rows remained after filtering.")
    sort_cols = ["model", "dataset", "subset_name", "pruner", "retention", "calibrator", "n_samples"]
    df = df.sort_values([col for col in sort_cols if col in df.columns])
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

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
        "accuracy",
        "ece",
        "delta_ece",
        "delta_ece_vs_raw",
        "aurc",
        "aurc_margin",
        "aurc_entropy",
        "delta_aurc",
        "delta_aurc_vs_raw",
        "max_softmax_mean",
        "delta_max_softmax_mean",
        "paper_eligible",
        "paper_eligible_summary",
    ]
    md_path.write_text(df[display_cols].to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(df)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
