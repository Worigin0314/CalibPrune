"""Evaluate whether aggregate result rows satisfy the current paper-evidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ALLOWED_EVIDENCE = {"unpruned", "literature-hook", "literature-midlayer-hook"}


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def row_reasons(
    row: pd.Series,
    *,
    min_pooled_samples: int,
    min_seeds: int,
    allowed_evidence: set[str],
    require_json_flag: bool,
) -> list[str]:
    reasons: list[str] = []
    n_samples = int(row.get("n_samples", 0))
    n_seeds = int(row.get("n_seeds", 1)) if not pd.isna(row.get("n_seeds", 1)) else 1
    pooled = n_samples * n_seeds
    evidence = str(row.get("pruner_evidence_type", "unknown"))
    if evidence not in allowed_evidence:
        reasons.append(f"evidence_type={evidence}")
    if n_seeds < min_seeds:
        reasons.append(f"n_seeds={n_seeds}<min_seeds={min_seeds}")
    if pooled < min_pooled_samples:
        reasons.append(f"pooled_samples={pooled}<min_pooled_samples={min_pooled_samples}")
    if require_json_flag and not bool(row.get("paper_eligible_all", row.get("paper_eligible", False))):
        reasons.append("source_json_not_marked_paper_eligible")
    return reasons


def evaluate(df: pd.DataFrame, *, min_pooled_samples: int, min_seeds: int, allowed_evidence: set[str], require_json_flag: bool) -> pd.DataFrame:
    out = df.copy()
    reasons = [
        row_reasons(
            row,
            min_pooled_samples=min_pooled_samples,
            min_seeds=min_seeds,
            allowed_evidence=allowed_evidence,
            require_json_flag=require_json_flag,
        )
        for _, row in out.iterrows()
    ]
    out["pooled_samples"] = pd.to_numeric(out.get("n_samples", 0), errors="coerce").fillna(0).astype(int) * pd.to_numeric(
        out.get("n_seeds", 1), errors="coerce"
    ).fillna(1).astype(int)
    out["paper_gate_pass"] = [not item for item in reasons]
    out["paper_gate_reasons"] = ["; ".join(item) if item else "pass" for item in reasons]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--min-pooled-samples", type=int, default=2000)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--allowed-evidence", default="unpruned,literature-hook,literature-midlayer-hook")
    parser.add_argument("--require-json-flag", action="store_true")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out = evaluate(
        df,
        min_pooled_samples=args.min_pooled_samples,
        min_seeds=args.min_seeds,
        allowed_evidence=parse_csv(args.allowed_evidence),
        require_json_flag=args.require_json_flag,
    )
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    md_path = Path(args.output_md) if args.output_md else csv_path.with_suffix(".md")
    display_cols = [
        col
        for col in [
            "model",
            "dataset",
            "subset_name",
            "pruner",
            "pruner_evidence_type",
            "retention",
            "calibrator",
            "n_samples",
            "n_seeds",
            "pooled_samples",
            "accuracy_mean",
            "ece_mean",
            "aurc_mean",
            "paper_gate_pass",
            "paper_gate_reasons",
        ]
        if col in out.columns
    ]
    md_path.write_text(out[display_cols].to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(out)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
