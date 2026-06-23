"""Build the compact AdaptiveCalibPrune-vs-temperature evidence table."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvidenceSource:
    dataset: str
    summary_csv: str
    stats_csv: str
    family_label: str


SOURCES = (
    EvidenceSource(
        dataset="POPE",
        summary_csv="results/tables/pope_lite_llava_128_512_adaptive_log_margin_3seed_test_comparison.csv",
        stats_csv="results/tables/pope_lite_llava_128_512_adaptive_vs_temperature_paired_stats.csv",
        family_label="POPE",
    ),
    EvidenceSource(
        dataset="ScienceQA",
        summary_csv="results/tables/scienceqa_lite_llava_128_256_adaptive_log_margin_3seed_test_comparison.csv",
        stats_csv="results/tables/scienceqa_lite_llava_128_256_adaptive_vs_temperature_paired_stats.csv",
        family_label="ScienceQA",
    ),
    EvidenceSource(
        dataset="MMBench-CN",
        summary_csv="results/tables/mmbench_cn_lite_llava_128_256_adaptive_log_margin_3seed_test_comparison.csv",
        stats_csv="results/tables/mmbench_cn_lite_llava_128_256_adaptive_vs_temperature_paired_stats.csv",
        family_label="MMBench-CN-fastv",
    ),
    EvidenceSource(
        dataset="MMBench-CN",
        summary_csv="results/tables/mmbench_cn_lite_llava_128_256_adaptive_log_margin_3seed_test_comparison.csv",
        stats_csv="results/tables/mmbench_cn_lite_llava_128_256_visionzip_adaptive_vs_temperature_paired_stats.csv",
        family_label="MMBench-CN-visionzip",
    ),
    EvidenceSource(
        dataset="A-OKVQA",
        summary_csv="results/tables/aokvqa_full_validation_llava_fastv_r05_calibrated_test_comparison.csv",
        stats_csv="results/tables/aokvqa_full_validation_llava_fastv_r05_adaptive_vs_temperature_paired_stats.csv",
        family_label="A-OKVQA",
    ),
)


def parse_pruner_retention(comparison: str) -> tuple[str, float]:
    match = re.search(r"\((?P<pruner>[A-Za-z0-9_\-]+) r=(?P<retention>[0-9.]+)\)", comparison)
    if not match:
        raise ValueError(f"Cannot parse comparison label: {comparison}")
    return match.group("pruner"), float(match.group("retention"))


def find_ece(summary: pd.DataFrame, pruner: str, retention: float, calibrator: str) -> float:
    rows = summary[
        (summary["subset_name"] == "test")
        & (summary["pruner"] == pruner)
        & (summary["calibrator"] == calibrator)
        & ((summary["retention"].astype(float) - retention).abs() < 1e-9)
    ]
    if rows.empty:
        raise ValueError(f"Missing {calibrator} ECE for {pruner} r={retention:g}.")
    return float(rows.iloc[0]["ece_mean"])


def build_rows() -> list[dict[str, object]]:
    family = pd.read_csv(PROJECT_ROOT / "results/tables/adaptive_vs_temperature_ece_family_holm.csv")
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        summary = pd.read_csv(PROJECT_ROOT / source.summary_csv)
        stats = pd.read_csv(PROJECT_ROOT / source.stats_csv)
        stats = stats[stats["metric"] == "ece"].copy()
        for _, stat_row in stats.iterrows():
            pruner, retention = parse_pruner_retention(str(stat_row["comparison"]))
            family_rows = family[
                (family["family_dataset"] == source.family_label)
                & (family["family_pruner"] == pruner)
                & ((family["family_retention"].astype(float) - retention).abs() < 1e-9)
            ]
            if family_rows.empty:
                raise ValueError(f"Missing family Holm row for {source.family_label} {pruner} r={retention:g}.")
            ts_ece = find_ece(summary, pruner, retention, "temperature_scaling")
            adaptive_ece = find_ece(summary, pruner, retention, "adaptive_calibprune")
            rows.append(
                {
                    "dataset": source.dataset,
                    "pruner": pruner,
                    "retention": retention,
                    "ts_ece_mean": ts_ece,
                    "adaptive_ece_mean": adaptive_ece,
                    "adaptive_minus_ts_mean": adaptive_ece - ts_ece,
                    "paired_pooled_delta": float(stat_row["estimate"]),
                    "ci_low": float(stat_row["ci_low"]),
                    "ci_high": float(stat_row["ci_high"]),
                    "family_holm_p": float(family_rows.iloc[0]["family_holm_p"]),
                    "family_holm_reject": bool(family_rows.iloc[0]["family_holm_reject"]),
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "results/tables/adaptive_log_margin_cross_dataset_ts_comparison.csv"
    md_path = PROJECT_ROOT / "results/tables/adaptive_log_margin_cross_dataset_ts_comparison.md"
    df.to_csv(csv_path, index=False)
    display = df.copy()
    for column in [
        "ts_ece_mean",
        "adaptive_ece_mean",
        "adaptive_minus_ts_mean",
        "paired_pooled_delta",
        "ci_low",
        "ci_high",
        "family_holm_p",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    display["95% CI"] = "[" + display.pop("ci_low") + ", " + display.pop("ci_high") + "]"
    display = display.rename(
        columns={
            "ts_ece_mean": "TS ECE mean",
            "adaptive_ece_mean": "Adaptive ECE mean",
            "adaptive_minus_ts_mean": "Adaptive - TS mean",
            "paired_pooled_delta": "Paired pooled delta",
            "family_holm_p": "Family Holm p",
            "family_holm_reject": "Family Holm reject",
        }
    )
    md_path.write_text(display.to_markdown(index=False), encoding="utf-8")
    print(json.dumps({"rows": int(len(df)), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
