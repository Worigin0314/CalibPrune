"""Aggregate efficiency and reliability fields from real result JSON files.

The output is intended for manuscript tables that connect visual-token reduction
with accuracy, calibration, and selective-reliability metrics.  Wall-clock values
come from the runner-level JSON field and should be treated as a coarse proxy,
not as a controlled CUDA latency benchmark.  Peak memory is not recorded by the
current runners and is therefore deliberately absent from this table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def metadata_tokens(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    metadata = payload.get("last_sample_metadata") or {}
    original = metadata.get("num_visual_tokens_original")
    kept = metadata.get("num_visual_tokens_kept")
    try:
        original_f = float(original)
        kept_f = float(kept)
    except (TypeError, ValueError):
        return None, None
    if original_f <= 0:
        return None, None
    return original_f, kept_f


def collect_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        full_pattern = str((PROJECT_ROOT / pattern).resolve()) if not Path(pattern).is_absolute() else pattern
        paths.extend(Path(p) for p in glob.glob(full_pattern))
    unique = sorted({p.resolve() for p in paths})
    return unique


def fmt_mean_std(values: list[float], digits: int = 4) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.{digits}f}"
    return f"{mean(clean):.{digits}f} +/- {stdev(clean):.{digits}f}"


def fmt_pct(values: list[float]) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.1f}%"
    return f"{mean(clean):.1f}% +/- {stdev(clean):.1f}%"


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join([":--" for _ in columns]) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-glob", action="append", required=True, help="Glob relative to the project root.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--calibrator", default="none")
    parser.add_argument("--min-samples", type=int, default=1)
    args = parser.parse_args()

    payloads = []
    for path in collect_paths(args.result_glob):
        payload = load_json(path)
        if payload.get("split") != "test" and payload.get("subset_name") != "test":
            continue
        if payload.get("calibrator", "none") != args.calibrator:
            continue
        if payload.get("dataset") is None or payload.get("model") is None:
            continue
        if payload.get("n_samples") is None or payload.get("seed") is None:
            continue
        if int(payload.get("n_samples")) < args.min_samples:
            continue
        payloads.append(payload)

    if not payloads:
        raise RuntimeError("No matching test result JSON files found.")

    baselines: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("pruner") == "none" and float(payload.get("retention", 1.0)) == 1.0:
            key = (str(payload["model"]), str(payload["dataset"]), int(payload["seed"]), int(payload["n_samples"]))
            baselines[key] = payload

    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        retention = float(payload.get("retention", 1.0))
        key = (str(payload["model"]), str(payload["dataset"]), str(payload.get("pruner", "none")), retention)
        grouped[key].append(payload)

    output_rows: list[dict[str, str]] = []
    for (model, dataset, pruner, retention), group in sorted(grouped.items()):
        accuracies: list[float] = []
        eces: list[float] = []
        aurcs: list[float] = []
        token_reductions: list[float] = []
        wall_clock_speedups: list[float] = []
        seconds_per_sample: list[float] = []
        seeds: list[str] = []
        sample_counts: list[int] = []
        for payload in sorted(group, key=lambda item: int(item["seed"])):
            seeds.append(str(payload["seed"]))
            sample_counts.append(int(payload["n_samples"]))
            for target, key_name in ((accuracies, "accuracy"), (eces, "ece"), (aurcs, "aurc")):
                value = safe_float(payload, key_name)
                if value is not None:
                    target.append(value)
            original, kept = metadata_tokens(payload)
            if original is not None and kept is not None:
                token_reductions.append(100.0 * (1.0 - kept / original))
            wall = safe_float(payload, "wall_clock_min")
            if wall is not None and int(payload["n_samples"]) > 0:
                current_sec = wall * 60.0 / int(payload["n_samples"])
                seconds_per_sample.append(current_sec)
                baseline = baselines.get((model, dataset, int(payload["seed"]), int(payload["n_samples"])))
                base_wall = safe_float(baseline or {}, "wall_clock_min")
                if base_wall is not None and base_wall > 0:
                    base_sec = base_wall * 60.0 / int(payload["n_samples"])
                    wall_clock_speedups.append(100.0 * (1.0 - current_sec / base_sec))

        output_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "pruner": pruner,
                "retention": f"{retention:.2f}",
                "n_samples_per_seed": ",".join(str(n) for n in sorted(set(sample_counts))),
                "n_seeds": str(len(set(seeds))),
                "seeds": ",".join(sorted(set(seeds))),
                "token_reduction": fmt_pct(token_reductions),
                "wall_clock_speedup_proxy": fmt_pct(wall_clock_speedups),
                "seconds_per_sample": fmt_mean_std(seconds_per_sample, digits=3),
                "accuracy": fmt_mean_std(accuracies, digits=4),
                "ece": fmt_mean_std(eces, digits=4),
                "aurc": fmt_mean_std(aurcs, digits=4),
            }
        )

    columns = [
        "model",
        "dataset",
        "pruner",
        "retention",
        "n_samples_per_seed",
        "n_seeds",
        "seeds",
        "token_reduction",
        "wall_clock_speedup_proxy",
        "seconds_per_sample",
        "accuracy",
        "ece",
        "aurc",
    ]
    out_csv = PROJECT_ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    out_md = PROJECT_ROOT / args.output_md if args.output_md else out_csv.with_suffix(".md")
    note = (
        "Wall-clock speedup is a coarse runner-level proxy computed against the matching unpruned "
        "test run with the same model, dataset, seed, and sample count. Positive means faster. "
        "Peak GPU memory is not recorded by current JSON outputs.\n\n"
    )
    out_md.write_text(note + markdown_table(output_rows, columns), encoding="utf-8")


if __name__ == "__main__":
    main()




