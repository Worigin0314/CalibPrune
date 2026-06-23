"""Build the manuscript efficiency measurement table from runner JSON outputs.

This table is intentionally hardware- and runner-scoped. It reports relative
changes against a matched unpruned run with the same model, dataset, seed, and
sample count, using the latency and peak-memory fields emitted by
scripts/run_pope_lite_inprocess.py.
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

ROOT = Path(__file__).resolve().parents[1]


def collect(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        full = str((ROOT / pattern).resolve()) if not Path(pattern).is_absolute() else pattern
        paths.extend(Path(p) for p in glob.glob(full))
    return sorted({p.resolve() for p in paths})


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_path"] = str(path)
    return payload


def f(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
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


def token_reduction(payload: dict[str, Any]) -> float | None:
    meta = payload.get("last_sample_metadata") or {}
    try:
        original = float(meta.get("num_visual_tokens_original"))
        kept = float(meta.get("num_visual_tokens_kept"))
    except (TypeError, ValueError):
        return None
    if original <= 0:
        return None
    return 100.0 * (1.0 - kept / original)


def reduction(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None
    return 100.0 * (1.0 - current / baseline)


def fmt_pct(values: list[float | None]) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:+.1f}%"
    return f"{mean(clean):+.1f}% +/- {stdev(clean):.1f}%"


def fmt_num(values: list[float | None], digits: int = 4) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.{digits}f}"
    return f"{mean(clean):.{digits}f} +/- {stdev(clean):.{digits}f}"


def md(rows: list[dict[str, str]], columns: list[str], note: str) -> str:
    lines = [note.rstrip(), ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join([":--" for _ in columns]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-glob", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--min-samples", type=int, default=1)
    args = parser.parse_args()

    payloads: list[dict[str, Any]] = []
    for path in collect(args.result_glob):
        payload = load(path)
        if payload.get("calibrator", "none") != "none":
            continue
        if payload.get("subset_name", payload.get("split")) != "test":
            continue
        if int(payload.get("n_samples", 0)) < args.min_samples:
            continue
        if payload.get("model") is None or payload.get("dataset") is None:
            continue
        payloads.append(payload)

    if not payloads:
        raise RuntimeError("No matching test JSON payloads found.")

    baselines: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        model = str(payload["model"])
        dataset = str(payload["dataset"])
        seed = int(payload["seed"])
        n = int(payload["n_samples"])
        pruner = str(payload.get("pruner", "none"))
        retention = float(payload.get("retention", 1.0))
        if pruner == "none" and retention == 1.0:
            baselines[(model, dataset, seed, n)] = payload
        groups[(model, dataset, pruner, retention)].append(payload)

    rows: list[dict[str, str]] = []
    for (model, dataset, pruner, retention), group in sorted(groups.items()):
        token_down: list[float | None] = []
        prefill_down: list[float | None] = []
        e2e_down: list[float | None] = []
        memory_down: list[float | None] = []
        reserved_down: list[float | None] = []
        prefill_s: list[float | None] = []
        e2e_s: list[float | None] = []
        memory_mb: list[float | None] = []
        reserved_mb: list[float | None] = []
        acc: list[float | None] = []
        ece: list[float | None] = []
        seeds: list[str] = []
        ns: list[str] = []
        evidence: set[str] = set()
        for payload in sorted(group, key=lambda p: int(p["seed"])):
            seed = int(payload["seed"])
            n = int(payload["n_samples"])
            baseline = baselines.get((model, dataset, seed, n))
            seeds.append(str(seed))
            ns.append(str(n))
            if payload.get("pruner_evidence_type"):
                evidence.add(str(payload["pruner_evidence_type"]))
            token_down.append(token_reduction(payload))
            cur_prefill = f(payload, "prefill_latency_mean_s")
            cur_e2e = f(payload, "end_to_end_latency_s_per_sample")
            cur_memory = f(payload, "peak_gpu_memory_allocated_mb")
            if cur_memory is None:
                cur_memory = f(payload, "peak_gpu_memory_mb")
            cur_reserved = f(payload, "peak_gpu_memory_reserved_mb")
            base_memory = f(baseline, "peak_gpu_memory_allocated_mb")
            if base_memory is None:
                base_memory = f(baseline, "peak_gpu_memory_mb")
            base_reserved = f(baseline, "peak_gpu_memory_reserved_mb")
            prefill_s.append(cur_prefill)
            e2e_s.append(cur_e2e)
            memory_mb.append(cur_memory)
            reserved_mb.append(cur_reserved)
            prefill_down.append(reduction(cur_prefill, f(baseline, "prefill_latency_mean_s")))
            e2e_down.append(reduction(cur_e2e, f(baseline, "end_to_end_latency_s_per_sample")))
            memory_down.append(reduction(cur_memory, base_memory))
            reserved_down.append(reduction(cur_reserved, base_reserved))
            acc.append(f(payload, "accuracy"))
            ece.append(f(payload, "ece"))

        rows.append(
            {
                "Model": model,
                "Dataset": dataset,
                "Method": pruner,
                "r": f"{retention:.2f}",
                "n": ",".join(sorted(set(ns))),
                "Seeds": ",".join(sorted(set(seeds))),
                "Token_down": fmt_pct(token_down),
                "Prefill_latency_down": fmt_pct(prefill_down),
                "End_to_end_latency_down": fmt_pct(e2e_down),
                "Peak_allocated_down": fmt_pct(memory_down),
                "Peak_reserved_down": fmt_pct(reserved_down),
                "Peak_memory_down": fmt_pct(memory_down),
                "Prefill_s_per_sample": fmt_num(prefill_s, 3),
                "E2E_s_per_sample": fmt_num(e2e_s, 3),
                "Peak_allocated_MB": fmt_num(memory_mb, 1),
                "Peak_reserved_MB": fmt_num(reserved_mb, 1),
                "Peak_MB": fmt_num(memory_mb, 1),
                "Acc": fmt_num(acc, 4),
                "ECE": fmt_num(ece, 4),
                "Evidence": ",".join(sorted(evidence)) if evidence else "runner-hook",
            }
        )

    columns = [
        "Model",
        "Dataset",
        "Method",
        "r",
        "n",
        "Seeds",
        "Token_down",
        "Prefill_latency_down",
        "End_to_end_latency_down",
        "Peak_allocated_down",
        "Peak_reserved_down",
        "Peak_memory_down",
        "Prefill_s_per_sample",
        "E2E_s_per_sample",
        "Peak_allocated_MB",
        "Peak_reserved_MB",
        "Peak_MB",
        "Acc",
        "ECE",
        "Evidence",
    ]
    out_csv = ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    out_md = ROOT / args.output_md if args.output_md else out_csv.with_suffix(".md")
    note = (
        "Runner-level microbenchmark table. Reductions are relative to the matched unpruned run "
        "with the same model, dataset, seed, and sample count; positive values mean lower token count, "
        "latency, peak allocated CUDA memory, or peak reserved CUDA memory."
    )
    out_md.write_text(md(rows, columns, note), encoding="utf-8")


if __name__ == "__main__":
    main()


