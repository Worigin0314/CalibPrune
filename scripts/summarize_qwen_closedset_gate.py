"""Summarize whether Qwen2-VL closed-set scoring is above random chance."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibprune.data.loaders import build


def softmax_argmax(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=1)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_path"] = str(path)
    return payload


def collect(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        full = str((ROOT / pattern).resolve()) if not Path(pattern).is_absolute() else pattern
        paths.extend(Path(p) for p in glob.glob(full))
    return sorted({p.resolve() for p in paths})


def fmt_ci(value: float, low: float, high: float) -> str:
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def bootstrap_margin(correct: np.ndarray, chance: np.ndarray, *, seed: int = 0, n_resamples: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    delta = correct.astype(float) - chance.astype(float)
    n = delta.shape[0]
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(delta[idx]))
    return float(np.mean(delta)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(payload: dict[str, Any], *, n_resamples: int) -> dict[str, str]:
    logits_path = ROOT / str(payload["logits_path"])
    data = np.load(logits_path, allow_pickle=True)
    logits = np.asarray(data["logits"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    sample_indices = np.asarray(data["sample_indices"], dtype=int)
    samples = list(build(str(payload["dataset"]), str(payload["split"]), n=len(sample_indices), indices=sample_indices.tolist()))
    if len(samples) != len(sample_indices):
        raise RuntimeError("Could not reconstruct sample choices for chance baseline.")
    chance = np.asarray([1.0 / len(sample.answer_choices) for sample in samples], dtype=float)
    pred = softmax_argmax(logits)
    correct = pred == labels
    margin, low, high = bootstrap_margin(correct, chance, seed=int(payload["seed"]), n_resamples=n_resamples)
    meta = payload.get("last_sample_metadata") or {}
    original = float(meta.get("num_visual_tokens_original", 0) or 0)
    kept = float(meta.get("num_visual_tokens_kept", original) or original)
    token_down = 100.0 * (1.0 - kept / original) if original > 0 else 0.0
    return {
        "Model": str(payload["model"]),
        "Dataset": str(payload["dataset"]),
        "Method": str(payload.get("pruner", "none")),
        "r": f"{float(payload.get('retention', 1.0)):.2f}",
        "n": str(int(payload["n_samples"])),
        "Accuracy": f"{float(payload['accuracy']):.4f}",
        "Mean_random_chance": f"{float(np.mean(chance)):.4f}",
        "Acc_minus_chance_95CI": fmt_ci(margin, low, high),
        "ECE": f"{float(payload['ece']):.4f}",
        "AURC": f"{float(payload['aurc']):.4f}",
        "Token_down": f"{token_down:.1f}%",
        "Verbalizer": str(meta.get("answer_verbalizer_mode", "")),
        "GPU": str(payload.get("gpu_name") or ""),
    }


def write_md(rows: list[dict[str, str]], columns: list[str], path: Path) -> None:
    lines = [
        "Qwen2-VL closed-set sanity gate. Acc-minus-chance uses a paired bootstrap over per-sample correctness minus the uniform random chance for that sample's answer set.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join([":--" for _ in columns]) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-glob", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()

    payloads = [load_json(path) for path in collect(args.result_glob)]
    payloads = [p for p in payloads if p.get("subset_name", p.get("split")) == "test" and p.get("logits_path")]
    if not payloads:
        raise RuntimeError("No matching Qwen closed-set payloads found.")
    rows = [summarize(payload, n_resamples=args.n_resamples) for payload in payloads]
    columns = ["Model", "Dataset", "Method", "r", "n", "Accuracy", "Mean_random_chance", "Acc_minus_chance_95CI", "ECE", "AURC", "Token_down", "Verbalizer", "GPU"]
    out_csv = ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    out_md = ROOT / args.output_md if args.output_md else out_csv.with_suffix(".md")
    write_md(rows, columns, out_md)


if __name__ == "__main__":
    main()

