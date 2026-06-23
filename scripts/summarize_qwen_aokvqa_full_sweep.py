"""Summarize Qwen2-VL A-OKVQA full-validation FastV retention cells."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw" / "qwen2vl_aokvqa_full_r05"
OUT_CSV = ROOT / "results" / "tables" / "qwen2vl_aokvqa_full_sweep.csv"
OUT_MD = ROOT / "results" / "tables" / "qwen2vl_aokvqa_full_sweep.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_residual(base_logits: np.ndarray, pruned_logits: np.ndarray) -> float:
    x = base_logits - np.mean(base_logits, axis=1, keepdims=True)
    y = pruned_logits - np.mean(pruned_logits, axis=1, keepdims=True)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    alpha = float(np.sum(x * y) / denom)
    residual = float(np.linalg.norm(y - alpha * x))
    scale = float(np.linalg.norm(y))
    return residual / scale if scale > 0 else 0.0


def logits_for(payload: dict) -> np.ndarray | None:
    path = payload.get("logits_path")
    if not path:
        return None
    npz_path = ROOT / str(path)
    if not npz_path.exists():
        return None
    return np.load(npz_path, allow_pickle=True)["logits"].astype(float)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    payloads = sorted(
        (load_json(path) for path in RAW_DIR.glob("qwen2vl_2b-aokvqa-test-*-n1145-seed20260628.json")),
        key=lambda item: (item["pruner"] != "none", float(item["retention"])),
    )
    baseline = next(item for item in payloads if item["pruner"] == "none")
    base_logits = logits_for(baseline)
    rows: list[dict[str, str]] = []
    for payload in payloads:
        retention = float(payload["retention"])
        logits = logits_for(payload)
        residual = ""
        if payload["pruner"] != "none" and base_logits is not None and logits is not None:
            residual = f"{scalar_residual(base_logits, logits):.4f}"
        original = float(payload["last_sample_metadata"].get("num_visual_tokens_original", 0))
        kept = float(payload["last_sample_metadata"].get("num_visual_tokens_kept", original))
        token_down = 0.0 if original <= 0 else 1.0 - kept / original
        rows.append(
            {
                "Method": str(payload["pruner"]),
                "r": f"{retention:.2f}",
                "n": str(payload["n_samples"]),
                "Accuracy": f"{float(payload['accuracy']):.4f}",
                "Acc_minus_chance": f"{float(payload['accuracy']) - 0.25:+.4f}",
                "ECE": f"{float(payload['ece']):.4f}",
                "AURC": f"{float(payload['aurc']):.4f}",
                "Conf": f"{float(payload['max_softmax_mean']):.4f}",
                "Residual": residual,
                "Token_down": pct(token_down),
                "E2E_s": f"{float(payload['end_to_end_latency_s_per_sample']):.3f}",
                "Peak_alloc_MB": f"{float(payload['peak_gpu_memory_allocated_mb']):.1f}",
                "Peak_reserved_MB": f"{float(payload['peak_gpu_memory_reserved_mb']):.1f}",
            }
        )

    columns = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Qwen2-VL A-OKVQA full-validation FastV retention sweep. Acc-minus-chance uses 0.25 for the four-choice option-letter setting.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(":--" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
