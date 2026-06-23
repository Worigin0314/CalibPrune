"""Exploratory open-ended A-OKVQA direct-answer reranking.

This script intentionally avoids the multiple-choice options in A-OKVQA. It
builds a small global vocabulary from free-form direct answers, prompts the VLM
for a short phrase, and uses answer-token likelihoods only for post-hoc
candidate reranking. The setting is exploratory because multi-token answers are
approximated by first-token likelihoods, matching the paper's answer-logit
infrastructure while exposing the open-answer normalization boundary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibprune.models.loader import load_model


def normalize_answer(text: object) -> str:
    clean = str(text).strip().lower()
    clean = re.sub(r"[^a-z0-9 ]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_direct_answers(value: object) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except Exception:
            raw = [value]
    answers = [normalize_answer(item) for item in raw]
    return [answer for answer in answers if answer]


def local_aokvqa_validation() -> Any:
    from datasets import Dataset

    path = (
        ROOT
        / "data"
        / "hf_cache"
        / "HuggingFaceM4___a-okvqa"
        / "default"
        / "0.0.0"
        / "d1b0efa3a436e9101dfbde3752db7607da696c35"
        / "a-okvqa-validation.arrow"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing local A-OKVQA validation Arrow file: {path}")
    return Dataset.from_file(str(path))


def ece_from_confidence(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, int(conf.shape[0]))
    score = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (conf >= edges[idx]) & (conf <= edges[idx + 1])
        else:
            mask = (conf >= edges[idx]) & (conf < edges[idx + 1])
        if not np.any(mask):
            continue
        score += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    return float(score)


def aurc_from_confidence(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf)
    sorted_correct = correct[order].astype(float)
    n = max(1, int(sorted_correct.shape[0]))
    coverage = np.arange(1, n + 1, dtype=float) / n
    risk = 1.0 - np.cumsum(sorted_correct) / np.arange(1, n + 1, dtype=float)
    if n == 1:
        return float(risk[0])
    return float(np.trapz(risk, coverage))


def scalar_residual(base: np.ndarray, pruned: np.ndarray) -> float:
    x = base - np.mean(base, axis=1, keepdims=True)
    y = pruned - np.mean(pruned, axis=1, keepdims=True)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    alpha = float(np.sum(x * y) / denom)
    residual = np.linalg.norm(y - alpha * x)
    scale = np.linalg.norm(y)
    return float(residual / scale) if scale > 0 else 0.0


def vqa_soft_score(prediction: str, references: list[str]) -> float:
    count = sum(1 for answer in references if answer == prediction)
    return float(min(1.0, count / 3.0))


def softmax_1d(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exp = np.exp(shifted)
    return exp / max(float(np.sum(exp)), 1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--output-dir", default="results/raw/aokvqa_open_ended_exploratory")
    args = parser.parse_args()

    os.environ["CALIBPRUNE_OPEN_ENDED_RERANK"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data" / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf-cache" / "transformers"))

    ds = local_aokvqa_validation()
    all_refs = [parse_direct_answers(row["direct_answers"]) for row in ds]
    vocab_counter: Counter[str] = Counter(answer for refs in all_refs for answer in refs if len(answer.split()) <= 3)
    vocab = tuple(answer for answer, _ in vocab_counter.most_common(args.vocab_size))
    if not vocab:
        raise RuntimeError("Direct-answer vocabulary is empty.")

    eligible = [idx for idx, refs in enumerate(all_refs) if any(answer in vocab for answer in refs)]
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(np.asarray(eligible), size=min(args.n, len(eligible)), replace=False).astype(int).tolist())
    vlm = load_model(args.model)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = [("none", 1.0), ("fastv", 0.5)]
    cell_payloads: dict[str, dict[str, Any]] = {}
    logits_by_cell: dict[str, np.ndarray] = {}
    for pruner, retention in cells:
        rows = []
        logits_rows = []
        started = time.time()
        for idx in selected:
            row = ds[int(idx)]
            refs = all_refs[int(idx)]
            query = f"Question: {str(row['question']).strip()}"
            bundle = vlm.forward_with_logits(
                row["image"].convert("RGB"),
                query,
                vocab,
                pruner_name=pruner,
                retention=retention,
            )
            scores = np.asarray([float(bundle.answer_set_logits[answer]) for answer in vocab], dtype=float)
            probs = softmax_1d(scores)
            pred_idx = int(np.argmax(scores))
            pred = vocab[pred_idx]
            score = vqa_soft_score(pred, refs)
            rows.append(
                {
                    "sample_index": int(idx),
                    "prediction": pred,
                    "soft_score": score,
                    "exact_any": bool(score > 0),
                    "confidence": float(probs[pred_idx]),
                    "references": refs[:10],
                    "question": str(row["question"]),
                }
            )
            logits_rows.append(scores)
        elapsed = time.time() - started
        logits = np.vstack(logits_rows)
        logits_by_cell[f"{pruner}_r{retention:g}"] = logits
        conf = np.asarray([row["confidence"] for row in rows], dtype=float)
        correct = np.asarray([row["exact_any"] for row in rows], dtype=bool)
        soft_scores = np.asarray([row["soft_score"] for row in rows], dtype=float)
        meta = rows[-1] if rows else {}
        token_down = 0.0
        if getattr(vlm, "name", args.model) and rows:
            # Re-run metadata is not stored per row; token reduction is inferred
            # from the last forward metadata available in the returned bundle.
            pass
        payload = {
            "model": args.model,
            "dataset": "aokvqa_direct_answer",
            "setting": "open_ended_candidate_rerank",
            "pruner": pruner,
            "retention": retention,
            "n_samples": len(rows),
            "vocab_size": len(vocab),
            "seed": args.seed,
            "mean_vqa_soft_score": float(np.mean(soft_scores)) if len(rows) else 0.0,
            "exact_any_accuracy": float(np.mean(correct)) if len(rows) else 0.0,
            "ece_exact": ece_from_confidence(conf, correct),
            "aurc_exact": aurc_from_confidence(conf, correct),
            "mean_confidence": float(np.mean(conf)) if len(rows) else 0.0,
            "seconds_per_sample": float(elapsed / max(1, len(rows))),
            "answer_vocab": list(vocab),
            "selected_indices": selected,
            "prompt_mode": "open_short_phrase_candidate_rerank",
            "likelihood_note": "first-token answer likelihood over a global direct-answer vocabulary",
            "rows_preview": rows[:20],
        }
        if pruner != "none":
            payload["token_reduction_target"] = float(1.0 - retention)
        cell_payloads[f"{pruner}_r{retention:g}"] = payload
        out_json = output_dir / f"{args.model}-aokvqa-open-{pruner}-r{retention:g}-n{len(rows)}-seed{args.seed}.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "none_r1" in logits_by_cell and "fastv_r0.5" in logits_by_cell:
        cell_payloads["fastv_r0.5"]["scalar_residual_vs_none"] = scalar_residual(
            logits_by_cell["none_r1"], logits_by_cell["fastv_r0.5"]
        )

    summary_rows = []
    base = cell_payloads["none_r1"]
    for key in ["none_r1", "fastv_r0.5"]:
        payload = cell_payloads[key]
        summary_rows.append(
            {
                "Model": payload["model"],
                "Dataset": payload["dataset"],
                "Pruner": payload["pruner"],
                "r": f"{float(payload['retention']):.2f}",
                "n": str(payload["n_samples"]),
                "Vocab": str(payload["vocab_size"]),
                "SoftScore": f"{float(payload['mean_vqa_soft_score']):.4f}",
                "SoftDelta": f"{float(payload['mean_vqa_soft_score']) - float(base['mean_vqa_soft_score']):+.4f}",
                "ExactAny": f"{float(payload['exact_any_accuracy']):.4f}",
                "ECE": f"{float(payload['ece_exact']):.4f}",
                "AURC": f"{float(payload['aurc_exact']):.4f}",
                "Conf": f"{float(payload['mean_confidence']):.4f}",
                "Residual": f"{float(payload.get('scalar_residual_vs_none', 0.0)):.4f}" if key != "none_r1" else "",
                "SecPerSample": f"{float(payload['seconds_per_sample']):.3f}",
            }
        )
    columns = list(summary_rows[0].keys())
    out_csv = ROOT / "results" / "tables" / "aokvqa_open_ended_exploratory.csv"
    out_md = ROOT / "results" / "tables" / "aokvqa_open_ended_exploratory.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "Bounded A-OKVQA direct-answer candidate reranking. The prompt does not expose the answer vocabulary; candidates are used only for likelihood-based post-hoc scoring.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join([":--" for _ in columns]) + "|",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
