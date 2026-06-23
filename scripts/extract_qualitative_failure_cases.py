"""Extract qualitative confidence-drift cases from paired saved logits."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibprune.data.loaders import build

UNPRUNED = ROOT / "results" / "raw" / "logits" / "llava15_7b_4bit-pope-test-none-r1-n512-seed20260616.npz"
PRUNED = ROOT / "results" / "raw" / "logits" / "llava15_7b_4bit-pope-test-fastv-r0.5-n512-seed20260616.npz"
OUT_CSV = ROOT / "results" / "tables" / "qualitative_failure_cases_llava_pope_fastv.csv"
OUT_MD = ROOT / "results" / "tables" / "qualitative_failure_cases_llava_pope_fastv.md"
OUT_PDF = ROOT / "paper" / "figures" / "qualitative_confidence_cases.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "qualitative_confidence_cases.png"


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def truncate(text: str, limit: int = 82) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def pick_cases(logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray) -> list[tuple[str, int]]:
    p_a = softmax(logits_a)
    p_b = softmax(logits_b)
    pred_a = np.argmax(p_a, axis=1)
    pred_b = np.argmax(p_b, axis=1)
    conf_a = np.max(p_a, axis=1)
    conf_b = np.max(p_b, axis=1)
    delta = conf_b - conf_a
    centered_a = logits_a - logits_a.mean(axis=1, keepdims=True)
    centered_b = logits_b - logits_b.mean(axis=1, keepdims=True)
    denom = float(np.sum(centered_a * centered_a))
    alpha = float(np.sum(centered_b * centered_a) / denom) if denom > 0 else 1.0
    residual = np.linalg.norm(centered_b - alpha * centered_a, axis=1) / (np.linalg.norm(centered_b, axis=1) + 1e-12)

    cases: list[tuple[str, int]] = []
    same_correct = np.where((pred_a == pred_b) & (pred_a == labels))[0]
    same_wrong = np.where((pred_a == pred_b) & (pred_a != labels))[0]
    flips = np.where(pred_a != pred_b)[0]
    if same_correct.size:
        cases.append(("same correct, confidence sharper", int(same_correct[np.argmax(delta[same_correct])])))
        cases.append(("same correct, confidence flatter", int(same_correct[np.argmin(delta[same_correct])])))
    if flips.size:
        cases.append(("answer flip after pruning", int(flips[np.argmax(np.abs(delta[flips]))])))
    pool = same_wrong if same_wrong.size else np.arange(logits_a.shape[0])
    cases.append(("high scalar residual", int(pool[np.argmax(residual[pool])])))
    seen = set()
    unique: list[tuple[str, int]] = []
    for label, idx in cases:
        if idx not in seen:
            unique.append((label, idx))
            seen.add(idx)
    return unique[:4]


def main() -> None:
    base = load_npz(UNPRUNED)
    pruned = load_npz(PRUNED)
    logits_a = np.asarray(base["logits"], dtype=float)
    logits_b = np.asarray(pruned["logits"], dtype=float)
    labels = np.asarray(base["labels"], dtype=int)
    choices = [str(x) for x in base["answer_choices"].tolist()]
    sample_indices = np.asarray(base["sample_indices"], dtype=int)
    if not np.array_equal(sample_indices, np.asarray(pruned["sample_indices"], dtype=int)):
        raise RuntimeError("Unpruned and pruned logits do not use the same sample order.")

    samples = list(build("pope", "test", n=len(sample_indices), indices=sample_indices.tolist()))
    p_a = softmax(logits_a)
    p_b = softmax(logits_b)
    pred_a = np.argmax(p_a, axis=1)
    pred_b = np.argmax(p_b, axis=1)
    conf_a = np.max(p_a, axis=1)
    conf_b = np.max(p_b, axis=1)
    centered_a = logits_a - logits_a.mean(axis=1, keepdims=True)
    centered_b = logits_b - logits_b.mean(axis=1, keepdims=True)
    alpha = float(np.sum(centered_b * centered_a) / max(np.sum(centered_a * centered_a), 1e-12))
    residual = np.linalg.norm(centered_b - alpha * centered_a, axis=1) / (np.linalg.norm(centered_b, axis=1) + 1e-12)

    rows = []
    for case_label, idx in pick_cases(logits_a, logits_b, labels):
        sample = samples[idx]
        rows.append(
            {
                "case": case_label,
                "sample_index": str(int(sample_indices[idx])),
                "question": truncate(sample.question),
                "gold": choices[int(labels[idx])],
                "unpruned_pred": choices[int(pred_a[idx])],
                "fastv_pred": choices[int(pred_b[idx])],
                "unpruned_conf": f"{conf_a[idx]:.3f}",
                "fastv_conf": f"{conf_b[idx]:.3f}",
                "conf_delta": f"{conf_b[idx] - conf_a[idx]:+.3f}",
                "scalar_residual": f"{residual[idx]:.3f}",
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "sample_index",
        "question",
        "gold",
        "unpruned_pred",
        "fastv_pred",
        "unpruned_conf",
        "fastv_conf",
        "conf_delta",
        "scalar_residual",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Qualitative cases extracted from paired LLaVA/POPE logits for FastV r=0.5, seed 20260616.",
        "",
        "| Case | idx | Gold | Unpruned | FastV | Conf delta | Residual | Question |",
        "|:--|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['sample_index']} | {row['gold']} | "
            f"{row['unpruned_pred']} ({row['unpruned_conf']}) | {row['fastv_pred']} ({row['fastv_conf']}) | "
            f"{row['conf_delta']} | {row['scalar_residual']} | {row['question']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels_plot = [row["case"].replace(", ", "\n") for row in rows]
    y = np.arange(len(rows))
    base_conf = np.array([float(row["unpruned_conf"]) for row in rows])
    pruned_conf = np.array([float(row["fastv_conf"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.barh(y + 0.18, base_conf, height=0.32, color="#999999", label="unpruned")
    ax.barh(y - 0.18, pruned_conf, height=0.32, color="#0072B2", label="FastV r=0.5")
    ax.set_yticks(y, labels_plot, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("max-softmax confidence")
    ax.set_title("Paired qualitative confidence-drift cases")
    ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)


if __name__ == "__main__":
    main()
