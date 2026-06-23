"""Run one evaluation cell."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calibprune.data.loaders import build
from calibprune.data.splits import indices_hash, limit_indices, load_split_indices
from calibprune.metrics.calibration import metrics_from_logits
from calibprune.metrics.selective import aurc, selective_accuracy_at_coverages
from calibprune.models.loader import load_model
from calibprune.models.vlm_wrapper import answer_logits_to_fixed_array, merge_answer_spaces
from calibprune.pruners.base import build_pruner


REAL_MODEL_PRUNERS = {
    "none",
    "random",
    "uniform",
    "feature_norm",
    "fastv",
    "sparsevlm",
    "visionzip",
    "pyramiddrop",
    "vtw",
}
MODEL_HOOK_RUN_EVAL_PRUNERS = {"fastv", "sparsevlm", "visionzip", "pyramiddrop", "vtw"}
GENERIC_RUN_EVAL_PRUNERS = REAL_MODEL_PRUNERS - MODEL_HOOK_RUN_EVAL_PRUNERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--pruner", default="none")
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--calibrator", default="none")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--subset-name", default="eval")
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--paper-eligible", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _save_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.model != "toy_vlm" and args.pruner not in REAL_MODEL_PRUNERS:
        raise RuntimeError(
            "Literature pruner hooks are not wired yet for real VLMs. "
            "Use one of {none, random, uniform, feature_norm, fastv, sparsevlm, visionzip, pyramiddrop, vtw} "
            "for real model smoke tests."
        )
    selected_indices = None
    split_payload = None
    if args.split_file:
        split_indices, split_payload = load_split_indices(args.split_file, args.subset_name)
        selected_indices = limit_indices(split_indices, args.n)
    sample_hash = indices_hash(selected_indices)

    stable_run_id = args.run_id or (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{args.model}-{args.dataset}-"
        f"{args.subset_name}-{args.pruner}-r{args.retention:g}-{args.calibrator}"
    )
    out_path = Path(args.output) if args.output else Path("results/raw") / f"{stable_run_id}.json"
    if args.skip_existing and out_path.exists():
        print(json.dumps({"skipped": True, "output": str(out_path)}, indent=2))
        return

    samples = build(
        args.dataset,
        args.split,
        n=args.n,
        offline_fixture=args.offline_fixture,
        indices=selected_indices,
    )
    vlm = load_model(args.model)
    if args.pruner in GENERIC_RUN_EVAL_PRUNERS:
        pruner = build_pruner(args.pruner)
        _ = pruner
    elif args.pruner not in MODEL_HOOK_RUN_EVAL_PRUNERS:
        raise RuntimeError(
            f"Pruner {args.pruner!r} is not available in the executable pipeline yet."
        )

    logits_rows = []
    labels = []
    answer_space = merge_answer_spaces([sample.answer_choices for sample in samples])
    for sample in samples:
        bundle = vlm.forward_with_logits(
            sample.image,
            sample.question,
            sample.answer_choices,
            pruner_name=args.pruner,
            retention=args.retention,
        )
        if bundle.answer_set_logits is None:
            raise RuntimeError("forward_with_logits must provide answer_set_logits for evaluation.")
        logits_rows.append(answer_logits_to_fixed_array(bundle.answer_set_logits, sample.answer_choices, answer_space))
        labels.append(answer_space.index(sample.gold_answer))

    logits = np.vstack(logits_rows)
    labels_array = np.asarray(labels, dtype=np.int64)
    metrics = metrics_from_logits(logits, labels_array)
    metrics["aurc"] = aurc(logits, labels_array)
    metrics["aurc_margin"] = aurc(logits, labels_array, score="margin")
    metrics["aurc_entropy"] = aurc(logits, labels_array, score="entropy")
    metrics["selective_acc"] = selective_accuracy_at_coverages(logits, labels_array)
    metrics["selective_acc_margin"] = selective_accuracy_at_coverages(logits, labels_array, score="margin")
    metrics["selective_acc_entropy"] = selective_accuracy_at_coverages(logits, labels_array, score="entropy")

    result = {
        "run_id": stable_run_id,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "subset_name": args.subset_name,
        "pruner": args.pruner,
        "retention": args.retention,
        "calibrator": args.calibrator,
        "seed": args.seed,
        "n_samples": int(labels_array.shape[0]),
        "sample_indices_hash": sample_hash,
        "sample_indices_count": len(selected_indices) if selected_indices is not None else None,
        "split_file": args.split_file,
        "split_seed": split_payload.get("seed") if split_payload else None,
        "source": samples[0].source if samples else "unknown",
        "paper_eligible": bool(args.paper_eligible),
        "wall_clock_min": round((time.time() - started) / 60.0, 4),
        **metrics,
    }
    if args.save_logits:
        logits_path = Path("results/raw/logits") / f"{stable_run_id}.npz"
        sample_indices = np.asarray(
            [sample.sample_index if sample.sample_index is not None else -1 for sample in samples],
            dtype=np.int64,
        )
        _save_npz_atomic(
            logits_path,
            logits=logits,
            labels=labels_array,
            answer_choices=np.asarray(answer_space, dtype=object),
            sample_indices=sample_indices,
            subset_name=np.asarray(args.subset_name, dtype=object),
            sample_indices_hash=np.asarray(sample_hash or "", dtype=object),
            run_id=np.asarray(stable_run_id, dtype=object),
            model=np.asarray(args.model, dtype=object),
            dataset=np.asarray(args.dataset, dtype=object),
            pruner=np.asarray(args.pruner, dtype=object),
            retention=np.asarray(args.retention, dtype=np.float64),
            seed=np.asarray(args.seed, dtype=np.int64),
        )
        result["logits_path"] = str(logits_path)
    if samples and "bundle" in locals() and bundle.metadata:
        metadata = dict(bundle.metadata)
        kept_indices = metadata.pop("kept_indices", None)
        if kept_indices is not None:
            metadata["kept_indices_count"] = len(kept_indices)
            metadata["kept_indices_head"] = kept_indices[:8]
            metadata["kept_indices_tail"] = kept_indices[-8:]
        result["last_sample_metadata"] = metadata

    output_text = json.dumps(result, indent=2)
    _write_text_atomic(out_path, output_text)
    print(output_text)


if __name__ == "__main__":
    main()
