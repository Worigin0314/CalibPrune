"""Run a resumable POPE-lite grid while reusing one loaded model instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.data.loaders import VQASample, build, dataset_population_size
from calibprune.data.splits import default_split_path, ensure_split_file, indices_hash, limit_indices, load_split_indices
from calibprune.metrics.calibration import metrics_from_logits
from calibprune.metrics.selective import aurc, selective_accuracy_at_coverages
from calibprune.models.loader import VLM, load_model
from calibprune.models.vlm_wrapper import answer_logits_to_fixed_array, merge_answer_spaces
from calibprune.pipelines.run_eval import REAL_MODEL_PRUNERS


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def grid_id(pruners: list[str], retentions: list[float]) -> str:
    payload = json.dumps({"pruners": pruners, "retentions": retentions}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def output_reusable(path: Path, expected_hash: str | None, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("sample_indices_hash") == expected_hash and payload.get("n_samples") == expected_n


def summarize_metadata(metadata: dict[str, object] | None) -> dict[str, object] | None:
    if not metadata:
        return None
    out = dict(metadata)
    kept_indices = out.pop("kept_indices", None)
    if kept_indices is not None:
        kept = list(kept_indices)
        out["kept_indices_count"] = len(kept)
        out["kept_indices_head"] = kept[:8]
        out["kept_indices_tail"] = kept[-8:]
    return out


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _reset_cuda_peak_memory() -> None:
    if not _cuda_available():
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    except Exception:
        return


def _cuda_peak_memory_mb() -> float | None:
    if not _cuda_available():
        return None
    try:
        import torch

        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    except Exception:
        return None


def _cuda_peak_reserved_memory_mb() -> float | None:
    if not _cuda_available():
        return None
    try:
        import torch

        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
    except Exception:
        return None


def _cuda_device_name() -> str | None:
    if not _cuda_available():
        return None
    try:
        import torch

        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return None


def evaluate_cell(
    vlm: VLM,
    samples: list[VQASample],
    *,
    model: str,
    dataset: str,
    split: str,
    subset_name: str,
    pruner: str,
    retention: float,
    seed: int,
    split_file: Path,
    split_seed: int | None,
    sample_hash: str | None,
    run_id: str,
    output_path: Path,
    save_logits: bool,
    paper_eligible: bool,
    warmup_samples: int,
) -> dict[str, object]:
    logits_rows = []
    labels = []
    sample_latencies: list[float] = []
    last_metadata: dict[str, object] | None = None
    answer_space = merge_answer_spaces([sample.answer_choices for sample in samples])
    if warmup_samples > 0 and samples:
        for sample in samples[:warmup_samples]:
            vlm.forward_with_logits(
                sample.image,
                sample.question,
                sample.answer_choices,
                pruner_name=pruner,
                retention=retention,
            )
    _reset_cuda_peak_memory()
    started = time.time()
    for sample in samples:
        sample_started = time.perf_counter()
        bundle = vlm.forward_with_logits(
            sample.image,
            sample.question,
            sample.answer_choices,
            pruner_name=pruner,
            retention=retention,
        )
        sample_latencies.append(time.perf_counter() - sample_started)
        if bundle.answer_set_logits is None:
            raise RuntimeError("forward_with_logits must provide answer_set_logits for evaluation.")
        logits_rows.append(answer_logits_to_fixed_array(bundle.answer_set_logits, sample.answer_choices, answer_space))
        labels.append(answer_space.index(sample.gold_answer))
        last_metadata = summarize_metadata(bundle.metadata)

    elapsed_seconds = time.time() - started
    peak_memory_mb = _cuda_peak_memory_mb()
    peak_reserved_memory_mb = _cuda_peak_reserved_memory_mb()
    logits = np.vstack(logits_rows)
    labels_array = np.asarray(labels, dtype=np.int64)
    metric_payload = metrics_from_logits(logits, labels_array)
    metric_payload["aurc"] = aurc(logits, labels_array)
    metric_payload["aurc_margin"] = aurc(logits, labels_array, score="margin")
    metric_payload["aurc_entropy"] = aurc(logits, labels_array, score="entropy")
    metric_payload["selective_acc"] = selective_accuracy_at_coverages(logits, labels_array)
    metric_payload["selective_acc_margin"] = selective_accuracy_at_coverages(logits, labels_array, score="margin")
    metric_payload["selective_acc_entropy"] = selective_accuracy_at_coverages(logits, labels_array, score="entropy")
    sample_indices = np.asarray(
        [sample.sample_index if sample.sample_index is not None else -1 for sample in samples],
        dtype=np.int64,
    )

    result: dict[str, object] = {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "split": split,
        "subset_name": subset_name,
        "pruner": pruner,
        "retention": retention,
        "calibrator": "none",
        "seed": seed,
        "n_samples": int(labels_array.shape[0]),
        "sample_indices_hash": sample_hash,
        "sample_indices_count": int(sample_indices.shape[0]),
        "split_file": str(split_file),
        "split_seed": split_seed,
        "source": samples[0].source if samples else "unknown",
        "paper_eligible": paper_eligible,
        "runner": "pope_lite_inprocess",
        "batch_size": 1,
        "warmup_samples": int(warmup_samples),
        "gpu_name": _cuda_device_name(),
        "latency_mode": "single-sample closed-set next-token scoring",
        "decode_latency_mean_s": None,
        "decode_latency_note": "not measured because closed-set answer scoring uses one next-token pass rather than autoregressive decoding",
        "wall_clock_min": round(elapsed_seconds / 60.0, 4),
        "end_to_end_latency_s_per_sample": float(elapsed_seconds / max(1, labels_array.shape[0])),
        "prefill_latency_mean_s": float(np.mean(sample_latencies)) if sample_latencies else None,
        "prefill_latency_p50_s": float(np.median(sample_latencies)) if sample_latencies else None,
        "prefill_latency_p95_s": float(np.quantile(sample_latencies, 0.95)) if sample_latencies else None,
        "peak_gpu_memory_mb": peak_memory_mb,
        "peak_gpu_memory_allocated_mb": peak_memory_mb,
        "peak_gpu_memory_reserved_mb": peak_reserved_memory_mb,
        **metric_payload,
    }
    if last_metadata and last_metadata.get("pruner_evidence_type"):
        result["pruner_evidence_type"] = str(last_metadata["pruner_evidence_type"])
    if save_logits:
        logits_path = Path("results/raw/logits") / f"{run_id}.npz"
        save_npz_atomic(
            logits_path,
            logits=logits,
            labels=labels_array,
            answer_choices=np.asarray(answer_space, dtype=object),
            sample_indices=sample_indices,
            subset_name=np.asarray(subset_name, dtype=object),
            sample_indices_hash=np.asarray(sample_hash or "", dtype=object),
            run_id=np.asarray(run_id, dtype=object),
            model=np.asarray(model, dtype=object),
            dataset=np.asarray(dataset, dtype=object),
            pruner=np.asarray(pruner, dtype=object),
            retention=np.asarray(retention, dtype=np.float64),
            seed=np.asarray(seed, dtype=np.int64),
        )
        result["logits_path"] = str(logits_path)
    if last_metadata:
        result["last_sample_metadata"] = last_metadata
    write_json_atomic(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pruners", default="none,fastv")
    parser.add_argument("--retentions", default="0.5")
    parser.add_argument("--n-cal", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--n-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="results/raw/pope_lite_inprocess")
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only the test subset; useful for full official-task validation runs without reserving calibration samples.",
    )
    parser.add_argument("--save-logits", dest="save_logits", action="store_true", default=True)
    parser.add_argument("--no-save-logits", dest="save_logits", action="store_false")
    parser.add_argument("--paper-eligible", action="store_true")
    parser.add_argument("--warmup-samples", type=int, default=0, help="Number of samples to run before timed measurement in each cell.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cells even when matching outputs already exist.")
    args = parser.parse_args()

    if args.model != "toy_vlm":
        unknown = set(parse_csv(args.pruners)).difference(REAL_MODEL_PRUNERS)
        if unknown:
            raise RuntimeError(f"Unsupported real-model pruners: {sorted(unknown)}")

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache" / "transformers"))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = args.n_total or dataset_population_size(
        args.dataset,
        args.split,
        offline_fixture=args.offline_fixture,
    )
    split_dataset_name = f"{args.dataset}_fixture" if args.offline_fixture else args.dataset
    if args.test_only:
        if args.n_test <= 0:
            raise ValueError("--n-test must be positive for --test-only runs.")
        if args.n_test > n_total:
            raise ValueError(f"--n-test={args.n_test} exceeds dataset population {n_total}.")
        split_file = (
            PROJECT_ROOT / args.split_file
            if args.split_file
            else PROJECT_ROOT / "data" / "splits" / f"{split_dataset_name}_fulltest_seed{args.seed}.json"
        )
        split_payload = {
            "dataset": split_dataset_name,
            "n_total": int(n_total),
            "n_cal": 0,
            "seed": int(args.seed),
            "order": "sequential_full_test",
            "cal": [],
            "test": list(range(int(n_total))),
        }
        write_json_atomic(split_file, split_payload)
        split_seed = int(args.seed)
        split_indices = {"test": split_payload["test"]}
        subset_specs = [("test", args.n_test)]
    else:
        split_n_cal = max(args.n_cal, 500)
        split_file = (
            PROJECT_ROOT / args.split_file
            if args.split_file
            else PROJECT_ROOT / default_split_path(split_dataset_name, split_n_cal, args.seed)
        )
        ensure_split_file(split_file, dataset=split_dataset_name, n_total=n_total, n_cal=split_n_cal, seed=args.seed)
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
        split_seed = int(split_payload.get("seed", args.seed))
        split_indices = {
            "cal": load_split_indices(split_file, "cal")[0],
            "test": load_split_indices(split_file, "test")[0],
        }
        subset_specs = [("cal", args.n_cal), ("test", args.n_test)]
    subset_indices = {
        subset: limit_indices(split_indices[subset], n_samples)
        for subset, n_samples in subset_specs
    }
    subset_hashes = {subset: indices_hash(indices) for subset, indices in subset_indices.items()}

    pruners = parse_csv(args.pruners)
    retentions = [float(value) for value in parse_csv(args.retentions)]
    tasks = []
    for subset_name, n_samples in subset_specs:
        for pruner in pruners:
            cell_retentions = [1.0] if pruner == "none" else retentions
            for retention in cell_retentions:
                run_id = (
                    f"{args.model}-{args.dataset}-{subset_name}-{pruner}-"
                    f"r{retention:g}-n{n_samples}-seed{args.seed}"
                )
                output_path = output_dir / f"{run_id}.json"
                tasks.append(
                    {
                        "subset_name": subset_name,
                        "n_samples": n_samples,
                        "pruner": pruner,
                        "retention": retention,
                        "run_id": run_id,
                        "output_path": str(output_path),
                        "sample_indices_hash": subset_hashes[subset_name],
                    }
                )

    manifest = {
        "runner": "pope_lite_inprocess",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "split_file": str(split_file),
        "n_total": n_total,
        "n_cal": 0 if args.test_only else args.n_cal,
        "n_test": args.n_test,
        "test_only": bool(args.test_only),
        "seed": args.seed,
        "tasks": tasks,
    }
    manifest_path = (
        output_dir
        / f"manifest-inprocess-seed{args.seed}-cal{0 if args.test_only else args.n_cal}-test{args.n_test}-grid{grid_id(pruners, retentions)}.json"
    )
    write_json_atomic(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "task_count": len(tasks), "manifest": str(manifest_path)}, indent=2))
        return 0

    pending = (
        tasks
        if args.force
        else [
            task
            for task in tasks
            if not output_reusable(Path(str(task["output_path"])), str(task["sample_indices_hash"]), int(task["n_samples"]))
        ]
    )
    for task in tasks:
        if task not in pending:
            print(json.dumps({"skipped": True, "run_id": task["run_id"], "output": task["output_path"]}, indent=2))
    if not pending:
        return 0

    samples_by_subset = {
        subset: list(
            build(
                args.dataset,
                args.split,
                n=n_samples,
                offline_fixture=args.offline_fixture,
                indices=subset_indices[subset],
            )
        )
        for subset, n_samples in subset_specs
    }
    vlm = load_model(args.model)
    failures = []
    for task in pending:
        try:
            result = evaluate_cell(
                vlm,
                samples_by_subset[str(task["subset_name"])],
                model=args.model,
                dataset=args.dataset,
                split=args.split,
                subset_name=str(task["subset_name"]),
                pruner=str(task["pruner"]),
                retention=float(task["retention"]),
                seed=args.seed,
                split_file=split_file,
                split_seed=split_seed,
                sample_hash=str(task["sample_indices_hash"]),
                run_id=str(task["run_id"]),
                output_path=Path(str(task["output_path"])),
                save_logits=args.save_logits,
                paper_eligible=args.paper_eligible,
                warmup_samples=max(0, args.warmup_samples),
            )
            print(json.dumps(result, indent=2))
        except Exception as exc:
            failure = {
                "run_id": task["run_id"],
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(json.dumps(failure, indent=2))
    if failures:
        failure_path = output_dir / f"failures-inprocess-seed{args.seed}-cal{0 if args.test_only else args.n_cal}-test{args.n_test}.json"
        write_json_atomic(failure_path, {"failures": failures})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




