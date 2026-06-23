"""Deterministic dataset split helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPLIT_SEED = 20260616


def default_split_path(
    dataset: str,
    n_cal: int,
    seed: int,
    output_dir: str | Path = "data/splits",
) -> Path:
    suffix = "" if seed == DEFAULT_SPLIT_SEED else f"_seed{seed}"
    return Path(output_dir) / f"{dataset}_cal{n_cal}{suffix}.json"


def make_split_indices(n_total: int, n_cal: int, seed: int) -> dict[str, Any]:
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if not 0 < n_cal < n_total:
        raise ValueError("n_cal must be in (0, n_total).")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    return {
        "n_total": int(n_total),
        "n_cal": int(n_cal),
        "seed": int(seed),
        "order": "shuffled",
        "cal": list(map(int, indices[:n_cal])),
        "test": list(map(int, indices[n_cal:])),
    }


def write_split_file(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def validate_split_payload(
    payload: dict[str, Any],
    *,
    path: str | Path,
    dataset: str,
    n_total: int,
    n_cal: int,
    seed: int,
) -> None:
    label = str(path)
    if payload.get("dataset") != dataset:
        raise ValueError(f"{label} dataset mismatch: expected {dataset!r}, got {payload.get('dataset')!r}.")
    if int(payload.get("n_total", -1)) != int(n_total):
        raise ValueError(f"{label} n_total mismatch: expected {n_total}, got {payload.get('n_total')}.")
    if int(payload.get("n_cal", -1)) != int(n_cal):
        raise ValueError(f"{label} n_cal mismatch: expected {n_cal}, got {payload.get('n_cal')}.")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"{label} seed mismatch: expected {seed}, got {payload.get('seed')}.")
    cal = [int(idx) for idx in payload.get("cal", [])]
    test = [int(idx) for idx in payload.get("test", [])]
    if len(cal) != n_cal:
        raise ValueError(f"{label} cal size mismatch: expected {n_cal}, got {len(cal)}.")
    if set(cal).intersection(test):
        raise ValueError(f"{label} has overlapping cal/test indices.")


def ensure_split_file(
    path: str | Path,
    *,
    dataset: str,
    n_total: int,
    n_cal: int,
    seed: int,
) -> Path:
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_split_payload(
            payload,
            path=path,
            dataset=dataset,
            n_total=n_total,
            n_cal=n_cal,
            seed=seed,
        )
        return path
    payload = make_split_indices(n_total=n_total, n_cal=n_cal, seed=seed)
    payload.update({"dataset": dataset})
    return write_split_file(path, payload)


def load_split_indices(path: str | Path, subset: str) -> tuple[list[int], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if subset not in payload:
        raise KeyError(f"Subset {subset!r} is not present in {path}.")
    indices = [int(idx) for idx in payload[subset]]
    return indices, payload


def limit_indices(indices: list[int] | None, n: int | None) -> list[int] | None:
    if indices is None:
        return None
    if n is None:
        return list(indices)
    if n < 0:
        raise ValueError("n must be non-negative.")
    return list(indices[:n])


def indices_hash(indices: list[int] | None) -> str | None:
    if indices is None:
        return None
    joined = ",".join(str(idx) for idx in indices)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
