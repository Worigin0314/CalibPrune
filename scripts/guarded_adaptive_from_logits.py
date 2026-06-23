"""Guarded adaptive calibration from saved logits.

This selector fits temperature scaling, base CalibPrune, and AdaptiveCalibPrune
on a train split of the calibration logits, chooses the candidate with the best
held-out calibration score, then evaluates the selected transform on test logits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.calibrators.calibprune import AdaptiveCalibPrune, CalibPrune
from calibprune.calibrators.temperature import TemperatureScaling
from calibprune.metrics.calibration import expected_calibration_error, metrics_from_logits, negative_log_likelihood
from calibprune.metrics.selective import aurc, selective_accuracy_at_coverages
from scripts.calibrate_from_logits import load_npz, parse_retention_path, scalar_text, subset_name, write_result


def split_indices(n: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")
    if n < 4:
        raise ValueError("Guarded calibration requires at least four calibration samples.")
    val_count = max(1, min(n - 1, int(round(n * validation_fraction))))
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(n, dtype=np.int64))
    return np.sort(order[val_count:]), np.sort(order[:val_count])


def slice_payload(payload: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    out = dict(payload)
    out["logits"] = np.asarray(payload["logits"], dtype=np.float64)[indices]
    out["labels"] = np.asarray(payload["labels"], dtype=np.int64)[indices]
    return out


def score_logits(logits: np.ndarray, labels: np.ndarray, objective: str) -> float:
    if objective == "nll":
        return negative_log_likelihood(logits, labels, probabilities=False)
    if objective == "ece":
        return expected_calibration_error(logits, labels, probabilities=False)
    raise ValueError(f"Unknown objective: {objective}")


def evaluate_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    metrics = metrics_from_logits(logits, labels)
    metrics["aurc"] = aurc(logits, labels)
    metrics["aurc_margin"] = aurc(logits, labels, score="margin")
    metrics["aurc_entropy"] = aurc(logits, labels, score="entropy")
    metrics["selective_acc"] = selective_accuracy_at_coverages(logits, labels)
    metrics["selective_acc_margin"] = selective_accuracy_at_coverages(logits, labels, score="margin")
    metrics["selective_acc_entropy"] = selective_accuracy_at_coverages(logits, labels, score="entropy")
    return metrics


def select_guarded_candidate(candidates: list[dict[str, object]], selection_margin: float) -> dict[str, object]:
    """Select the best candidate, conservatively falling back to TS for tiny validation gaps."""

    if selection_margin < 0.0:
        raise ValueError("selection_margin must be non-negative.")
    best = min(candidates, key=lambda item: float(item["score"]))
    ts = next((item for item in candidates if item["name"] == "temperature_scaling"), None)
    if ts is None or best["name"] == "temperature_scaling":
        return best
    improvement = float(ts["score"]) - float(best["score"])
    if improvement < selection_margin:
        return ts
    return best


def fit_temperature(train_payload: dict[str, object]) -> TemperatureScaling:
    return TemperatureScaling().fit(
        np.asarray(train_payload["logits"], dtype=np.float64),
        np.asarray(train_payload["labels"], dtype=np.int64),
    )


def fit_calibprune(train_payloads: list[tuple[float, dict[str, object]]], objective: str) -> CalibPrune:
    labels = np.asarray(train_payloads[0][1]["labels"], dtype=np.int64)
    logits_at_ratios = {ret: np.asarray(payload["logits"], dtype=np.float64) for ret, payload in train_payloads}
    return CalibPrune(objective=objective).fit(logits_at_ratios, labels)


def fit_adaptive(
    train_payloads: list[tuple[float, dict[str, object]]],
    *,
    objective: str,
    adaptive_feature: str,
    adaptive_temperature_mode: str,
    adaptive_gamma_l2: float,
    adaptive_selective_weight: float,
    adaptive_selective_score: str,
) -> AdaptiveCalibPrune:
    labels = np.asarray(train_payloads[0][1]["labels"], dtype=np.int64)
    logits_at_ratios = {ret: np.asarray(payload["logits"], dtype=np.float64) for ret, payload in train_payloads}
    return AdaptiveCalibPrune(
        objective=objective,
        feature=adaptive_feature,
        temperature_mode=adaptive_temperature_mode,
        gamma_l2=adaptive_gamma_l2,
        selective_weight=adaptive_selective_weight,
        selective_score=adaptive_selective_score,
    ).fit(logits_at_ratios, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", action="append", type=parse_retention_path, required=True)
    parser.add_argument("--test", type=parse_retention_path, required=True)
    parser.add_argument("--objective", choices=["ece", "nll"], default="ece")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--validation-seed", type=int, default=20260620)
    parser.add_argument("--adaptive-feature", choices=["confidence", "margin", "entropy"], default="margin")
    parser.add_argument("--adaptive-temperature-mode", choices=["linear", "log"], default="log")
    parser.add_argument("--adaptive-gamma-l2", type=float, default=0.05)
    parser.add_argument("--adaptive-selective-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-selective-score", choices=["max_softmax", "margin", "entropy"], default="max_softmax")
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--refit-selected", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cal_payloads = [(ret, load_npz(path)) for ret, path in args.cal]
    test_retention, test_path = args.test
    test_payload = load_npz(test_path)
    labels = np.asarray(cal_payloads[0][1]["labels"], dtype=np.int64)
    for _, payload in cal_payloads[1:]:
        if not np.array_equal(labels, np.asarray(payload["labels"], dtype=np.int64)):
            raise RuntimeError("All calibration logits must share labels and ordering.")
    train_idx, val_idx = split_indices(labels.shape[0], args.validation_fraction, args.validation_seed)
    train_payloads = [(ret, slice_payload(payload, train_idx)) for ret, payload in cal_payloads]
    val_payloads = [(ret, slice_payload(payload, val_idx)) for ret, payload in cal_payloads]
    val_by_retention = {ret: payload for ret, payload in val_payloads}
    if test_retention not in val_by_retention:
        raise RuntimeError("Guarded selection currently requires a calibration retention matching the test retention.")
    val_payload = val_by_retention[test_retention]
    val_labels = np.asarray(val_payload["labels"], dtype=np.int64)

    candidates: list[dict[str, object]] = []
    matching_train_payload = dict(train_payloads)[test_retention]
    ts = fit_temperature(matching_train_payload)
    ts_val = ts.transform(np.asarray(val_payload["logits"], dtype=np.float64))
    candidates.append({"name": "temperature_scaling", "model": ts, "score": score_logits(ts_val, val_labels, args.objective)})

    cp = fit_calibprune(train_payloads, args.objective)
    cp_val = cp.transform(np.asarray(val_payload["logits"], dtype=np.float64), retention=test_retention)
    candidates.append({"name": "calibprune", "model": cp, "score": score_logits(cp_val, val_labels, args.objective)})

    adaptive = fit_adaptive(
        train_payloads,
        objective=args.objective,
        adaptive_feature=args.adaptive_feature,
        adaptive_temperature_mode=args.adaptive_temperature_mode,
        adaptive_gamma_l2=args.adaptive_gamma_l2,
        adaptive_selective_weight=args.adaptive_selective_weight,
        adaptive_selective_score=args.adaptive_selective_score,
    )
    adaptive_val = adaptive.transform(np.asarray(val_payload["logits"], dtype=np.float64), retention=test_retention)
    candidates.append({"name": "adaptive_calibprune", "model": adaptive, "score": score_logits(adaptive_val, val_labels, args.objective)})

    selected = select_guarded_candidate(candidates, args.selection_margin)
    final_model = selected["model"]
    if args.refit_selected:
        if selected["name"] == "temperature_scaling":
            final_model = fit_temperature(dict(cal_payloads)[test_retention])
        elif selected["name"] == "calibprune":
            final_model = fit_calibprune(cal_payloads, args.objective)
        elif selected["name"] == "adaptive_calibprune":
            final_model = fit_adaptive(
                cal_payloads,
                objective=args.objective,
                adaptive_feature=args.adaptive_feature,
                adaptive_temperature_mode=args.adaptive_temperature_mode,
                adaptive_gamma_l2=args.adaptive_gamma_l2,
                adaptive_selective_weight=args.adaptive_selective_weight,
                adaptive_selective_score=args.adaptive_selective_score,
            )
        else:
            raise RuntimeError(f"Unknown selected calibrator: {selected['name']}")

    test_logits = np.asarray(test_payload["logits"], dtype=np.float64)
    if selected["name"] == "temperature_scaling":
        transformed = final_model.transform(test_logits)
        selected_state = {"name": "temperature_scaling", "temperature": final_model.temperature_}
    else:
        transformed = final_model.transform(test_logits, retention=test_retention)
        selected_state = final_model.state_dict()

    test_labels = np.asarray(test_payload["labels"], dtype=np.int64)
    out_path = Path(args.output)
    result: dict[str, object] = {
        "run_id": out_path.stem,
        "model": scalar_text(test_payload.get("model")),
        "dataset": scalar_text(test_payload.get("dataset")),
        "subset_name": subset_name(test_payload),
        "pruner": scalar_text(test_payload.get("pruner")),
        "retention": test_retention,
        "calibrator": "guarded_adaptive_calibprune",
        "objective": args.objective,
        "paper_eligible": False,
        "selected_calibrator": selected["name"],
        "selection_scores": {item["name"]: float(item["score"]) for item in candidates},
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "selection_margin": args.selection_margin,
        "refit_selected": bool(args.refit_selected),
        "validation_count": int(val_idx.shape[0]),
        "train_count": int(train_idx.shape[0]),
        "state": {"name": "guarded_adaptive_calibprune", "selected": selected_state},
        "calibration_inputs": [
            {"retention": ret, "path": str(payload["path"]), "subset_name": subset_name(payload)}
            for ret, payload in cal_payloads
        ],
        "test_input": {"retention": test_retention, "path": str(test_payload["path"]), "subset_name": subset_name(test_payload)},
        "n_cal": int(labels.shape[0]),
        "n_test": int(test_labels.shape[0]),
        "n_samples": int(test_labels.shape[0]),
        "seed": int(np.asarray(test_payload.get("seed", -1)).item()),
        "sample_indices_hash": scalar_text(test_payload.get("sample_indices_hash")),
        **evaluate_metrics(transformed, test_labels),
    }
    write_result(out_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

