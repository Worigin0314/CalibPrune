"""Fit post-hoc calibrators from saved answer-level logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.calibrators.calibprune import AdaptiveCalibPrune, CalibPrune
from calibprune.calibrators.temperature import TemperatureScaling
from calibprune.metrics.calibration import metrics_from_logits
from calibprune.metrics.selective import aurc, selective_accuracy_at_coverages


def parse_retention_path(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RETENTION=PATH.")
    retention, path = value.split("=", 1)
    return float(retention), Path(path)


def load_npz(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    payload["path"] = str(path)
    return payload


def subset_name(payload: dict[str, object]) -> str:
    return scalar_text(payload.get("subset_name"))


def scalar_text(value: object) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr.tolist())


def fit_and_evaluate(
    cal_payloads: list[tuple[float, dict[str, object]]],
    test_retention: float,
    test_payload: dict[str, object],
    *,
    calibrator_name: str,
    objective: str = "ece",
    run_id: str | None = None,
    adaptive_feature: str = "margin",
    adaptive_temperature_mode: str = "log",
    adaptive_gamma_l2: float = 0.05,
    adaptive_selective_weight: float = 0.0,
    adaptive_selective_score: str = "max_softmax",
    adaptive_validation_fraction: float = 0.0,
    adaptive_validation_seed: int = 20260620,
) -> dict[str, object]:
    labels = np.asarray(cal_payloads[0][1]["labels"], dtype=np.int64)
    for _, payload in cal_payloads[1:]:
        if not np.array_equal(labels, np.asarray(payload["labels"], dtype=np.int64)):
            raise RuntimeError("All calibration logits must share labels and ordering.")

    if calibrator_name == "temperature_scaling":
        cal_logits = np.vstack([np.asarray(payload["logits"], dtype=np.float64) for _, payload in cal_payloads])
        cal_labels = np.concatenate([np.asarray(payload["labels"], dtype=np.int64) for _, payload in cal_payloads])
        calibrator = TemperatureScaling().fit(cal_logits, cal_labels)
        transformed = calibrator.transform(np.asarray(test_payload["logits"], dtype=np.float64))
        state = {"name": calibrator.name, "temperature": calibrator.temperature_}
    elif calibrator_name in {"calibprune", "adaptive_calibprune"}:
        logits_at_ratios = {
            retention: np.asarray(payload["logits"], dtype=np.float64)
            for retention, payload in cal_payloads
        }
        if calibrator_name == "adaptive_calibprune":
            calibrator = AdaptiveCalibPrune(
                objective=objective,
                feature=adaptive_feature,
                temperature_mode=adaptive_temperature_mode,
                gamma_l2=adaptive_gamma_l2,
                selective_weight=adaptive_selective_weight,
                selective_score=adaptive_selective_score,
                validation_fraction=adaptive_validation_fraction,
                validation_seed=adaptive_validation_seed,
            ).fit(logits_at_ratios, labels)
        else:
            calibrator = CalibPrune(objective=objective).fit(logits_at_ratios, labels)
        transformed = calibrator.transform(np.asarray(test_payload["logits"], dtype=np.float64), retention=test_retention)
        state = calibrator.state_dict()
    else:
        raise ValueError(f"Unknown calibrator: {calibrator_name}")

    test_labels = np.asarray(test_payload["labels"], dtype=np.int64)
    metrics = metrics_from_logits(transformed, test_labels)
    metrics["aurc"] = aurc(transformed, test_labels)
    metrics["aurc_margin"] = aurc(transformed, test_labels, score="margin")
    metrics["aurc_entropy"] = aurc(transformed, test_labels, score="entropy")
    metrics["selective_acc"] = selective_accuracy_at_coverages(transformed, test_labels)
    metrics["selective_acc_margin"] = selective_accuracy_at_coverages(transformed, test_labels, score="margin")
    metrics["selective_acc_entropy"] = selective_accuracy_at_coverages(transformed, test_labels, score="entropy")
    result: dict[str, object] = {
        "run_id": run_id,
        "model": scalar_text(test_payload.get("model")),
        "dataset": scalar_text(test_payload.get("dataset")),
        "subset_name": subset_name(test_payload),
        "pruner": scalar_text(test_payload.get("pruner")),
        "retention": test_retention,
        "calibrator": calibrator_name,
        "objective": objective,
        "adaptive_feature": adaptive_feature if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_temperature_mode": adaptive_temperature_mode if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_gamma_l2": adaptive_gamma_l2 if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_selective_weight": adaptive_selective_weight if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_selective_score": adaptive_selective_score if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_validation_fraction": adaptive_validation_fraction if calibrator_name == "adaptive_calibprune" else None,
        "adaptive_validation_seed": adaptive_validation_seed if calibrator_name == "adaptive_calibprune" else None,
        "paper_eligible": False,
        "calibration_inputs": [
            {"retention": retention, "path": str(payload["path"]), "subset_name": subset_name(payload)}
            for retention, payload in cal_payloads
        ],
        "test_input": {
            "retention": test_retention,
            "path": str(test_payload["path"]),
            "subset_name": subset_name(test_payload),
        },
        "n_cal": int(labels.shape[0]),
        "n_test": int(test_labels.shape[0]),
        "n_samples": int(test_labels.shape[0]),
        "seed": int(np.asarray(test_payload.get("seed", -1)).item()),
        "sample_indices_hash": scalar_text(test_payload.get("sample_indices_hash")),
        "state": state,
        **metrics,
    }
    return result


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", action="append", type=parse_retention_path, required=True)
    parser.add_argument("--test", type=parse_retention_path, required=True)
    parser.add_argument("--calibrator", choices=["temperature_scaling", "calibprune", "adaptive_calibprune"], required=True)
    parser.add_argument("--objective", choices=["ece", "nll"], default="ece")
    parser.add_argument("--adaptive-feature", choices=["confidence", "margin", "entropy"], default="margin")
    parser.add_argument("--adaptive-temperature-mode", choices=["linear", "log"], default="log")
    parser.add_argument("--adaptive-gamma-l2", type=float, default=0.05)
    parser.add_argument("--adaptive-selective-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-selective-score", choices=["max_softmax", "margin", "entropy"], default="max_softmax")
    parser.add_argument("--adaptive-validation-fraction", type=float, default=0.0)
    parser.add_argument("--adaptive-validation-seed", type=int, default=20260620)
    parser.add_argument("--allow-noncanonical-subsets", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cal_payloads = [(ret, load_npz(path)) for ret, path in args.cal]
    test_retention, test_path = args.test
    test_payload = load_npz(test_path)
    if not args.allow_noncanonical_subsets:
        for _, payload in cal_payloads:
            if subset_name(payload) != "cal":
                raise RuntimeError(f"Calibration logits must have subset_name=cal: {payload['path']}")
        if subset_name(test_payload) != "test":
            raise RuntimeError(f"Test logits must have subset_name=test: {test_payload['path']}")

    out_path = Path(args.output)
    result = fit_and_evaluate(
        cal_payloads,
        test_retention,
        test_payload,
        calibrator_name=args.calibrator,
        objective=args.objective,
        run_id=out_path.stem,
        adaptive_feature=args.adaptive_feature,
        adaptive_temperature_mode=args.adaptive_temperature_mode,
        adaptive_gamma_l2=args.adaptive_gamma_l2,
        adaptive_selective_weight=args.adaptive_selective_weight,
        adaptive_selective_score=args.adaptive_selective_score,
        adaptive_validation_fraction=args.adaptive_validation_fraction,
        adaptive_validation_seed=args.adaptive_validation_seed,
    )
    write_result(out_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()




