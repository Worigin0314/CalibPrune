"""Batch calibrate a saved lite-grid logits directory.

This script assumes logits saved by ``run_pope_lite_inprocess.py`` using the
stable naming convention:

``{model}-{dataset}-{subset}-{pruner}-r{retention}-n{n}-seed{seed}.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.calibrate_from_logits import fit_and_evaluate, load_npz, write_result


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_retention_csv(text: str) -> list[float]:
    values = [float(part) for part in parse_csv(text)]
    if not values:
        raise argparse.ArgumentTypeError("At least one retention is required.")
    return values


def retention_text(value: float) -> str:
    return f"{value:g}"


def retention_tag(value: float) -> str:
    return retention_text(value).replace(".", "")


def safe_tag(value: str) -> str:
    return value.replace("-", "_").replace(".", "")


def logits_path(
    logits_dir: Path,
    *,
    model: str,
    dataset: str,
    subset: str,
    pruner: str,
    retention: float,
    n_samples: int,
    seed: int,
) -> Path:
    return logits_dir / (
        f"{model}-{dataset}-{subset}-{pruner}-"
        f"r{retention_text(retention)}-n{n_samples}-seed{seed}.npz"
    )


def display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def calibrate_one(
    *,
    output_dir: Path,
    output_prefix: str,
    cal_payloads: list[tuple[float, dict[str, object]]],
    test_retention: float,
    test_payload: dict[str, object],
    pruner: str,
    calibrator: str,
    adaptive_feature: str = "margin",
    adaptive_temperature_mode: str = "log",
    adaptive_gamma_l2: float = 0.05,
    adaptive_selective_weight: float = 0.0,
    adaptive_selective_score: str = "max_softmax",
    adaptive_validation_fraction: float = 0.0,
    adaptive_validation_seed: int = 20260620,
) -> Path:
    if pruner == "none":
        stem = f"{output_prefix}_none_temp"
    elif calibrator == "adaptive_calibprune":
        selective_tag = (
            f"_sw{safe_tag(f'{adaptive_selective_weight:g}')}_{safe_tag(adaptive_selective_score)}"
            if adaptive_selective_weight > 0
            else ""
        )
        stem = (
            f"{output_prefix}_{pruner}_adaptive_calibprune_"
            f"{safe_tag(adaptive_temperature_mode)}_{safe_tag(adaptive_feature)}_"
            f"g{safe_tag(f'{adaptive_gamma_l2:g}')}{selective_tag}_r{retention_tag(test_retention)}"
        )
    else:
        stem = f"{output_prefix}_{pruner}_{'temp' if calibrator == 'temperature_scaling' else 'calibprune'}_r{retention_tag(test_retention)}"
    output_path = output_dir / f"{stem}.json"
    result = fit_and_evaluate(
        cal_payloads,
        test_retention,
        test_payload,
        calibrator_name=calibrator,
        run_id=output_path.stem,
        adaptive_feature=adaptive_feature,
        adaptive_temperature_mode=adaptive_temperature_mode,
        adaptive_gamma_l2=adaptive_gamma_l2,
        adaptive_selective_weight=adaptive_selective_weight,
        adaptive_selective_score=adaptive_selective_score,
        adaptive_validation_fraction=adaptive_validation_fraction,
        adaptive_validation_seed=adaptive_validation_seed,
    )
    write_result(output_path, result)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--n-cal", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--pruners", default="fastv,visionzip")
    parser.add_argument("--retentions", type=parse_retention_csv, default=parse_retention_csv("0.25,0.5,0.75"))
    parser.add_argument("--logits-dir", default="results/raw/logits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--include-adaptive-calibprune",
        action="store_true",
        help="Also fit the sample-confidence adaptive CalibPrune variant for each pruner/retention.",
    )
    parser.add_argument("--adaptive-feature", choices=["confidence", "margin", "entropy"], default="margin")
    parser.add_argument("--adaptive-temperature-mode", choices=["linear", "log"], default="log")
    parser.add_argument("--adaptive-gamma-l2", type=float, default=0.05)
    parser.add_argument("--adaptive-selective-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-selective-score", choices=["max_softmax", "margin", "entropy"], default="max_softmax")
    parser.add_argument("--adaptive-validation-fraction", type=float, default=0.0)
    parser.add_argument("--adaptive-validation-seed", type=int, default=20260620)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logits_dir = PROJECT_ROOT / args.logits_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_prefix = args.output_prefix or f"{args.model}_{args.dataset}_n{args.n_test}_seed{args.seed}"
    pruners = parse_csv(args.pruners)

    tasks: list[dict[str, object]] = []
    none_cal = logits_path(
        logits_dir,
        model=args.model,
        dataset=args.dataset,
        subset="cal",
        pruner="none",
        retention=1.0,
        n_samples=args.n_cal,
        seed=args.seed,
    )
    none_test = logits_path(
        logits_dir,
        model=args.model,
        dataset=args.dataset,
        subset="test",
        pruner="none",
        retention=1.0,
        n_samples=args.n_test,
        seed=args.seed,
    )
    tasks.append(
        {
            "pruner": "none",
            "calibrator": "temperature_scaling",
            "retention": 1.0,
            "cal": [none_cal],
            "test": none_test,
        }
    )

    for pruner in pruners:
        cal_paths = [
            logits_path(
                logits_dir,
                model=args.model,
                dataset=args.dataset,
                subset="cal",
                pruner=pruner,
                retention=retention,
                n_samples=args.n_cal,
                seed=args.seed,
            )
            for retention in args.retentions
        ]
        for retention in args.retentions:
            test_path = logits_path(
                logits_dir,
                model=args.model,
                dataset=args.dataset,
                subset="test",
                pruner=pruner,
                retention=retention,
                n_samples=args.n_test,
                seed=args.seed,
            )
            tasks.append(
                {
                    "pruner": pruner,
                    "calibrator": "temperature_scaling",
                    "retention": retention,
                    "cal": [cal_paths[args.retentions.index(retention)]],
                    "test": test_path,
                }
            )
            tasks.append(
                {
                    "pruner": pruner,
                    "calibrator": "calibprune",
                    "retention": retention,
                    "cal": cal_paths,
                    "test": test_path,
                }
            )
            if args.include_adaptive_calibprune:
                tasks.append(
                    {
                        "pruner": pruner,
                        "calibrator": "adaptive_calibprune",
                        "retention": retention,
                        "cal": cal_paths,
                        "test": test_path,
                    }
                )

    missing = sorted(
        {
            display_path(Path(path))
            for task in tasks
            for path in [*task["cal"], task["test"]]
            if not Path(path).exists()
        }
    )
    if args.dry_run:
        payload = {"dry_run": True, "task_count": len(tasks)}
        if missing:
            payload["missing"] = missing
        print(json.dumps(payload, indent=2))
        return

    if missing:
        raise FileNotFoundError("Missing logits files:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task in tasks:
        retention = float(task["retention"])
        cal_payloads = [(retention, load_npz(path)) for path in task["cal"]]
        if task["calibrator"] in {"calibprune", "adaptive_calibprune"}:
            cal_payloads = list(zip(args.retentions, [load_npz(path) for path in task["cal"]]))
        test_payload = load_npz(task["test"])
        written_path = calibrate_one(
            output_dir=output_dir,
            output_prefix=output_prefix,
            cal_payloads=cal_payloads,
            test_retention=retention,
            test_payload=test_payload,
            pruner=str(task["pruner"]),
            calibrator=str(task["calibrator"]),
            adaptive_feature=args.adaptive_feature,
            adaptive_temperature_mode=args.adaptive_temperature_mode,
            adaptive_gamma_l2=args.adaptive_gamma_l2,
            adaptive_selective_weight=args.adaptive_selective_weight,
            adaptive_selective_score=args.adaptive_selective_score,
            adaptive_validation_fraction=args.adaptive_validation_fraction,
            adaptive_validation_seed=args.adaptive_validation_seed,
        )
        written.append(display_path(written_path))

    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()




