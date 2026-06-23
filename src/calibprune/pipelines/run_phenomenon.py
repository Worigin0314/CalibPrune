"""Runnable launcher for the current real-model phenomenon gate.

The original experiment plan listed a broad aspirational grid. The executable
project has converged on a smaller evidence gate: LLaVA, POPE/ScienceQA/MMBench-
CN, FastV plus VisionZip where wired, three retentions, and three deterministic
seeds. This launcher emits or dispatches those concrete commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEEDS = (20260616, 20260617, 20260618)
DEFAULT_RETENTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class GateSpec:
    dataset: str
    n_cal: int
    n_test: int
    pruners: tuple[str, ...]


DEFAULT_GATES = (
    GateSpec("pope", 128, 512, ("fastv",)),
    GateSpec("scienceqa", 128, 256, ("fastv",)),
    GateSpec("mmbench_cn", 128, 256, ("fastv", "visionzip")),
)


@dataclass(frozen=True)
class LaunchCommand:
    dataset: str
    seed: int
    pruners: tuple[str, ...]
    command: tuple[str, ...]

    def powershell(self) -> str:
        return " ".join(quote_arg(part) for part in self.command)


def quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value) or any(char in value for char in '"`'):
        return '"' + value.replace('"', '`"') + '"'
    return value


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return items


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in parse_csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in parse_csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def selected_gates(datasets: set[str] | None) -> tuple[GateSpec, ...]:
    gates = DEFAULT_GATES if datasets is None else tuple(gate for gate in DEFAULT_GATES if gate.dataset in datasets)
    if not gates:
        raise ValueError("No runnable gates matched the requested datasets.")
    return gates


def output_dir(output_root: str, gate: GateSpec, seed: int, adaptive: bool) -> str:
    suffix = "_adaptive_log_margin" if adaptive else ""
    seed_part = "" if seed == DEFAULT_SEEDS[0] else f"_seed{seed}"
    return str(Path(output_root) / f"{gate.dataset}_lite_llava_{gate.n_cal}_{gate.n_test}{seed_part}{suffix}")


def build_command(
    *,
    python_exe: str,
    model: str,
    gate: GateSpec,
    seed: int,
    retentions: tuple[float, ...],
    output_root: str,
    include_adaptive: bool,
) -> LaunchCommand:
    retentions_text = ",".join(f"{value:g}" for value in retentions)
    pruners_text = ",".join(gate.pruners)
    command = [
        python_exe,
        "scripts/calibrate_lite_grid.py",
        "--model",
        model,
        "--dataset",
        gate.dataset,
        "--seed",
        str(seed),
        "--n-cal",
        str(gate.n_cal),
        "--n-test",
        str(gate.n_test),
        "--pruners",
        pruners_text,
        "--retentions",
        retentions_text,
        "--output-dir",
        output_dir(output_root, gate, seed, include_adaptive),
        "--output-prefix",
        f"llava_{gate.dataset}_{gate.n_cal}_{gate.n_test}",
    ]
    if include_adaptive:
        command.append("--include-adaptive-calibprune")
    return LaunchCommand(gate.dataset, seed, gate.pruners, tuple(command))


def iter_commands(
    *,
    python_exe: str,
    model: str,
    datasets: set[str] | None,
    seeds: tuple[int, ...],
    retentions: tuple[float, ...],
    output_root: str,
    include_adaptive: bool,
) -> Iterable[LaunchCommand]:
    for gate in selected_gates(datasets):
        for seed in seeds:
            yield build_command(
                python_exe=python_exe,
                model=model,
                gate=gate,
                seed=seed,
                retentions=retentions,
                output_root=output_root,
                include_adaptive=include_adaptive,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the current runnable phenomenon gate.")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--datasets", type=parse_csv, default=None, help="Comma-separated subset of pope,scienceqa,mmbench_cn.")
    parser.add_argument("--seeds", type=parse_int_csv, default=DEFAULT_SEEDS)
    parser.add_argument("--retentions", type=parse_float_csv, default=DEFAULT_RETENTIONS)
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--include-adaptive-calibprune", action="store_true")
    args = parser.parse_args()

    datasets = set(args.datasets) if args.datasets else None
    commands = list(
        iter_commands(
            python_exe=args.python_exe,
            model=args.model,
            datasets=datasets,
            seeds=args.seeds,
            retentions=args.retentions,
            output_root=args.output_root,
            include_adaptive=args.include_adaptive_calibprune,
        )
    )
    print(f"planned_commands={len(commands)}")
    for item in commands:
        print(item.powershell())
        if args.execute:
            subprocess.run(item.command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

