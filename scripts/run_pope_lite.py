"""Run a resumable LLaVA/POPE-lite grid.

This launcher is intentionally small: it validates the real POPE pathway,
uses deterministic cal/test splits, writes stable file names, and skips
completed cells. It is a robustness bridge before the full Phase A grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.data.splits import default_split_path, ensure_split_file, indices_hash, limit_indices, load_split_indices


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def grid_id(pruners: list[str], retentions: list[float]) -> str:
    payload = json.dumps({"pruners": pruners, "retentions": retentions}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def run_command(cmd: list[str], env: dict[str, str], timeout_sec: int) -> dict[str, object]:
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava15_7b_4bit")
    parser.add_argument("--pruners", default="none,fastv")
    parser.add_argument("--retentions", default="0.5")
    parser.add_argument("--n-cal", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="results/raw/pope_lite")
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=5400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_n_cal = max(args.n_cal, 500)
    split_file = PROJECT_ROOT / args.split_file if args.split_file else PROJECT_ROOT / default_split_path("pope", split_n_cal, args.seed)
    ensure_split_file(split_file, dataset="pope", n_total=9000, n_cal=split_n_cal, seed=args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
    env.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / "hf_cache"))
    env.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache" / "transformers"))

    pruners = parse_csv(args.pruners)
    retentions = [float(value) for value in parse_csv(args.retentions)]
    split_indices = {
        "cal": load_split_indices(split_file, "cal")[0],
        "test": load_split_indices(split_file, "test")[0],
    }
    tasks: list[dict[str, object]] = []
    for subset_name, n_samples in [("cal", args.n_cal), ("test", args.n_test)]:
        expected_indices = limit_indices(split_indices[subset_name], n_samples)
        expected_hash = indices_hash(expected_indices)
        for pruner in pruners:
            cell_retentions = [1.0] if pruner == "none" else retentions
            for retention in cell_retentions:
                run_id = (
                    f"{args.model}-pope-{subset_name}-{pruner}-"
                    f"r{retention:g}-n{n_samples}-seed{args.seed}"
                )
                output = output_dir / f"{run_id}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "calibprune.pipelines.run_eval",
                    "--model",
                    args.model,
                    "--dataset",
                    "pope",
                    "--split",
                    "test",
                    "--n",
                    str(n_samples),
                    "--pruner",
                    pruner,
                    "--retention",
                    str(retention),
                    "--calibrator",
                    "none",
                    "--split-file",
                    str(split_file),
                    "--subset-name",
                    subset_name,
                    "--run-id",
                    run_id,
                    "--output",
                    str(output),
                    "--save-logits",
                    "--skip-existing",
                ]
                if args.offline_fixture:
                    cmd.append("--offline-fixture")
                tasks.append(
                    {
                        "run_id": run_id,
                        "output": str(output),
                        "cmd": cmd,
                        "expected_hash": expected_hash,
                        "expected_n": n_samples,
                    }
                )

    manifest = {
        "model": args.model,
        "dataset": "pope",
        "split_file": str(split_file),
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "seed": args.seed,
        "tasks": tasks,
    }
    manifest_path = output_dir / f"manifest-seed{args.seed}-cal{args.n_cal}-test{args.n_test}-grid{grid_id(pruners, retentions)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "task_count": len(tasks), "manifest": str(manifest_path)}, indent=2))
        return 0

    failures = []
    for task in tasks:
        output = Path(str(task["output"]))
        force_rerun = False
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                reusable = (
                    existing.get("sample_indices_hash") == task["expected_hash"]
                    and existing.get("n_samples") == task["expected_n"]
                )
            except Exception:
                reusable = False
            if reusable:
                print(json.dumps({"skipped": True, "run_id": task["run_id"], "output": str(output)}, indent=2))
                continue
            print(
                json.dumps(
                    {
                        "rerun": True,
                        "run_id": task["run_id"],
                        "reason": "existing output does not match requested split hash or n",
                        "output": str(output),
                    },
                    indent=2,
                )
            )
            force_rerun = True
        cmd = list(task["cmd"])
        if force_rerun and "--skip-existing" in cmd:
            cmd.remove("--skip-existing")
        result = run_command(cmd, env=env, timeout_sec=args.timeout_sec)
        if result["returncode"] != 0:
            failures.append({"run_id": task["run_id"], **result})
            print(json.dumps(failures[-1], indent=2))
        else:
            print(result["stdout"].strip())

    if failures:
        failure_path = output_dir / f"failures-seed{args.seed}-cal{args.n_cal}-test{args.n_test}.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
