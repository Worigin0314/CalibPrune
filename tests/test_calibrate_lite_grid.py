import json
import subprocess
import sys

import numpy as np


def write_logits(path, *, subset, pruner, retention, n):
    logits = np.tile(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float), (max(1, n // 2), 1))[:n]
    labels = np.arange(n, dtype=np.int64) % 2
    np.savez_compressed(
        path,
        logits=logits,
        labels=labels,
        answer_choices=np.array(["no", "yes"], dtype=object),
        sample_indices=np.arange(n, dtype=np.int64),
        subset_name=np.asarray(subset, dtype=object),
        sample_indices_hash=np.asarray(f"{subset}-hash", dtype=object),
        model=np.asarray("toy_vlm", dtype=object),
        dataset=np.asarray("pope", dtype=object),
        pruner=np.asarray(pruner, dtype=object),
        retention=np.asarray(retention, dtype=np.float64),
        seed=np.asarray(123, dtype=np.int64),
    )


def seed_logits_tree(logits_dir):
    model = "toy_vlm"
    dataset = "pope"
    seed = 123
    for subset, n in [("cal", 4), ("test", 4)]:
        write_logits(
            logits_dir / f"{model}-{dataset}-{subset}-none-r1-n{n}-seed{seed}.npz",
            subset=subset,
            pruner="none",
            retention=1.0,
            n=n,
        )
        for retention in [0.25, 0.5]:
            write_logits(
                logits_dir / f"{model}-{dataset}-{subset}-fastv-r{retention:g}-n{n}-seed{seed}.npz",
                subset=subset,
                pruner="fastv",
                retention=retention,
                n=n,
            )


def test_calibrate_lite_grid_dry_run_counts_tasks(tmp_path):
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"dry_run": True, "task_count": 5}


def test_calibrate_lite_grid_dry_run_counts_adaptive_tasks(tmp_path):
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--include-adaptive-calibprune",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"dry_run": True, "task_count": 7}


def test_calibrate_lite_grid_writes_outputs(tmp_path):
    logits_dir = tmp_path / "logits"
    out_dir = tmp_path / "out"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(out_dir),
            "--output-prefix",
            "toy_gate",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    written = {path.name for path in out_dir.glob("*.json")}
    assert written == {
        "toy_gate_none_temp.json",
        "toy_gate_fastv_temp_r025.json",
        "toy_gate_fastv_temp_r05.json",
        "toy_gate_fastv_calibprune_r025.json",
        "toy_gate_fastv_calibprune_r05.json",
    }
    payload = json.loads((out_dir / "toy_gate_fastv_calibprune_r05.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "toy_gate_fastv_calibprune_r05"
    assert payload["calibrator"] == "calibprune"


def test_calibrate_lite_grid_writes_adaptive_outputs(tmp_path):
    logits_dir = tmp_path / "logits"
    out_dir = tmp_path / "out"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(out_dir),
            "--output-prefix",
            "toy_gate",
            "--include-adaptive-calibprune",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out_dir / "toy_gate_fastv_adaptive_calibprune_log_margin_g005_r05.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "toy_gate_fastv_adaptive_calibprune_log_margin_g005_r05"
    assert payload["calibrator"] == "adaptive_calibprune"
    assert payload["state"]["name"] == "adaptive_calibprune"
    assert "gamma_star" in payload["state"]


def test_calibrate_lite_grid_names_adaptive_feature_outputs(tmp_path):
    logits_dir = tmp_path / "logits"
    out_dir = tmp_path / "out"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(out_dir),
            "--output-prefix",
            "toy_gate",
            "--include-adaptive-calibprune",
            "--adaptive-feature",
            "margin",
            "--adaptive-temperature-mode",
            "log",
            "--adaptive-gamma-l2",
            "0.05",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = out_dir / "toy_gate_fastv_adaptive_calibprune_log_margin_g005_r05.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["adaptive_feature"] == "margin"
    assert payload["adaptive_temperature_mode"] == "log"
    assert payload["adaptive_gamma_l2"] == 0.05
    assert payload["state"]["temperature_mode"] == "log"


def test_calibrate_lite_grid_names_selective_adaptive_outputs(tmp_path):
    logits_dir = tmp_path / "logits"
    out_dir = tmp_path / "out"
    logits_dir.mkdir()
    seed_logits_tree(logits_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_lite_grid.py",
            "--model",
            "toy_vlm",
            "--dataset",
            "pope",
            "--seed",
            "123",
            "--n-cal",
            "4",
            "--n-test",
            "4",
            "--pruners",
            "fastv",
            "--retentions",
            "0.25,0.5",
            "--logits-dir",
            str(logits_dir),
            "--output-dir",
            str(out_dir),
            "--output-prefix",
            "toy_gate",
            "--include-adaptive-calibprune",
            "--adaptive-selective-weight",
            "0.1",
            "--adaptive-selective-score",
            "margin",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = out_dir / "toy_gate_fastv_adaptive_calibprune_log_margin_g005_sw01_margin_r05.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["adaptive_selective_weight"] == 0.1
    assert payload["adaptive_selective_score"] == "margin"
    assert payload["state"]["selective_weight"] == 0.1
    assert payload["state"]["selective_score"] == "margin"
