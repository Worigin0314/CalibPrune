import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.analyze_results import add_baseline_deltas, add_raw_calibrator_deltas, filter_result_rows, load_rows
from scripts.evaluate_paper_gate import evaluate


def write_result(path, *, pruner, retention, calibrator, ece, paper_eligible=False, pruner_evidence_type=None):
    payload = {
        "run_id": path.stem,
        "model": "toy_vlm",
        "dataset": "pope",
        "subset_name": "test",
        "pruner": pruner,
        "retention": retention,
        "calibrator": calibrator,
        "seed": 123,
        "n_samples": 4,
        "sample_indices_hash": "abc",
        "paper_eligible": paper_eligible,
        "accuracy": 0.5,
        "ece": ece,
        "adaptive_ece": ece,
        "aurc": 0.1,
        "max_softmax_mean": 0.7,
    }
    if pruner_evidence_type is not None:
        payload["pruner_evidence_type"] = pruner_evidence_type
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deltas_use_matching_calibrator_baseline(tmp_path):
    write_result(tmp_path / "none_raw.json", pruner="none", retention=1.0, calibrator="none", ece=0.10)
    write_result(tmp_path / "fastv_raw.json", pruner="fastv", retention=0.5, calibrator="none", ece=0.13)
    write_result(
        tmp_path / "none_temp.json",
        pruner="none",
        retention=1.0,
        calibrator="temperature_scaling",
        ece=0.04,
    )
    write_result(
        tmp_path / "fastv_temp.json",
        pruner="fastv",
        retention=0.5,
        calibrator="temperature_scaling",
        ece=0.07,
    )

    df = add_baseline_deltas(load_rows([str(tmp_path / "*.json")]))
    raw = df[(df["pruner"] == "fastv") & (df["calibrator"] == "none")].iloc[0]
    temp = df[(df["pruner"] == "fastv") & (df["calibrator"] == "temperature_scaling")].iloc[0]

    assert raw["delta_ece"] == pytest.approx(0.03)
    assert temp["delta_ece"] == pytest.approx(0.03)


def test_deltas_vs_raw_use_matching_pruner_retention(tmp_path):
    write_result(tmp_path / "fastv_raw_r05.json", pruner="fastv", retention=0.5, calibrator="none", ece=0.13)
    write_result(
        tmp_path / "fastv_calib_r05.json",
        pruner="fastv",
        retention=0.5,
        calibrator="calibprune",
        ece=0.08,
    )
    write_result(tmp_path / "fastv_raw_r025.json", pruner="fastv", retention=0.25, calibrator="none", ece=0.20)
    write_result(
        tmp_path / "fastv_calib_r025.json",
        pruner="fastv",
        retention=0.25,
        calibrator="calibprune",
        ece=0.11,
    )

    df = add_raw_calibrator_deltas(load_rows([str(tmp_path / "*.json")]))
    r05 = df[(df["retention"] == 0.5) & (df["calibrator"] == "calibprune")].iloc[0]
    r025 = df[(df["retention"] == 0.25) & (df["calibrator"] == "calibprune")].iloc[0]

    assert r05["delta_ece_vs_raw"] == pytest.approx(-0.05)
    assert r025["delta_ece_vs_raw"] == pytest.approx(-0.09)


def test_aggregate_results_pruner_filter_keeps_named_pruners(tmp_path):
    write_result(tmp_path / "none_raw.json", pruner="none", retention=1.0, calibrator="none", ece=0.10)
    write_result(tmp_path / "fastv_raw.json", pruner="fastv", retention=0.5, calibrator="none", ece=0.13)
    write_result(tmp_path / "random_raw.json", pruner="random", retention=0.5, calibrator="none", ece=0.14)
    out = tmp_path / "aggregate.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--input-glob",
            str(tmp_path / "*.json"),
            "--pruners",
            "none,fastv",
            "--output-csv",
            str(out),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    df = pd.read_csv(out)
    assert set(df["pruner"]) == {"none", "fastv"}


def test_load_rows_marks_sanity_only_pruners(tmp_path):
    write_result(tmp_path / "feature_norm_raw.json", pruner="feature_norm", retention=0.5, calibrator="none", ece=0.08)
    write_result(tmp_path / "fastv_raw.json", pruner="fastv", retention=0.5, calibrator="none", ece=0.13)
    write_result(tmp_path / "sparsevlm_raw.json", pruner="sparsevlm", retention=0.5, calibrator="none", ece=0.12)

    df = load_rows([str(tmp_path / "*.json")])
    feature_norm = df[df["pruner"] == "feature_norm"].iloc[0]
    fastv = df[df["pruner"] == "fastv"].iloc[0]
    sparsevlm = df[df["pruner"] == "sparsevlm"].iloc[0]

    assert feature_norm["pruner_evidence_type"] == "sanity-only"
    assert bool(feature_norm["paper_eligible_summary"]) is False
    assert fastv["pruner_evidence_type"] == "literature-hook"
    assert sparsevlm["pruner_evidence_type"] == "literature-proxy-hook"
    assert bool(sparsevlm["paper_eligible_summary"]) is False

def test_midlayer_evidence_override_is_allowed_when_json_marks_it(tmp_path):
    write_result(
        tmp_path / "sparsevlm_midlayer.json",
        pruner="sparsevlm",
        retention=0.5,
        calibrator="none",
        ece=0.12,
        paper_eligible=True,
        pruner_evidence_type="literature-midlayer-hook",
    )

    df = load_rows([str(tmp_path / "*.json")])
    row = df.iloc[0]

    assert row["pruner_evidence_type"] == "literature-midlayer-hook"
    assert bool(row["paper_eligible_summary"]) is True


def test_paper_gate_accepts_midlayer_hook_evidence():
    df = pd.DataFrame(
        [
            {
                "model": "toy_vlm",
                "dataset": "pope",
                "subset_name": "test",
                "pruner": "sparsevlm",
                "pruner_evidence_type": "literature-midlayer-hook",
                "retention": 0.5,
                "calibrator": "none",
                "n_samples": 512,
                "n_seeds": 4,
                "accuracy_mean": 0.5,
                "ece_mean": 0.1,
                "aurc_mean": 0.2,
            }
        ]
    )

    out = evaluate(
        df,
        min_pooled_samples=2000,
        min_seeds=3,
        allowed_evidence={"unpruned", "literature-hook", "literature-midlayer-hook"},
        require_json_flag=False,
    )

    assert bool(out.iloc[0]["paper_gate_pass"]) is True


def test_filter_result_rows_can_exclude_sanity_pruners(tmp_path):
    write_result(tmp_path / "none_raw.json", pruner="none", retention=1.0, calibrator="none", ece=0.10)
    write_result(tmp_path / "feature_norm_raw.json", pruner="feature_norm", retention=0.5, calibrator="none", ece=0.08)

    df = filter_result_rows(load_rows([str(tmp_path / "*.json")]), exclude_sanity=True)

    assert set(df["pruner"]) == {"none"}
