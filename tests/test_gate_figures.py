import json

import numpy as np
import pytest

from scripts.generate_gate_figures import apply_calibprune_state, find_calibprune_states


def test_apply_calibprune_state_scales_by_retention():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    state = {"t0_star": 1.0, "beta_star": 2.0}

    transformed = apply_calibprune_state(logits, retention=0.5, state=state)

    assert np.allclose(transformed, logits / 2.0)


def test_apply_calibprune_state_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="positive"):
        apply_calibprune_state(np.array([[1.0, 0.0]]), retention=0.5, state={"t0_star": -1.0, "beta_star": 0.0})


def test_find_calibprune_states_filters_by_dataset_and_retention(tmp_path, monkeypatch):
    good = {
        "dataset": "pope",
        "calibrator": "calibprune",
        "pruner": "fastv",
        "retention": 0.5,
        "seed": 20260616,
        "state": {"t0_star": 1.0, "beta_star": 0.5},
    }
    wrong_retention = {**good, "retention": 0.25, "seed": 20260617}
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "wrong.json").write_text(json.dumps(wrong_retention), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.generate_gate_figures.PROJECT_ROOT", tmp_path)

    states = find_calibprune_states(["*.json"], dataset="pope", retention=0.5)

    assert states == {20260616: {"t0_star": 1.0, "beta_star": 0.5}}
