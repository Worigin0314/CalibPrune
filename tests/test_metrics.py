import json

import numpy as np

from calibprune.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    metrics_from_logits,
    negative_log_likelihood,
    softmax,
)
from calibprune.metrics.selective import aurc, selective_accuracy_at_coverages, selective_confidence
from calibprune.models.vlm_wrapper import answer_logits_to_fixed_array, merge_answer_spaces


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0], [0.5, -0.5]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_metrics_are_finite():
    logits = np.array([[3.0, 0.1], [0.1, 2.5], [1.2, 1.0]])
    labels = np.array([0, 1, 0])
    assert expected_calibration_error(logits, labels) >= 0.0
    assert brier_score(logits, labels) >= 0.0
    assert negative_log_likelihood(logits, labels) >= 0.0
    assert aurc(logits, labels) >= 0.0
    assert selective_accuracy_at_coverages(logits, labels)


def test_selective_confidence_supports_margin_and_entropy():
    logits = np.array([[3.0, 0.0, -1.0], [1.0, 0.9, 0.8], [0.0, 0.0, 0.0]])

    max_scores = selective_confidence(logits, score="max_softmax")
    margin_scores = selective_confidence(logits, score="margin")
    entropy_scores = selective_confidence(logits, score="entropy")

    assert max_scores.shape == margin_scores.shape == entropy_scores.shape == (3,)
    assert margin_scores[0] > margin_scores[1]
    assert entropy_scores[0] > entropy_scores[2]


def test_aurc_accepts_alternative_selective_scores():
    logits = np.array([[3.0, 0.0, -1.0], [1.0, 0.9, 0.8], [0.0, 0.0, 0.0]])
    labels = np.array([0, 1, 2])

    assert aurc(logits, labels, score="margin") >= 0.0
    assert aurc(logits, labels, score="entropy") >= 0.0
    assert selective_accuracy_at_coverages(logits, labels, score="margin")


def test_schema_round_trip():
    logits = np.array([[3.0, 0.1], [0.1, 2.5], [1.2, 1.0]])
    labels = np.array([0, 1, 0])
    payload = {
        "run_id": "fixture",
        "model": "toy_vlm",
        "dataset": "pope",
        "pruner": "none",
        "retention": 1.0,
        "calibrator": "none",
        "n_samples": 3,
        **metrics_from_logits(logits, labels),
    }
    restored = json.loads(json.dumps(payload))
    assert restored["run_id"] == "fixture"
    assert "ece" in restored
    assert "reliability" in restored


def test_fixed_answer_space_pads_missing_choices():
    answer_space = merge_answer_spaces([("A", "B"), ("A", "B", "C")])
    row = answer_logits_to_fixed_array({"A": 1.0, "B": 2.0}, ("A", "B"), answer_space)
    assert answer_space == ("A", "B", "C")
    assert row.shape == (3,)
    assert row[0] == 1.0
    assert row[1] == 2.0
    assert row[2] < -1.0e8
