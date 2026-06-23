import numpy as np
import pytest

from scripts.generate_gate_figures import ConditionData
from scripts.generate_gate_statistics import (
    accuracy_metric,
    compute_rows,
    make_comparisons,
    paired_bootstrap_ci,
    paired_metric_delta,
    transform_with_result_state,
)


def condition(label, logits, labels):
    return ConditionData(label=label, logits=np.asarray(logits, dtype=float), labels=np.asarray(labels, dtype=np.int64))


def test_paired_metric_delta_requires_aligned_labels():
    left = condition("left", [[2.0, 0.0]], [0])
    right = condition("right", [[2.0, 0.0]], [1])

    with pytest.raises(ValueError, match="identical labels"):
        paired_metric_delta(left, right, accuracy_metric)


def test_paired_bootstrap_ci_contains_point_estimate_for_stable_delta():
    left = condition("left", [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]], [0, 1, 0])
    right = condition("right", [[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]], [0, 1, 0])
    estimate = paired_metric_delta(left, right, accuracy_metric)

    low, high = paired_bootstrap_ci(left, right, accuracy_metric, n_resamples=50, confidence=0.95, seed=1)

    assert low <= estimate <= high
    assert estimate == pytest.approx(1.0)


def test_make_comparisons_includes_drift_and_calibration_pairs():
    conditions = [
        condition("Unpruned raw", [[2.0, 0.0]], [0]),
        condition("FastV r=0.5 raw", [[2.0, 0.0]], [0]),
        condition("FastV r=0.5 + CalibPrune", [[2.0, 0.0]], [0]),
    ]

    names = {comparison.name for comparison in make_comparisons(conditions)}

    assert names == {
        "FastV raw - unpruned raw",
        "CalibPrune - FastV raw",
        "CalibPrune - unpruned raw",
    }


def test_transform_with_result_state_applies_adaptive_log_margin():
    logits = np.array([[4.0, 0.0], [0.2, 0.0]])
    payload = {
        "calibrator": "adaptive_calibprune",
        "state": {
            "t0_star": 1.0,
            "beta_star": 0.0,
            "gamma_star": 0.5,
            "feature": "margin",
            "temperature_mode": "log",
            "feature_center": 0.0,
            "feature_scale": 1.0,
        },
    }

    transformed = transform_with_result_state(logits, 0.5, payload)

    assert transformed.shape == logits.shape
    assert transformed[0, 0] < logits[0, 0]


def test_compute_rows_adds_wilcoxon_and_holm_fields():
    left_a = condition("left", [[2.0, 0.0], [0.0, 2.0]], [0, 1])
    right_a = condition("right", [[1.0, 0.0], [0.0, 1.0]], [0, 1])
    left_b = condition("left", [[2.0, 0.0], [0.0, 2.0]], [0, 1])
    right_b = condition("right", [[1.0, 0.0], [0.0, 1.0]], [0, 1])
    comparison = make_comparisons([
        condition("Unpruned raw", [[1.0, 0.0]], [0]),
        condition("FastV r=0.5 raw", [[1.0, 0.0]], [0]),
        condition("FastV r=0.5 + CalibPrune", [[1.0, 0.0]], [0]),
    ])[0]
    comparison = comparison.__class__(
        "toy comparison",
        condition("pooled left", [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]], [0, 1, 0, 1]),
        condition("pooled right", [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], [0, 1, 0, 1]),
        "negative_ece_means_improvement",
        ((left_a, right_a), (left_b, right_b)),
    )

    rows = compute_rows([comparison], n_resamples=20, confidence=0.8, seed=1)

    assert "wilcoxon_p" in rows[0]
    assert "wilcoxon_holm_p" in rows[0]

