import numpy as np
import pytest

from calibprune.calibrators.calibprune import AdaptiveCalibPrune, CalibPrune
from calibprune.calibrators.temperature import TemperatureScaling
from calibprune.metrics.calibration import negative_log_likelihood, softmax


def test_temperature_scaling_does_not_increase_nll_on_fit_grid():
    logits = np.array([[2.0, 0.0], [0.0, 1.8], [1.0, 0.2], [0.4, 1.2]])
    labels = np.array([0, 1, 0, 1])
    before = negative_log_likelihood(logits, labels)
    calibrator = TemperatureScaling(grid=np.arange(0.5, 2.1, 0.1)).fit(logits, labels)
    after = negative_log_likelihood(calibrator.transform(logits), labels)
    assert after <= before + 1e-9


def test_calibprune_recovers_temperature_on_synth():
    rng = np.random.default_rng(20260616)
    base = rng.normal(size=(3000, 3))
    probs = softmax(base)
    labels = np.asarray([rng.choice(3, p=row) for row in probs])
    t0 = 1.1
    beta = 0.8
    ratios = [0.25, 0.5, 0.75]
    logits_at_ratios = {r: base * (t0 + beta * (1.0 - r)) for r in ratios}
    model = CalibPrune(
        t0_grid=np.arange(0.8, 1.41, 0.1),
        beta_grid=np.arange(0.0, 1.21, 0.1),
        objective="nll",
    ).fit(logits_at_ratios, labels)
    assert abs(model.t0_star - t0) <= 0.3
    assert abs(model.beta_star - beta) <= 0.3


def test_calibprune_requires_multiple_retentions():
    logits = np.array([[2.0, 0.0], [0.0, 1.8], [1.0, 0.2], [0.4, 1.2]])
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="at least two retention"):
        CalibPrune().fit({0.5: logits}, labels)


def test_adaptive_calibprune_uses_samplewise_confidence_temperature():
    logits = np.array([[5.0, 0.0], [0.7, 0.0]])
    model = AdaptiveCalibPrune(feature="confidence", temperature_mode="linear", gamma_l2=0.0)
    model.t0_star = 1.0
    model.beta_star = 0.0
    model.gamma_star = 0.5
    model.feature_center_ = 0.5
    model.feature_scale_ = 1.0

    temps = model.sample_temperatures(logits, retention=0.5)
    transformed = model.transform(logits, retention=0.5)

    assert temps[0] > temps[1]
    assert np.allclose(transformed, logits / temps[:, None])


def test_adaptive_calibprune_state_records_feature_parameters():
    rng = np.random.default_rng(20260619)
    logits = rng.normal(size=(12, 3))
    labels = rng.integers(0, 3, size=12)
    model = AdaptiveCalibPrune(
        t0_grid=np.array([1.0]),
        beta_grid=np.array([0.0, 0.2]),
        gamma_grid=np.array([0.0, 0.1]),
        feature="confidence",
        temperature_mode="linear",
        gamma_l2=0.0,
        objective="nll",
    ).fit({0.5: logits, 0.75: logits * 1.1}, labels)

    state = model.state_dict()

    assert state["name"] == "adaptive_calibprune"
    assert state["feature"] == "confidence"
    assert state["feature_scale"] > 0
    assert "gamma_star" in state


def test_adaptive_calibprune_log_temperature_stays_positive():
    logits = np.array([[8.0, 0.0], [0.1, 0.0]])
    model = AdaptiveCalibPrune(feature="margin", temperature_mode="log")
    model.t0_star = 0.2
    model.beta_star = 0.0
    model.gamma_star = -0.5
    model.feature_center_ = 0.0
    model.feature_scale_ = 1.0

    temps = model.sample_temperatures(logits, retention=0.5)

    assert np.all(temps > 0)
    assert temps[0] < temps[1]


def test_adaptive_calibprune_entropy_feature_prefers_sharp_samples():
    sharp = np.array([[4.0, 0.0]])
    uncertain = np.array([[0.1, 0.0]])
    model = AdaptiveCalibPrune(feature="entropy")

    features = model._raw_feature(np.vstack([sharp, uncertain]))

    assert features[0] > features[1]


def test_adaptive_calibprune_selective_weight_is_recorded():
    rng = np.random.default_rng(20260620)
    logits = rng.normal(size=(16, 3))
    labels = rng.integers(0, 3, size=16)
    model = AdaptiveCalibPrune(
        t0_grid=np.array([1.0]),
        beta_grid=np.array([0.0]),
        gamma_grid=np.array([0.0]),
        selective_weight=0.2,
        selective_score="margin",
    ).fit({0.5: logits, 0.75: logits * 1.05}, labels)

    state = model.state_dict()

    assert state["selective_weight"] == 0.2
    assert state["selective_score"] == "margin"


def test_adaptive_calibprune_validation_split_is_recorded():
    rng = np.random.default_rng(20260621)
    logits = rng.normal(size=(20, 3))
    labels = rng.integers(0, 3, size=20)
    model = AdaptiveCalibPrune(
        t0_grid=np.array([1.0]),
        beta_grid=np.array([0.0]),
        gamma_grid=np.array([0.0]),
        validation_fraction=0.25,
        validation_seed=7,
    ).fit({0.5: logits, 0.75: logits * 1.05}, labels)

    state = model.state_dict()

    assert state["validation_fraction"] == 0.25
    assert state["validation_seed"] == 7
    assert state["validation_count"] == 5
    assert state["selection_count"] == 5
