import numpy as np

from scripts.generate_logit_diagnostics import entropy, metric_value, pooled_alpha, residual_relative_norm


def test_pooled_alpha_recovers_scalar_scale():
    unpruned = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float)
    pruned = 1.5 * unpruned

    assert np.isclose(pooled_alpha(pruned, unpruned), 1.5)
    assert residual_relative_norm(pruned, unpruned) < 1e-12
    assert np.isclose(metric_value(pruned, unpruned, "alpha_pooled_minus_1"), 0.5)


def test_entropy_shift_is_negative_when_logits_sharpen():
    unpruned = np.array([[1.0, 0.0], [0.2, 0.0]], dtype=float)
    pruned = 2.0 * unpruned

    assert np.mean(entropy(pruned) - entropy(unpruned)) < 0.0
    assert metric_value(pruned, unpruned, "max_softmax_mean_shift") > 0.0
    assert metric_value(pruned, unpruned, "logit_l2_ratio_mean_minus_1") > 0.0