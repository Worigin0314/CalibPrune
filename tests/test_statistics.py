import numpy as np
import pytest

from calibprune.metrics.statistics import bca_bootstrap_ci, holm_bonferroni, paired_delta, percentile_bootstrap_ci, wilcoxon_signed_rank_test


def test_percentile_bootstrap_ci_contains_sample_mean():
    values = np.array([0.0, 1.0, 2.0, 3.0])

    low, high = percentile_bootstrap_ci(values, n_resamples=200, seed=7)

    assert low <= np.mean(values) <= high


def test_bca_bootstrap_ci_contains_sample_mean_on_small_sample():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    low, high = bca_bootstrap_ci(values, n_resamples=200, seed=7)

    assert low <= np.mean(values) <= high


def test_bca_bootstrap_requires_at_least_two_values():
    with pytest.raises(ValueError, match="at least two"):
        bca_bootstrap_ci(np.array([1.0]))


def test_paired_delta_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        paired_delta(np.array([1.0, 2.0]), np.array([1.0]))


def test_holm_bonferroni_step_down():
    out = holm_bonferroni(np.array([0.001, 0.02, 0.08]), alpha=0.05)

    assert out["reject"].tolist() == [True, True, False]
    assert np.all(out["adjusted_p_values"] >= np.array([0.001, 0.02, 0.08]))


def test_wilcoxon_signed_rank_detects_positive_shift():
    out = wilcoxon_signed_rank_test(np.array([0.2, 0.3, 0.1]), alternative="greater")

    assert out["n"] == 3
    assert out["p_value"] <= 0.25


def test_wilcoxon_signed_rank_rejects_unknown_alternative():
    with pytest.raises(ValueError, match="alternative"):
        wilcoxon_signed_rank_test(np.array([0.1, -0.2]), alternative="sideways")
