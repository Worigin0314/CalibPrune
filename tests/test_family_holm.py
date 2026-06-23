import pandas as pd

from scripts.apply_family_holm import apply_family_holm, parse_pruner_retention


def test_parse_pruner_retention_from_comparison_label():
    pruner, retention = parse_pruner_retention("AdaptiveCalibPrune - temperature scaling (fastv r=0.5)")

    assert pruner == "fastv"
    assert retention == 0.5


def test_apply_family_holm_adjusts_across_all_rows():
    df = pd.DataFrame({"wilcoxon_p": [0.001, 0.02, 0.2]})

    out = apply_family_holm(df, alpha=0.05)

    assert out["family_holm_reject"].tolist() == [True, True, False]
    assert out.loc[0, "family_holm_p"] <= out.loc[1, "family_holm_p"]
