import numpy as np
import pytest

from calibprune.pruners.base import RandomPruner, UniformPruner, build_pruner
from calibprune.pruners.fastv import select_fastv_indices
from calibprune.pruners.pyramiddrop import select_pyramiddrop_tokens
from calibprune.pruners.sparsevlm import select_sparsevlm_tokens
from calibprune.pruners.vtw import select_vtw_proxy_tokens
from calibprune.pruners.visionzip import select_visionzip_tokens, visionzip_budget


def test_token_count_invariant():
    tokens = np.arange(20 * 4).reshape(20, 4)
    for name in ["random", "uniform", "feature_norm"]:
        out = build_pruner(name)(tokens, attentions=None, retention_ratio=0.5)
        assert out.tokens.shape[0] == 10
        assert out.kept_indices.shape[0] == 10


def test_model_hook_and_literature_placeholders_are_not_silent_score_pruners():
    for name in ["fastv", "sparsevlm", "visionzip", "pyramiddrop", "vtw"]:
        with pytest.raises(NotImplementedError):
            build_pruner(name)


def test_none_keeps_all_tokens():
    tokens = np.arange(12).reshape(6, 2)
    out = build_pruner("none")(tokens, attentions=None, retention_ratio=1.0)
    assert out.tokens.shape[0] == 6
    assert np.array_equal(out.kept_indices, np.arange(6))


def test_random_is_deterministic():
    tokens = np.arange(20 * 4).reshape(20, 4)
    a = RandomPruner(seed=123)(tokens, None, 0.4).kept_indices
    b = RandomPruner(seed=123)(tokens, None, 0.4).kept_indices
    assert np.array_equal(a, b)


def test_uniform_is_ordered():
    tokens = np.arange(20 * 4).reshape(20, 4)
    out = UniformPruner()(tokens, None, 0.3)
    assert np.all(np.diff(out.kept_indices) > 0)


def test_fastv_selects_high_text_to_image_attention():
    torch = pytest.importorskip("torch")
    attention = torch.zeros((2, 8, 8), dtype=torch.float32)
    # Merged sequence: text positions [0, 1, 6, 7], image span [2, 3, 4, 5].
    attention[:, [0, 1, 6, 7], 2] = 0.1
    attention[:, [0, 1, 6, 7], 3] = 0.9
    attention[:, [0, 1, 6, 7], 4] = 0.4
    attention[:, [0, 1, 6, 7], 5] = 0.8

    kept = select_fastv_indices(
        attention,
        merged_seq_len=8,
        image_start=2,
        n_visual_tokens=4,
        retention_ratio=0.5,
    )

    assert kept.detach().cpu().numpy().tolist() == [1, 3]


def test_sparsevlm_text_guided_recycles_pruned_tokens():
    torch = pytest.importorskip("torch")
    image_features = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    attention = torch.zeros((2, 8, 8), dtype=torch.float32)
    # Merged sequence: text positions [0, 1, 6, 7], image span [2, 3, 4, 5].
    attention[:, [0, 1, 6, 7], 2] = 0.1
    attention[:, [0, 1, 6, 7], 3] = 0.9
    attention[:, [0, 1, 6, 7], 4] = 0.4
    attention[:, [0, 1, 6, 7], 5] = 0.8
    attention[:, [0, 1, 6, 7], [0, 1, 6, 7]] = 0.25

    out = select_sparsevlm_tokens(
        image_features,
        attention,
        merged_seq_len=8,
        image_start=2,
        n_visual_tokens=4,
        retention_ratio=0.5,
    )

    assert out.tokens.shape == (1, 2, 3)
    assert out.kept_indices.detach().cpu().numpy().tolist() == [1]
    assert out.recycled_count == 3


def test_pyramiddrop_similarity_proxy_keeps_budget():
    torch = pytest.importorskip("torch")
    image_features = torch.eye(6, dtype=torch.float32).view(1, 6, 6)

    out = select_pyramiddrop_tokens(image_features, retention_ratio=0.5)

    assert out.tokens.shape == (1, 3, 6)
    assert out.kept_indices.numel() == 3
    assert out.mean_redundancy <= 1.0


def test_vtw_proxy_keeps_boundary_tokens_and_budget():
    torch = pytest.importorskip("torch")
    image_features = torch.arange(1 * 8 * 2, dtype=torch.float32).view(1, 8, 2)

    out = select_vtw_proxy_tokens(image_features, retention_ratio=0.5)

    kept = out.kept_indices.detach().cpu().numpy().tolist()
    assert out.tokens.shape == (1, 4, 2)
    assert 0 in kept
    assert 7 in kept
    assert out.boundary_count == 2


def test_visionzip_budget_reserves_cls_token():
    total, dominant_patch, contextual = visionzip_budget(576, 0.5)

    assert total == 288
    assert dominant_patch + contextual + 1 == total
    assert contextual > 0


def test_visionzip_selects_cls_dominant_and_contextual_tokens():
    torch = pytest.importorskip("torch")
    hidden = torch.arange(1 * 7 * 4, dtype=torch.float32).view(1, 7, 4)
    metric = hidden[..., :2].clone()
    attention = torch.zeros((1, 2, 7, 7), dtype=torch.float32)
    # CLS attention ranks visual patch indices 3 and 5 highest.
    attention[:, :, 0, 1] = 0.1
    attention[:, :, 0, 2] = 0.2
    attention[:, :, 0, 3] = 0.9
    attention[:, :, 0, 4] = 0.3
    attention[:, :, 0, 5] = 0.8
    attention[:, :, 0, 6] = 0.4

    out = select_visionzip_tokens(
        hidden,
        attention,
        metric,
        retention_ratio=0.5,
        dominant_fraction=0.75,
    )

    assert out.tokens.shape == (1, 3, 4)
    assert out.dominant_indices.detach().cpu().numpy().tolist() == [[0, 3]]
    assert out.contextual_count == 1
    assert out.contextual_target_indices.shape == (1, 1)


def test_pyramiddrop_attention_selector_uses_budget_and_layers():
    torch = pytest.importorskip("torch")
    image_features = torch.eye(6, dtype=torch.float32).view(1, 6, 6)
    attentions = []
    for _ in range(4):
        attn = torch.zeros((1, 2, 10, 10), dtype=torch.float32)
        # Text positions [0,1,8,9], image span [2..7]. Prefer visual token 4.
        attn[:, :, [0, 1, 8, 9], 2 + 4] = 1.0
        attentions.append(attn)

    from calibprune.pruners.pyramiddrop import select_pyramiddrop_attention_tokens

    out = select_pyramiddrop_attention_tokens(
        image_features,
        attentions,
        merged_seq_len=10,
        image_start=2,
        n_visual_tokens=6,
        retention_ratio=0.5,
        layer_indices=(1, 2, 3),
    )

    assert out.tokens.shape == (1, 3, 6)
    assert out.kept_indices.numel() == 3
    assert out.layer_indices == (1, 2, 3)
    assert 4 in out.kept_indices.detach().cpu().numpy().tolist()


def test_vtw_attention_selector_keeps_boundary_and_attention_tokens():
    torch = pytest.importorskip("torch")
    image_features = torch.arange(1 * 8 * 2, dtype=torch.float32).view(1, 8, 2)
    attention = torch.zeros((2, 12, 12), dtype=torch.float32)
    # Text positions [0,1,10,11], image span [2..9]. Prefer visual token 4.
    attention[:, [0, 1, 10, 11], 2 + 4] = 1.0

    from calibprune.pruners.vtw import select_vtw_attention_tokens

    out = select_vtw_attention_tokens(
        image_features,
        attention,
        merged_seq_len=12,
        image_start=2,
        n_visual_tokens=8,
        retention_ratio=0.5,
    )

    kept = out.kept_indices.detach().cpu().numpy().tolist()
    assert out.tokens.shape == (1, 4, 2)
    assert 0 in kept
    assert 7 in kept
    assert 4 in kept
    assert out.attention_count == 2


def test_midlayer_index_env_override_clamps_and_validates(monkeypatch):
    from calibprune.models.loader import _midlayer_index

    monkeypatch.delenv("CALIBPRUNE_TEST_LAYER", raising=False)
    assert _midlayer_index("CALIBPRUNE_TEST_LAYER", 8, 32) == 8

    monkeypatch.setenv("CALIBPRUNE_TEST_LAYER", "-1")
    assert _midlayer_index("CALIBPRUNE_TEST_LAYER", 8, 32) == 31

    monkeypatch.setenv("CALIBPRUNE_TEST_LAYER", "100")
    assert _midlayer_index("CALIBPRUNE_TEST_LAYER", 8, 32) == 31

    monkeypatch.setenv("CALIBPRUNE_TEST_LAYER", "bad")
    with pytest.raises(ValueError):
        _midlayer_index("CALIBPRUNE_TEST_LAYER", 8, 32)