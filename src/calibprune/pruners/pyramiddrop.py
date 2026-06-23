"""PyramidDrop-style visual redundancy reduction utilities.

PyramidDrop progressively removes visual redundancy across language-model
layers.  A fully faithful implementation needs mid-layer sequence surgery.  The
selector below moves the local hook closer to the paper semantics by using
multi-layer decoder text-to-image attention plus a diversity constraint before
rerunning the language model on the selected visual tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calibprune.pruners.base import ScorePruner, kept_count


@dataclass(frozen=True)
class PyramidDropOutput:
    tokens: Any
    kept_indices: Any
    mean_redundancy: float
    layer_indices: tuple[int, ...] = ()


def _diverse_select(features: Any, scores: Any, k: int) -> tuple[Any, float]:
    import torch

    n_tokens = int(features.shape[0])
    normed = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = normed @ normed.T
    redundancy = (similarity.sum(dim=1) - 1.0) / max(1, n_tokens - 1)
    selected = [int(torch.argmax(scores).item())]
    min_distance = 1.0 - similarity[selected[0]]
    score_scale = scores.abs().max().clamp_min(1e-12)
    for _ in range(1, k):
        diversity_bonus = min_distance.clamp_min(0.0)
        combined = scores / score_scale + 0.25 * diversity_bonus
        combined[torch.as_tensor(selected, device=features.device)] = -float("inf")
        next_idx = int(torch.argmax(combined).item())
        selected.append(next_idx)
        min_distance = torch.minimum(min_distance, 1.0 - similarity[next_idx])
    indices = torch.as_tensor(sorted(selected), dtype=torch.long, device=features.device)
    return indices, float(redundancy[indices].mean().detach().cpu().item())


def select_pyramiddrop_tokens(image_features: Any, retention_ratio: float) -> PyramidDropOutput:
    """Fallback visual-boundary selector when decoder attentions are unavailable."""

    import torch

    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise ValueError("PyramidDrop hook expects image_features shaped [1, tokens, dim].")
    n_tokens = int(image_features.shape[1])
    k = kept_count(n_tokens, retention_ratio)
    if k >= n_tokens:
        indices = torch.arange(n_tokens, device=image_features.device)
        return PyramidDropOutput(tokens=image_features, kept_indices=indices, mean_redundancy=0.0)

    features = image_features[0]
    scores = -features.norm(dim=-1)
    indices, mean_redundancy = _diverse_select(features, scores, k)
    tokens = image_features.index_select(1, indices)
    return PyramidDropOutput(tokens=tokens, kept_indices=indices, mean_redundancy=mean_redundancy)


def _text_to_image_scores(attention: Any, *, merged_seq_len: int, image_start: int, n_visual_tokens: int) -> Any:
    import torch

    device = attention.device
    image_positions = torch.arange(image_start, image_start + n_visual_tokens, device=device)
    text_mask = torch.ones(merged_seq_len, dtype=torch.bool, device=device)
    text_mask[image_positions] = False
    text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
    if text_positions.numel() == 0:
        raise ValueError("PyramidDrop attention selector needs at least one text token.")
    return attention[:, text_positions][:, :, image_positions].mean(dim=(0, 1))


def select_pyramiddrop_attention_tokens(
    image_features: Any,
    attentions: Any,
    *,
    merged_seq_len: int,
    image_start: int,
    n_visual_tokens: int,
    retention_ratio: float,
    layer_indices: tuple[int, ...] = (8, 16, 24),
) -> PyramidDropOutput:
    """Select visual tokens using multi-layer text attention plus diversity."""

    import torch

    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise ValueError("PyramidDrop hook expects image_features shaped [1, tokens, dim].")
    k = kept_count(n_visual_tokens, retention_ratio)
    if k >= n_visual_tokens:
        indices = torch.arange(n_visual_tokens, device=image_features.device)
        return PyramidDropOutput(tokens=image_features, kept_indices=indices, mean_redundancy=0.0, layer_indices=())
    if not attentions:
        return select_pyramiddrop_tokens(image_features, retention_ratio)

    chosen_layers = tuple(sorted({min(max(0, idx), len(attentions) - 1) for idx in layer_indices}))
    weights = torch.linspace(1.0, 2.0, steps=len(chosen_layers), device=image_features.device)
    scores = torch.zeros(n_visual_tokens, dtype=torch.float32, device=image_features.device)
    for weight, layer_idx in zip(weights, chosen_layers):
        layer_attn = attentions[layer_idx][0]
        scores = scores + weight.to(scores.dtype) * _text_to_image_scores(
            layer_attn,
            merged_seq_len=merged_seq_len,
            image_start=image_start,
            n_visual_tokens=n_visual_tokens,
        ).to(scores.dtype)
    indices, mean_redundancy = _diverse_select(image_features[0], scores, k)
    return PyramidDropOutput(
        tokens=image_features.index_select(1, indices),
        kept_indices=indices,
        mean_redundancy=mean_redundancy,
        layer_indices=chosen_layers,
    )


class PyramidDropPruner(ScorePruner):
    def __init__(self) -> None:
        super().__init__(name="pyramiddrop")
        self.requires_attentions = True
