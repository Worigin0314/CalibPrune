"""Visual Token Withdrawal utilities.

VTW withdraws visual tokens after early layers once visual information has
migrated to text tokens.  A fully faithful implementation requires continuing
the decoder layer loop after removing the visual span.  The local selector uses
decoder attention to keep boundary and still text-relevant visual tokens before
a rerun, and records this as an attention-withdrawal approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calibprune.pruners.base import ScorePruner, kept_count


@dataclass(frozen=True)
class VTWOutput:
    tokens: Any
    kept_indices: Any
    boundary_count: int
    attention_count: int = 0


def select_vtw_proxy_tokens(image_features: Any, retention_ratio: float, *, boundary_fraction: float = 0.25) -> VTWOutput:
    """Fallback selector: keep boundary tokens and representative middle tokens."""

    import torch

    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise ValueError("VTW hook expects image_features shaped [1, tokens, dim].")
    n_tokens = int(image_features.shape[1])
    k = kept_count(n_tokens, retention_ratio)
    if k >= n_tokens:
        indices = torch.arange(n_tokens, device=image_features.device)
        return VTWOutput(tokens=image_features, kept_indices=indices, boundary_count=n_tokens, attention_count=0)

    boundary_indices = _boundary_indices(n_tokens, k, boundary_fraction, image_features.device)
    remaining_k = k - int(boundary_indices.numel())
    if remaining_k > 0:
        features = image_features[0]
        centroid = features.mean(dim=0, keepdim=True)
        distances = torch.norm(features - centroid, dim=-1)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=image_features.device)
        mask[boundary_indices] = False
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        chosen = candidates[torch.topk(-distances[candidates], k=remaining_k, largest=True).indices]
        indices = torch.cat([boundary_indices, chosen]).sort().values
    else:
        indices = boundary_indices.sort().values
    return VTWOutput(tokens=image_features.index_select(1, indices), kept_indices=indices, boundary_count=int(boundary_indices.numel()))


def _boundary_indices(n_tokens: int, k: int, boundary_fraction: float, device: Any) -> Any:
    import torch

    boundary = max(2, int(round(k * boundary_fraction)))
    boundary = min(boundary, k, n_tokens)
    left = boundary // 2
    right = boundary - left
    return torch.cat(
        [
            torch.arange(left, device=device),
            torch.arange(n_tokens - right, n_tokens, device=device),
        ]
    ).unique(sorted=True)


def _text_to_image_scores(attention: Any, *, merged_seq_len: int, image_start: int, n_visual_tokens: int) -> Any:
    import torch

    device = attention.device
    image_positions = torch.arange(image_start, image_start + n_visual_tokens, device=device)
    text_mask = torch.ones(merged_seq_len, dtype=torch.bool, device=device)
    text_mask[image_positions] = False
    text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
    if text_positions.numel() == 0:
        raise ValueError("VTW attention selector needs at least one text token.")
    return attention[:, text_positions][:, :, image_positions].mean(dim=(0, 1))


def select_vtw_attention_tokens(
    image_features: Any,
    attention: Any,
    *,
    merged_seq_len: int,
    image_start: int,
    n_visual_tokens: int,
    retention_ratio: float,
    boundary_fraction: float = 0.25,
) -> VTWOutput:
    """Keep boundary tokens plus text-relevant tokens from decoder attention."""

    import torch

    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise ValueError("VTW hook expects image_features shaped [1, tokens, dim].")
    k = kept_count(n_visual_tokens, retention_ratio)
    if k >= n_visual_tokens:
        indices = torch.arange(n_visual_tokens, device=image_features.device)
        return VTWOutput(tokens=image_features, kept_indices=indices, boundary_count=n_visual_tokens, attention_count=0)
    boundary_indices = _boundary_indices(n_visual_tokens, k, boundary_fraction, image_features.device)
    remaining_k = k - int(boundary_indices.numel())
    if remaining_k > 0:
        scores = _text_to_image_scores(
            attention,
            merged_seq_len=merged_seq_len,
            image_start=image_start,
            n_visual_tokens=n_visual_tokens,
        )
        mask = torch.ones(n_visual_tokens, dtype=torch.bool, device=image_features.device)
        mask[boundary_indices] = False
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        chosen = candidates[torch.topk(scores[candidates], k=remaining_k, largest=True).indices]
        indices = torch.cat([boundary_indices, chosen]).sort().values
    else:
        indices = boundary_indices.sort().values
        remaining_k = 0
    return VTWOutput(
        tokens=image_features.index_select(1, indices),
        kept_indices=indices,
        boundary_count=int(boundary_indices.numel()),
        attention_count=int(remaining_k),
    )


class VTWPruner(ScorePruner):
    def __init__(self) -> None:
        super().__init__(name="vtw")
        self.requires_attentions = True
