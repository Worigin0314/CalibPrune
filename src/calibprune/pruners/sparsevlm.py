"""SparseVLM-style text-guided visual token sparsification.

The full SparseVLM method applies layer-adaptive sparsification inside the
language model and recycles pruned visual tokens.  The LLaVA hook in this
repository implements the executable single-stage core needed for calibration
experiments: score image tokens by text-guided self-attention, keep the most
relevant tokens, and recycle the dropped tokens into one compact visual token
when the budget allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calibprune.pruners.base import ScorePruner, kept_count


@dataclass(frozen=True)
class SparseVLMOutput:
    tokens: Any
    kept_indices: Any
    recycled_count: int
    recycled_weight: float


def sparsevlm_text_guided_scores(
    attention: Any,
    *,
    merged_seq_len: int,
    image_start: int,
    n_visual_tokens: int,
) -> Any:
    """Score image tokens by text-query to image-key attention.

    ``attention`` is one-sample language-model self-attention shaped
    ``[heads, query, key]`` after image features have been merged into the text
    sequence.
    """

    import torch

    if n_visual_tokens <= 0:
        raise ValueError("n_visual_tokens must be positive.")
    if attention.ndim != 3:
        raise ValueError("SparseVLM expects an attention tensor shaped [heads, query, key].")
    if image_start < 0 or image_start + n_visual_tokens > merged_seq_len:
        raise ValueError("Image token span is outside the merged sequence.")

    device = attention.device
    image_positions = torch.arange(image_start, image_start + n_visual_tokens, device=device)
    text_mask = torch.ones(merged_seq_len, dtype=torch.bool, device=device)
    text_mask[image_positions] = False
    text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
    if text_positions.numel() == 0:
        raise ValueError("SparseVLM needs at least one text token to guide pruning.")

    text_to_image = attention[:, text_positions][:, :, image_positions].mean(dim=0)
    text_importance = attention[:, text_positions][:, :, text_positions].mean(dim=(0, 2))
    text_importance = text_importance / text_importance.sum().clamp_min(1e-12)
    return (text_to_image * text_importance[:, None]).sum(dim=0)


def select_sparsevlm_tokens(
    image_features: Any,
    attention: Any,
    *,
    merged_seq_len: int,
    image_start: int,
    n_visual_tokens: int,
    retention_ratio: float,
) -> SparseVLMOutput:
    """Keep text-relevant tokens and recycle dropped tokens into one vector."""

    import torch

    scores = sparsevlm_text_guided_scores(
        attention,
        merged_seq_len=merged_seq_len,
        image_start=image_start,
        n_visual_tokens=n_visual_tokens,
    )
    k = kept_count(n_visual_tokens, retention_ratio)
    if k >= n_visual_tokens:
        indices = torch.arange(n_visual_tokens, device=image_features.device)
        return SparseVLMOutput(tokens=image_features, kept_indices=indices, recycled_count=0, recycled_weight=0.0)

    reserve_recycled = k > 1
    keep_k = k - 1 if reserve_recycled else k
    keep = torch.topk(scores, k=keep_k, largest=True).indices.sort().values
    kept_tokens = image_features[:, keep, :]
    if not reserve_recycled:
        return SparseVLMOutput(tokens=kept_tokens, kept_indices=keep, recycled_count=0, recycled_weight=0.0)

    keep_mask = torch.zeros(n_visual_tokens, dtype=torch.bool, device=image_features.device)
    keep_mask[keep] = True
    dropped = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    dropped_scores = scores[dropped]
    weights = torch.softmax(dropped_scores, dim=0).to(image_features.dtype)
    recycled = torch.sum(image_features[:, dropped, :] * weights.view(1, -1, 1), dim=1, keepdim=True)
    tokens = torch.cat([kept_tokens, recycled], dim=1)
    return SparseVLMOutput(
        tokens=tokens,
        kept_indices=keep,
        recycled_count=int(dropped.numel()),
        recycled_weight=float(weights.max().detach().cpu().item()) if weights.numel() else 0.0,
    )


class SparseVLMPruner(ScorePruner):
    def __init__(self) -> None:
        super().__init__(name="sparsevlm")
        self.requires_attentions = True
