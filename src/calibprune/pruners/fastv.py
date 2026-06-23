"""FastV attention scoring utilities.

FastV is model-hook based in this repository: LLaVA exposes the merged
language sequence, and this module owns the text-to-image attention scoring used
by that hook.  It is intentionally separate from generic feature pruners so the
paper baseline is not silently replaced by an L2 feature fallback.
"""

from __future__ import annotations

from typing import Any

from calibprune.pruners.base import ScorePruner, kept_count


def select_fastv_indices(
    attention: Any,
    *,
    merged_seq_len: int,
    image_start: int,
    n_visual_tokens: int,
    retention_ratio: float,
) -> Any:
    """Select visual token indices by average text-to-image attention.

    Parameters are expressed in the merged language-model sequence, where the
    visual features replace the single image placeholder token. ``attention`` is
    expected to be a tensor shaped ``[heads, query, key]`` for one sample.
    """

    if n_visual_tokens <= 0:
        raise ValueError("n_visual_tokens must be positive.")
    if image_start < 0 or image_start + n_visual_tokens > merged_seq_len:
        raise ValueError("Image token span is outside the merged sequence.")
    if attention.ndim != 3:
        raise ValueError("FastV expects an attention tensor shaped [heads, query, key].")

    import torch

    device = attention.device
    image_positions = torch.arange(image_start, image_start + n_visual_tokens, device=device)
    text_mask = torch.ones(merged_seq_len, dtype=torch.bool, device=device)
    text_mask[image_positions] = False
    text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
    if text_positions.numel() == 0:
        raise ValueError("FastV needs at least one text token to score visual tokens.")

    scores = attention[:, text_positions][:, :, image_positions].mean(dim=(0, 1))
    k = kept_count(n_visual_tokens, retention_ratio)
    return torch.topk(scores, k=k, largest=True).indices.sort().values


class FastVPruner(ScorePruner):
    def __init__(self) -> None:
        super().__init__(name="fastv")
        self.requires_attentions = True
