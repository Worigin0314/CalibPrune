"""VisionZip-style CLIP vision-token selection for LLaVA.

The implemented core follows the training-free VisionZip recipe: use the CLIP
CLS token attention to select dominant visual tokens, then merge the remaining
tokens into contextual tokens using key-space similarity. The runtime hook in
``models.loader`` applies this before the multimodal projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calibprune.pruners.base import ScorePruner, kept_count


DEFAULT_DOMINANT_FRACTION = 54.0 / 64.0
DEFAULT_SELECT_LAYER = -2


@dataclass(frozen=True)
class VisionZipOutput:
    tokens: Any
    dominant_indices: Any
    contextual_target_indices: Any
    contextual_count: int


def visionzip_budget(
    n_visual_tokens: int,
    retention_ratio: float,
    *,
    dominant_fraction: float = DEFAULT_DOMINANT_FRACTION,
) -> tuple[int, int, int]:
    """Return ``(total_tokens, dominant_patch_tokens, contextual_tokens)``.

    The total budget follows the repository retention ratio. One dominant slot
    is reserved for the CLS token, mirroring the official LLaVA VisionZip hook.
    """

    if not 0.0 < dominant_fraction <= 1.0:
        raise ValueError("dominant_fraction must be in (0, 1].")
    total_tokens = kept_count(n_visual_tokens, retention_ratio)
    if total_tokens == 1:
        return total_tokens, 0, 0
    contextual = int(round(total_tokens * (1.0 - dominant_fraction)))
    contextual = max(0, min(total_tokens - 1, contextual))
    dominant_patch = total_tokens - contextual - 1
    dominant_patch = max(0, min(n_visual_tokens, dominant_patch))
    contextual = total_tokens - dominant_patch - 1
    return total_tokens, dominant_patch, contextual


def _select_layer_index(n_layers: int, select_layer: int) -> int:
    layer_idx = n_layers + select_layer if select_layer < 0 else select_layer
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"select_layer {select_layer} is outside {n_layers} CLIP layers.")
    return layer_idx


def clip_key_metric_from_hidden_states(
    vision_tower: Any,
    hidden_states: tuple[Any, ...],
    *,
    select_layer: int = DEFAULT_SELECT_LAYER,
) -> Any:
    """Reconstruct the key metric used by the official VisionZip CLIP patch."""

    n_layers = len(vision_tower.vision_model.encoder.layers)
    layer_idx = _select_layer_index(n_layers, select_layer)
    layer = vision_tower.vision_model.encoder.layers[layer_idx]
    layer_input = hidden_states[layer_idx]
    normed = layer.layer_norm1(layer_input)
    key_states = layer.self_attn._shape(layer.self_attn.k_proj(normed), -1, normed.shape[0])
    return key_states.mean(1)


def select_visionzip_tokens(
    hidden_states_with_cls: Any,
    attentions: Any,
    key_metric_with_cls: Any,
    retention_ratio: float,
    *,
    dominant_fraction: float = DEFAULT_DOMINANT_FRACTION,
) -> VisionZipOutput:
    """Select dominant tokens and merge contextual tokens.

    All tensors are expected to include the leading CLIP CLS token. Attention is
    shaped ``[batch, heads, seq, seq]``.
    """

    import torch

    if hidden_states_with_cls.ndim != 3:
        raise ValueError("hidden_states_with_cls must have shape [batch, seq, dim].")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape [batch, heads, seq, seq].")
    if key_metric_with_cls.ndim != 3:
        raise ValueError("key_metric_with_cls must have shape [batch, seq, dim].")
    if hidden_states_with_cls.shape[:2] != key_metric_with_cls.shape[:2]:
        raise ValueError("hidden states and key metric must share batch and sequence dimensions.")
    if attentions.shape[0] != hidden_states_with_cls.shape[0] or attentions.shape[-1] != hidden_states_with_cls.shape[1]:
        raise ValueError("attention shape is incompatible with hidden states.")

    batch, seq_len, hidden_dim = hidden_states_with_cls.shape
    n_visual = seq_len - 1
    if n_visual <= 0:
        raise ValueError("VisionZip expects at least one visual patch token after CLS.")
    total_tokens, dominant_patch_count, contextual_count = visionzip_budget(
        n_visual,
        retention_ratio,
        dominant_fraction=dominant_fraction,
    )

    if dominant_patch_count > 0:
        cls_attention = attentions[:, :, 0, 1:].sum(dim=1)
        dominant_patches = torch.topk(cls_attention, k=dominant_patch_count, dim=1).indices + 1
        dominant_patches = dominant_patches.sort(dim=1).values
    else:
        dominant_patches = torch.empty((batch, 0), dtype=torch.long, device=hidden_states_with_cls.device)
    cls_index = torch.zeros((batch, 1), dtype=torch.long, device=hidden_states_with_cls.device)
    dominant_indices = torch.cat([cls_index, dominant_patches], dim=1)
    dominant_tokens = torch.gather(
        hidden_states_with_cls,
        dim=1,
        index=dominant_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )

    if contextual_count <= 0 or dominant_indices.shape[1] >= seq_len:
        return VisionZipOutput(
            tokens=dominant_tokens[:, :total_tokens, :],
            dominant_indices=dominant_indices,
            contextual_target_indices=torch.empty((batch, 0), dtype=torch.long, device=hidden_states_with_cls.device),
            contextual_count=0,
        )

    all_indices = torch.arange(seq_len, device=hidden_states_with_cls.device).unsqueeze(0).expand(batch, -1)
    remaining_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=hidden_states_with_cls.device)
    remaining_mask.scatter_(1, dominant_indices, False)
    remaining_indices = all_indices[remaining_mask].view(batch, seq_len - dominant_indices.shape[1])
    contextual_count = min(contextual_count, remaining_indices.shape[1])
    step = max(1, remaining_indices.shape[1] // contextual_count)
    target_rel = torch.arange(0, remaining_indices.shape[1], step, device=hidden_states_with_cls.device)[:contextual_count]
    target_indices = remaining_indices[:, target_rel]

    remaining_hidden = torch.gather(
        hidden_states_with_cls,
        dim=1,
        index=remaining_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )
    remaining_metric = torch.gather(
        key_metric_with_cls,
        dim=1,
        index=remaining_indices.unsqueeze(-1).expand(-1, -1, key_metric_with_cls.shape[-1]),
    )
    metric_normalized = remaining_metric / remaining_metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target_metric = metric_normalized[:, target_rel, :]
    target_hidden = torch.gather(
        hidden_states_with_cls,
        dim=1,
        index=target_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )

    merge_mask = torch.ones((batch, remaining_indices.shape[1]), dtype=torch.bool, device=hidden_states_with_cls.device)
    merge_mask[:, target_rel] = False
    if not merge_mask.any():
        contextual_tokens = target_hidden
    else:
        merge_metric = metric_normalized[merge_mask].view(batch, -1, metric_normalized.shape[-1])
        merge_hidden = remaining_hidden[merge_mask].view(batch, -1, hidden_dim)
        similarity = torch.bmm(merge_metric, target_metric.transpose(1, 2))
        assignments = similarity.argmax(dim=2)
        assign_one_hot = torch.zeros(
            (batch, merge_metric.shape[1], contextual_count),
            dtype=hidden_states_with_cls.dtype,
            device=hidden_states_with_cls.device,
        )
        assign_one_hot.scatter_(2, assignments.unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated = torch.bmm(assign_one_hot.transpose(1, 2), merge_hidden) / counts
        contextual_tokens = target_hidden + aggregated

    return VisionZipOutput(
        tokens=torch.cat([dominant_tokens, contextual_tokens], dim=1)[:, :total_tokens, :],
        dominant_indices=dominant_indices,
        contextual_target_indices=target_indices,
        contextual_count=int(contextual_tokens.shape[1]),
    )


class VisionZipPruner(ScorePruner):
    def __init__(self) -> None:
        super().__init__(name="visionzip")
        self.requires_attentions = True
