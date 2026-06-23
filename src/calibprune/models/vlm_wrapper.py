"""Unified VLM wrapper types."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class LogitsBundle:
    logits: torch.Tensor
    pred_token: int
    pred_text: str
    answer_set_logits: Optional[dict[str, float]] = None
    metadata: Optional[dict[str, object]] = None


def extract_answer_logits(
    vocab_logits: torch.Tensor,
    answer_token_ids: dict[str, list[int] | int],
) -> dict[str, float]:
    """Extract first-step answer logits for a closed answer set."""
    if vocab_logits.ndim != 1:
        raise ValueError("vocab_logits must be a 1D tensor.")
    scores: dict[str, float] = {}
    for answer, ids in answer_token_ids.items():
        token_ids = [ids] if isinstance(ids, int) else list(ids)
        selected = vocab_logits[torch.as_tensor(token_ids, dtype=torch.long, device=vocab_logits.device)]
        scores[answer] = float(torch.logsumexp(selected, dim=0).detach().cpu())
    return scores


def answer_logits_to_array(answer_set_logits: dict[str, float], answer_choices: tuple[str, ...]) -> np.ndarray:
    return np.asarray([answer_set_logits[choice] for choice in answer_choices], dtype=np.float64)


def merge_answer_spaces(answer_choice_rows: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    merged: list[str] = []
    for choices in answer_choice_rows:
        for choice in choices:
            if choice not in merged:
                merged.append(choice)
    return tuple(merged)


def answer_logits_to_fixed_array(
    answer_set_logits: dict[str, float],
    answer_choices: tuple[str, ...],
    answer_space: tuple[str, ...],
    fill_value: float = -1.0e9,
) -> np.ndarray:
    row = np.full(len(answer_space), fill_value, dtype=np.float64)
    offsets = {choice: idx for idx, choice in enumerate(answer_space)}
    for choice in answer_choices:
        row[offsets[choice]] = answer_set_logits[choice]
    return row
