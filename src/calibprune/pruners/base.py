"""Visual token pruner contracts and implemented lightweight baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PruneOutput:
    tokens: Any
    kept_indices: np.ndarray


class Pruner(Protocol):
    name: str
    requires_attentions: bool

    def __call__(
        self,
        visual_tokens: Any,
        attentions: Any | None,
        retention_ratio: float,
        layer_idx: int = 0,
    ) -> PruneOutput:
        ...


def kept_count(n_tokens: int, retention_ratio: float) -> int:
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be in (0, 1].")
    return max(1, min(n_tokens, int(round(retention_ratio * n_tokens))))


def _slice_tokens(tokens: Any, indices: np.ndarray) -> Any:
    if hasattr(tokens, "index_select"):
        import torch

        idx = torch.as_tensor(indices, dtype=torch.long, device=tokens.device)
        return tokens.index_select(0, idx)
    return np.asarray(tokens)[indices]


def _token_norm_scores(tokens: Any) -> np.ndarray:
    arr = tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else np.asarray(tokens)
    arr = arr.astype(np.float32, copy=False)
    return np.linalg.norm(arr, axis=-1)


class NonePruner:
    name = "none"
    requires_attentions = False

    def __call__(self, visual_tokens: Any, attentions: Any | None, retention_ratio: float, layer_idx: int = 0) -> PruneOutput:
        n = int(visual_tokens.shape[0])
        return PruneOutput(tokens=visual_tokens, kept_indices=np.arange(n, dtype=np.int64))


class RandomPruner:
    name = "random"
    requires_attentions = False

    def __init__(self, seed: int = 20260616) -> None:
        self.seed = seed

    def __call__(self, visual_tokens: Any, attentions: Any | None, retention_ratio: float, layer_idx: int = 0) -> PruneOutput:
        n = int(visual_tokens.shape[0])
        k = kept_count(n, retention_ratio)
        rng = np.random.default_rng(self.seed + layer_idx)
        indices = np.sort(rng.choice(np.arange(n), size=k, replace=False))
        return PruneOutput(tokens=_slice_tokens(visual_tokens, indices), kept_indices=indices)


class UniformPruner:
    name = "uniform"
    requires_attentions = False

    def __call__(self, visual_tokens: Any, attentions: Any | None, retention_ratio: float, layer_idx: int = 0) -> PruneOutput:
        n = int(visual_tokens.shape[0])
        k = kept_count(n, retention_ratio)
        indices = np.unique(np.round(np.linspace(0, n - 1, k)).astype(np.int64))
        while len(indices) < k:
            missing = np.setdiff1d(np.arange(n), indices, assume_unique=False)
            indices = np.sort(np.concatenate([indices, missing[: k - len(indices)]]))
        return PruneOutput(tokens=_slice_tokens(visual_tokens, indices), kept_indices=indices)


class ScorePruner:
    requires_attentions = False

    def __init__(self, name: str, use_low_scores: bool = False) -> None:
        self.name = name
        self.use_low_scores = use_low_scores

    def score(self, visual_tokens: Any, attentions: Any | None, layer_idx: int) -> np.ndarray:
        return _token_norm_scores(visual_tokens)

    def __call__(self, visual_tokens: Any, attentions: Any | None, retention_ratio: float, layer_idx: int = 0) -> PruneOutput:
        n = int(visual_tokens.shape[0])
        k = kept_count(n, retention_ratio)
        scores = np.asarray(self.score(visual_tokens, attentions, layer_idx), dtype=np.float64)[:n]
        order = np.argsort(scores)
        if not self.use_low_scores:
            order = order[::-1]
        indices = np.sort(order[:k].astype(np.int64))
        return PruneOutput(tokens=_slice_tokens(visual_tokens, indices), kept_indices=indices)


class FeatureNormPruner(ScorePruner):
    """Keep visual tokens with the largest projected feature norms.

    This is a sanity baseline for the local experiment pipeline, not a
    published training-free pruning method.
    """

    def __init__(self) -> None:
        super().__init__(name="feature_norm")


GENERIC_FEATURE_PRUNERS = {"none", "random", "uniform", "feature_norm"}
MODEL_HOOK_PRUNERS = {"fastv", "sparsevlm", "visionzip", "pyramiddrop", "vtw"}
LITERATURE_PLACEHOLDER_PRUNERS: set[str] = set()


def build_pruner(name: str) -> Pruner:
    name = name.lower()
    if name == "none":
        return NonePruner()
    if name == "random":
        return RandomPruner()
    if name == "uniform":
        return UniformPruner()
    if name == "feature_norm":
        return FeatureNormPruner()
    if name in MODEL_HOOK_PRUNERS:
        raise NotImplementedError(
            f"{name} is implemented as a model-specific forward hook, not as a generic feature pruner."
        )
    if name in LITERATURE_PLACEHOLDER_PRUNERS:
        raise NotImplementedError(
            f"{name} is listed in the experiment plan, but its faithful model hook is not implemented yet."
        )
    raise ValueError(f"Unknown pruner: {name}")
