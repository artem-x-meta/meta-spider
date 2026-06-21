"""IntrospectionCache — adaptive per-token refresh of cognitive tokens.

In static two-pass generation cognitive tokens are computed ONCE from the prompt and
frozen for the whole answer. For long reasoning (math, agentic chains) this is bad: the
model's uncertainty evolves over the course of generation, and a frozen signal misses it.

IntrospectionCache solves this: it periodically re-collects activations of the current
context (question + partial answer) and decides whether to refresh the cognitive tokens,
based on the cosine similarity between the current and cached activations. It refreshes
when the model's state has diverged (sim < threshold) — otherwise it reuses the cache (cheap).

Port of `publish/github/src/phase3_dynamic_llama8b/introspection_cache.py` (March 2026,
validated on GSM8K: dynamic +7.2pp vs static, threshold 0.5 optimal).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["IntrospectionCache"]


class IntrospectionCache:
    """Decides when to recompute cognitive tokens during generation.

    Args:
        threshold: cosine similarity below which we refresh (0.5 optimal for GSM8K;
            should be above the dataset's average similarity, otherwise the cache is never
            invalidated).
        min_interval: minimum tokens between refreshes (stabilization after a refresh).
        max_interval: maximum tokens without a refresh (forced refresh).
    """

    def __init__(self, threshold: float = 0.5, min_interval: int = 3,
                 max_interval: int = 20):
        self.threshold = threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.last_activations: Optional[list[torch.Tensor]] = None
        self.last_cognitive_tokens: Optional[torch.Tensor] = None
        self.tokens_since_refresh = 0
        self.refresh_count = 0
        self.total_tokens = 0
        self.similarity_history: list[float] = []

    def should_refresh(self, current_activations: list[torch.Tensor]) -> bool:
        """True if: no cache (first time) OR max_interval reached OR
        (min_interval passed AND cosine similarity dropped below threshold)."""
        if self.last_activations is None:
            return True
        if self.tokens_since_refresh >= self.max_interval:
            return True
        if self.tokens_since_refresh < self.min_interval:
            return False
        avg_sim = self._avg_similarity(current_activations)
        self.similarity_history.append(avg_sim)
        return avg_sim < self.threshold

    def _avg_similarity(self, current: list[torch.Tensor]) -> float:
        sims = [
            F.cosine_similarity(c.flatten().float(), l.flatten().float(), dim=0).item()
            for c, l in zip(current, self.last_activations)
        ]
        return sum(sims) / len(sims) if sims else 1.0

    def store(self, activations: list[torch.Tensor],
              cognitive_tokens: torch.Tensor) -> None:
        self.last_activations = [a.detach().clone() for a in activations]
        self.last_cognitive_tokens = cognitive_tokens.detach().clone()
        self.tokens_since_refresh = 0
        self.refresh_count += 1

    def tick(self, n_tokens: int = 1) -> None:
        self.tokens_since_refresh += n_tokens
        self.total_tokens += n_tokens

    def clear(self) -> None:
        self.last_activations = None
        self.last_cognitive_tokens = None
        self.tokens_since_refresh = 0
        self.refresh_count = 0
        self.total_tokens = 0
        self.similarity_history.clear()

    def get_stats(self) -> dict:
        return {
            "total_refreshes": self.refresh_count,
            "total_tokens": self.total_tokens,
            "avg_refresh_interval": self.total_tokens / max(self.refresh_count, 1),
            "cache_hit_rate": 1.0 - self.refresh_count / max(self.total_tokens, 1),
            "avg_similarity": (sum(self.similarity_history) / len(self.similarity_history)
                               if self.similarity_history else 0.0),
        }
