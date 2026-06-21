"""MultiTokenEncoder — the original Phase 1-2 encoder (Gemma-2-2B / Llama-8B without Selective).

Unlike `SelectiveEncoder` (1 cog token per activation layer) and `TransformerEncoder`
(self-attention over all tokens), `MultiTokenEncoder` uses **learnable queries**
with cross-attention over the activations:

    N activation layers → projected (LN + Linear + GELU) → K, V
                                                          ↓
                                                  cross_attn
                                                          ↑
                              M learnable queries → Q
                                                          ↓
                                                M cognitive tokens

Architecture from `publish/github/src/models/introspection_encoder.py:MultiTokenIntrospectionEncoder`.
Implements the Phase 1 (Gemma-2-2B) and Phase 2 Llama-8B convention: N=5 activation layers →
M=8 cognitive tokens.

Phase 1R record on Gemma-2-2B (trivia_qa 5K): refusal_precision 81-90% (4/5 checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from meta_core.encoders.base import Encoder

if TYPE_CHECKING:
    pass

__all__ = ["MultiTokenEncoder"]


class MultiTokenEncoder(nn.Module, Encoder):
    """N→M encoder with learnable queries + cross-attention.

    Unlike `SelectiveEncoder` (1 cog token per layer), MultiTokenEncoder
    maps N activation layers → M cognitive tokens independently. This lets
    M (number of cog tokens) differ from N (number of target layers).

    Args:
        hidden_dim: the base model's hidden state dimensionality.
        num_layers: number of target layers (N).
        num_cognitive_tokens: number of output cognitive tokens (M, default 8).
        bottleneck_dim: bottleneck for projector + queries + cross-attn (default 256).
        num_heads: number of heads in the cross-attention (default 8).
        dropout: dropout (default 0.1).
        prenorm_projector: whether to use a LayerNorm + GELU wrapper (default True
            per the `MultiTokenIntrospectionEncoder` convention).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_cognitive_tokens: int = 8,
        bottleneck_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        prenorm_projector: bool = True,
    ):
        nn.Module.__init__(self)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.num_cognitive_tokens = num_cognitive_tokens
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.prenorm_projector = prenorm_projector

        # Per-layer projectors: hidden_dim → bottleneck_dim
        def _build_projector() -> nn.Module:
            if prenorm_projector:
                return nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, bottleneck_dim),
                    nn.GELU(),
                )
            return nn.Linear(hidden_dim, bottleneck_dim, bias=False)

        self.layer_projectors = nn.ModuleList([
            _build_projector() for _ in range(num_layers)
        ])

        # Learnable queries: "what to ask" of the activations. Init with small values for
        # a near-identity start (per source convention std=0.02).
        self.queries = nn.Parameter(
            torch.randn(num_cognitive_tokens, bottleneck_dim) * 0.02
        )

        # Cross-attention: queries × kv
        self.cross_attn = nn.MultiheadAttention(
            bottleneck_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # Output projection: bottleneck → hidden_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, activation_list: list[torch.Tensor]) -> torch.Tensor:
        """Run the activations through the cross-attention encoder.

        Args:
            activation_list: list of `num_layers` tensors [B, hidden_dim].

        Returns:
            cognitive_tokens: [B, num_cognitive_tokens, hidden_dim].
        """
        assert len(activation_list) == self._num_layers, (
            f"Expected {self._num_layers} activations, got {len(activation_list)}"
        )

        batch_size = activation_list[0].shape[0]

        # Project each activation layer → bottleneck
        projected = [
            self.layer_projectors[i](activation_list[i])
            for i in range(self._num_layers)
        ]
        # [B, num_layers, bottleneck_dim] — K/V for cross-attn
        kv = torch.stack(projected, dim=1)

        # Expand queries across the batch: [B, num_cog, bottleneck]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries → kv → [B, num_cog, bottleneck]
        cognitive_raw, _ = self.cross_attn(queries, kv, kv)

        # Output projection into hidden_dim space
        cognitive_tokens = self.output_proj(cognitive_raw)
        cognitive_tokens = self.output_norm(cognitive_tokens)
        return cognitive_tokens

    # --- Diagnostics ---

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers
