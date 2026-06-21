"""TransformerEncoder — a mini-transformer over the cognitive tokens.

Replaces SelectiveEncoder for cases where explicit inter-layer relationships are needed
DURING the formation of the cognitive tokens (rather than as post-processing via
token_preference in `BottleneckCrossAttention`).

Phase 8 finding: the transformer encoder unlocked self-correction on Llama-1B
(correction_acc 50%, where feedforward Selective had 0 attempts). Empirically:
the encoder's internal design determines which properties of the model the
introspection can express. Feedforward gives refusal calibration; transformer — self-correction.

Pipeline:

    activations [B, hidden_dim] × n_layers
        → per-layer projectors (Linear → encoder_dim)
        → + learnable layer position embeddings
        → input LayerNorm + dropout
        → N × TransformerBlock (pre-norm: LN → Self-Attn → res → LN → FFN → res)
        → output_proj (LN + Linear→hidden + GELU + Linear→hidden)
        → output_norm (LN)
        → cognitive tokens                            [B, n_layers, hidden_dim]

NO gates inside the transformer blocks — Phase 7 showed they create a bottleneck
for gradients. A near-identity start is ensured by the standard init with a small
amplitude (std=0.02) on the sub-layers' outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from meta_core.encoders.base import Encoder

__all__ = ["TransformerEncoder"]


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Self-Attn → res → LN → FFN → res.

    NO gates (Phase 7 showed they bottleneck the gradients).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must divide num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Self-attention sub-layer
        self.norm_attn = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN sub-layer
        self.norm_ffn = nn.LayerNorm(dim)
        ffn_inner = dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_inner, bias=False),
            nn.GELU(),
            nn.Linear(ffn_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.02)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        h = self.norm_attn(x)
        batch_size, num_tokens, _ = h.shape

        Q = self.q_proj(h).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dim)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        # FFN
        residual = x
        h = self.norm_ffn(x)
        h = self.ffn(h)
        x = residual + h
        return x


class TransformerEncoder(nn.Module, Encoder):
    """Mini-transformer over the layers' activations → cognitive tokens.

    Args:
        hidden_dim: the base model's hidden state dimensionality (2048 for 1B, 4096 for 8B).
        num_layers: number of target layers (= number of cognitive tokens).
        encoder_dim: the encoder's internal dimensionality (default 384 for 1B, 512 for 8B).
        num_blocks: number of transformer blocks (default 2 — validated in Phase 8).
        num_heads: number of attention heads inside the blocks (default 8).
        ffn_expansion: FFN multiplier (default 4).
        dropout: dropout (default 0.1).
        use_per_layer_projectors: per-layer Linear (True) vs shared projector (False).
        use_layer_pos_embeddings: add a learnable position embedding per layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 16,
        encoder_dim: int = 384,
        num_blocks: int = 2,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        use_per_layer_projectors: bool = True,
        use_layer_pos_embeddings: bool = True,
        prenorm_projector: bool = False,
    ):
        nn.Module.__init__(self)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.encoder_dim = encoder_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.dropout = dropout
        self.use_per_layer_projectors = use_per_layer_projectors
        self.use_layer_pos_embeddings = use_layer_pos_embeddings
        self.prenorm_projector = prenorm_projector

        # Per-layer projectors (like an embedding lookup) or shared.
        # prenorm_projector=True adds LayerNorm + GELU around the Linear (Phase 8 convention).
        def _build_projector() -> nn.Module:
            if prenorm_projector:
                return nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, encoder_dim),
                    nn.GELU(),
                )
            return nn.Linear(hidden_dim, encoder_dim, bias=False)

        if use_per_layer_projectors:
            self.layer_projectors = nn.ModuleList([_build_projector() for _ in range(num_layers)])
        else:
            self.shared_projector = _build_projector()

        # Learnable layer position embeddings (optional)
        if use_layer_pos_embeddings:
            self.layer_pos_embed = nn.Parameter(
                torch.randn(num_layers, encoder_dim) * 0.02
            )

        self.input_norm = nn.LayerNorm(encoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                dim=encoder_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Output projection back into the base's embedding space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self._init_input_layers()

    def _init_input_layers(self) -> None:
        def _init_proj(proj: nn.Module) -> None:
            # With prenorm_projector the projector is Sequential(LN, Linear, GELU); init only the Linear
            if isinstance(proj, nn.Linear):
                nn.init.normal_(proj.weight, std=0.02)
            elif isinstance(proj, nn.Sequential):
                for sub in proj:
                    if isinstance(sub, nn.Linear):
                        nn.init.normal_(sub.weight, std=0.02)

        if self.use_per_layer_projectors:
            for proj in self.layer_projectors:
                _init_proj(proj)
        else:
            _init_proj(self.shared_projector)
        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

    def forward(self, activation_list: list[torch.Tensor]) -> torch.Tensor:
        """Run the activations through the mini-transformer encoder.

        Args:
            activation_list: list of num_layers tensors [B, hidden_dim].

        Returns:
            cognitive_tokens: [B, num_layers, hidden_dim].
        """
        assert len(activation_list) == self._num_layers, (
            f"Expected {self._num_layers} activations, got {len(activation_list)}"
        )

        # Input projection
        if self.use_per_layer_projectors:
            projected = [
                self.layer_projectors[i](activation_list[i])
                for i in range(self._num_layers)
            ]
        else:
            projected = [self.shared_projector(act) for act in activation_list]

        # [B, num_layers, encoder_dim]
        x = torch.stack(projected, dim=1)

        # Layer position embeddings
        if self.use_layer_pos_embeddings:
            x = x + self.layer_pos_embed.unsqueeze(0).to(x.dtype)

        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output → [B, num_layers, hidden_dim]
        cognitive_tokens = self.output_proj(x)
        cognitive_tokens = self.output_norm(cognitive_tokens)
        return cognitive_tokens

    # --- Diagnostics ---

    def num_trainable_params(self) -> int:
        """Sum of numel() over all trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers
