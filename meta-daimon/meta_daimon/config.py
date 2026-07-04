"""Meta-Daimon voice configs — per-modifier dataclasses on top of the meta-core mechanism.

`MetaSpiderConfig` (the mechanism: base model, layers, dtype) stays in `meta_core.config`;
here live the VOICE configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from meta_core.config import EncoderType

__all__ = ["DoubterConfig"]


@dataclass
class DoubterConfig:
    """Config for the Doubter modifier — cognitive-token pipeline.

    Two-pass forward through a frozen base + trainable wrapper (encoder +
    32 BottleneckCrossAttention). Calibration is achieved **through training** (target =
    confirm / correct / refuse in Phase 2), not via a post-hoc threshold.

    Defaults — from Phase 2 Selective Llama-8B (record selective accuracy 89.1%).
    """

    encoder_type: EncoderType = "selective"
    encoder_bottleneck: int = 256
    encoder_gate_init: float = 0.3
    encoder_gate_lr_multiplier: float = 5.0

    # Only for encoder_type="transformer" (the Phase 8 variant)
    transformer_encoder_dim: int = 384
    transformer_num_blocks: int = 2
    transformer_num_heads: int = 8
    transformer_ffn_expansion: int = 4
    transformer_dropout: float = 0.1

    # Cross-attention injection
    num_cognitive_tokens: int = 32  # = base's num_layers for the selective convention
    ca_bottleneck_dim: int = 256
    ca_num_heads: int = 4
    ca_dropout: float = 0.1
    ca_gate_init: float = 0.3
    ca_gate_lr_multiplier: float = 5.0
    token_preference_init: float = 0.0
    token_preference_lr_multiplier: float = 5.0

    # Phase 2 self-correction target (optional)
    correction_ratio: float = 0.5
    enable_self_correction: bool = True

    # Tokenization (Phase 5 finding: concat_last_mean = +1pp sel_acc)
    pool: Literal["last", "mean", "concat_last_mean"] = "last"
