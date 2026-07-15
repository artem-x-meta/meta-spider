"""Daimon voice configs — per-voice dataclasses on top of the meta-attention mechanism.

`MetaAttentionConfig` (the mechanism: base model, layers, dtype) stays in `meta_attention.config`;
here live the VOICE configs: what encoder a voice uses, its cross-attention shape, its
trigger discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from meta_attention.config import EncoderType

__all__ = ["DoubterConfig", "GoalAnchorConfig", "ChronographerConfig",
           "ChronoAnchorConfig"]


@dataclass
class DoubterConfig:
    """Config for the Doubter voice — cognitive-token pipeline.

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




@dataclass
class GoalAnchorConfig:
    """Config for the GoalAnchor voice — the goal-drift Watchdog LEG (behavior voice).

    Not to be confused with `daimon_voices.watchdog.Watchdog` (the read-only confidence sensor
    inside the Doubter stack). GoalAnchor is a full injection voice: the GOAL text is
    encoded once into cognitive tokens (a static anchor), and during generation a trigger-gated
    cross-attention re-injects the anchor to keep long generations from drifting off-goal.

    Port of `archive/src/watchdog_v2_llama1b` (v2.1, validated on Llama-3.2-1B: the latent
    anchor MATCHES text reminders on the forbid family WITHOUT the goal in the prompt, at
    ~30% fewer interventions; within-family generalization 3/3 WIN).
    """

    # Encoder (the Phase-8 transformer — meta_attention.encoders.TransformerEncoder)
    encoder_dim: int = 384
    encoder_num_blocks: int = 2
    encoder_num_heads: int = 8
    encoder_ffn_expansion: int = 4
    encoder_dropout: float = 0.1
    use_per_layer_projectors: bool = True
    use_layer_pos_embeddings: bool = True

    # Cross-attention injection (same BottleneckCrossAttention as the Doubter)
    ca_bottleneck_dim: int = 256
    ca_num_heads: int = 4
    ca_dropout: float = 0.1
    ca_gate_init: float = 0.3
    token_preference_init: float = 0.0

    # Trigger: WHEN to re-inject the anchor during generation.
    #   "always"    — inject on every step (v1 / Step-4a training baseline)
    #   "fixed"     — every trigger_k tokens (BinaryTrigger; optimal_K is task-dependent,
    #                 e.g. 100 for forbid_the on 1B)
    #   "learnable" — MLP probe on trigger_decision_layer + cooldown (LearnableTrigger)
    #   "agent_step" — AGENTIC mode (decide-then-detach): injection is OFF by default and is
    #                 armed only for the agent-loop DECISION window (MetaAgent step_hooks call
    #                 on_step_start/on_step_end). Matches the measured finding that injection
    #                 is a decision modulator: constant injection corrupts long generation,
    #                 point injection at the decision window is +1/0-losses (ODEX).
    trigger: Literal["always", "fixed", "learnable", "agent_step"] = "fixed"
    trigger_k: int = 100
    trigger_decision_layer: int = 9
    trigger_k_min: int = 50
    trigger_k_max: int = 500
    trigger_mlp_hidden: int = 64


@dataclass
class ChronographerConfig:
    """Config for the Chronographer voice — episodic memory over a rolling bank.

    Third lifecycle type in the daimon family: the Doubter rebuilds its buffer from each
    prompt, the GoalAnchor encodes the goal once and holds it, the Chronographer
    ACCUMULATES — each recorded episode is compressed into K memory tokens and appended
    to a persistent bank that memory cross-attention reads on every forward.

    Defaults — the validated Llama-3.2-1B `content_pipeline.pt` shape (clean eval Δ +0.083,
    `docs/results/llama-1b/chronographer-clean-eval.md`). The wrapper is model-specific:
    reads/injects on every decoder layer of a 16-layer base.
    """

    # Read taps (episode recording) and injection points. For the 1B checkpoint both
    # are all 16 layers; resolved against the actual base at attach.
    num_layers: int = 16
    activation_layers: tuple = tuple(range(16))
    cross_attn_layers: tuple = tuple(range(16))

    # Episode snapshot = concat(last_token, mean) per layer → 2 × hidden_dim.
    # 0 = derive from the base's hidden_size at attach (2 * hidden).
    projector_input_dim: int = 0

    # Encoder (per-layer projectors over the episode snapshot)
    encoder_bottleneck: int = 384
    encoder_gate_init: float = 0.3

    # Compressor (episode → K memory tokens in the base's hidden space)
    K_memory_tokens: int = 4
    compressor_n_heads: int = 4

    # Memory cross-attention (per injected layer)
    memory_ca_bottleneck: int = 256
    memory_ca_num_heads: int = 4
    memory_ca_dropout: float = 0.1
    memory_ca_gate_init: float = 0.3
    use_adaptive_gates: bool = False
    adaptive_gate_hidden: int = 64
    adaptive_use_bank_size: bool = True

    # Rolling bank: novelty write-gate + eviction discipline
    bank_max_episodes: int = 50
    novelty_threshold: float = 0.999
    eviction_mode: Literal["consolidate", "centroid", "fifo"] = "consolidate"


@dataclass
class ChronoAnchorConfig(ChronographerConfig):
    """Config for the ChronoAnchor — goal-conditioned episodic memory (ONE organ).

    Inherits the memory-organ shape from ChronographerConfig and adds the goal channel.
    Defaults = the validated one-organ recipe (Llama-1B: 1.000 vs 0.500 blind ceiling;
    `docs/results/llama-1b/tandem-one-organ-scratch.md`).
    """

    # How the goal is read off the frozen base:
    #   "last"   — last token per read layer  → [n_layers, hidden] (validated on 1B)
    #   "tokens" — every goal token           → [n_layers*T, hidden] (compositional goals:
    #              a single summary vector merges the distinguishing clause with the task)
    goal_pool: Literal["last", "tokens"] = "last"

    # Goal-conditioned queries of the compressor (of K_memory_tokens). 0 = plain Chronographer.
    # The rest stay unconditioned — a two-component bank guards against over-conditioning.
    goal_cond_k: int = 2

    # Episode recording (agentic lifecycle: one episode per agent step)
    max_episode_tokens: int = 8192
