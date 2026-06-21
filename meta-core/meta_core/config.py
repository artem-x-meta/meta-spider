"""Meta-Spider Framework configuration.

A single model-agnostic dataclass `MetaSpiderConfig` plus the modifier sub-config
(`DoubterConfig`). Parameter sources:
  - publish/github/src/phase1_selective_llama8b/config_selective.py (Phase 1 Selective)
  - publish/github/src/phase2_selective_llama8b/config_p2.py (Phase 2 Selective)
  - publish/github/src/phase8_transformer_encoder_llama1b/config_p8_1b.py (Phase 8 transformer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

__all__ = ["MetaSpiderConfig", "DoubterConfig"]


EncoderType = Literal["selective", "transformer", "multi_token"]


@dataclass
class MetaSpiderConfig:
    """Top-level pipeline config. Model-agnostic.

    `hidden_dim` and `num_layers` may be `None` — then `MetaSpiderPipeline`
    auto-detects them from the loaded model via `model.config`.

    By default target_layers = all base layers (range(num_layers)) — this is the Phase 1+
    selective convention "1 cognitive token per layer".
    """

    model_name: str
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None

    # Layers: an explicit list of indices OR a preset string:
    #   "all"  — all base layers
    #   "late" — the top third (late layers are the most receptive to injection —
    #            All-CA injection map; backward flows only from the lowest CA layer
    #            upward, so 'late' is noticeably cheaper to train)
    target_layers: Optional[Union[list[int], str]] = None  # None → "all"
    cross_attn_layers: Optional[Union[list[int], str]] = None  # None → target_layers

    device: str = "auto"  # "auto" → cuda if available, else cpu (resolved in from_pretrained)
    dtype: str = "bfloat16"

    # Offload a large frozen base onto a small GPU: {0: "3GiB", "cpu": "14GiB"} →
    # accelerate distributes layers GPU↔RAM (forward works; wrapper training too,
    # but per-step swap is slow). None → the whole model on config.device.
    max_memory: Optional[dict] = None

    # Explicit device_map (overrides auto/max_memory). SLICE-TRAINER+offload: top→GPU
    # (the slice is computed), bottom→cpu (identity patch, not computed). See build_slice_device_map.
    # Also the load path for already-quantized (pre-quant bnb) models without .to().
    device_map: Optional[dict] = None

    # SLICE-TRAINER on 4GB for models with a huge vocabulary (Gemma-4 262K → lm_head ~2GB bf16):
    # bnb by default does NOT quantize lm_head (tied) → we force nf4 on it (2GB→0.5GB) via
    # llm_int8_skip_modules=[]. Decouples the tied embed (the embed is not needed in the slice → on cpu).
    quantize_lm_head: bool = False
    # llm_int8_enable_fp32_cpu_offload — needed when device_map places some of the quantized
    # layers on cpu (offloading the bottom during 8-12B slice training).
    cpu_offload_fp32: bool = False
    # attn_implementation for from_pretrained ("eager"/"sdpa"). Gemma-4 with custom masks
    # (local/global hybrid, global head_dim=512) → "eager" (FA2 is not supported there).
    attn_implementation: Optional[str] = None

    # Compression of the frozen base (QLoRA-style; backward through the quantized weights
    # works — the wrapper lives outside of them). The ladder of degrees:
    #   None   — full precision in `dtype` (1× memory bf16/fp16)
    #   "int8" — bitsandbytes LLM.int8 (~2× smaller)
    #   "nf4"  — 4-bit NormalFloat (~4× smaller) — recommended
    #   "fp4"  — 4-bit FloatPoint (~4× smaller, slightly worse quality than nf4)
    quantization: Optional[Literal["int8", "nf4", "fp4"]] = None
    # Double quantization of the constants for 4-bit (another ~0.4 bytes/parameter)
    double_quant: bool = True

    # Gradient checkpointing of the base: activation memory ↓ several-fold, time ↑ ~30%.
    # Useful for large models / small GPUs.
    gradient_checkpointing: bool = False

    seed: int = 42

    freeze_base_model: bool = True

    def resolve_defaults(self, base_num_layers: int, base_hidden_dim: int) -> None:
        """Fill `hidden_dim` / `num_layers` / `target_layers` / `cross_attn_layers`
        with defaults from the loaded base model.

        Selective convention (Phase 1+): target_layers = all base layers, one
        cognitive token per layer; cross_attn_layers = target_layers.

        Args:
            base_num_layers: `model.config.num_hidden_layers` of the loaded HF model.
            base_hidden_dim: `model.config.hidden_size`.
        """
        if self.hidden_dim is None:
            self.hidden_dim = base_hidden_dim
        elif self.hidden_dim != base_hidden_dim:
            raise ValueError(
                f"config.hidden_dim={self.hidden_dim} does not match the model ({base_hidden_dim})"
            )

        if self.num_layers is None:
            self.num_layers = base_num_layers
        elif self.num_layers > base_num_layers:
            raise ValueError(
                f"config.num_layers={self.num_layers} > base ({base_num_layers})"
            )

        def resolve_preset(value, field_name: str) -> list[int]:
            """'all' / 'late' / list → a concrete list of layers."""
            if isinstance(value, str):
                if value == "all":
                    return list(range(self.num_layers))
                if value == "late":
                    # Top third: for 26 layers → [17..25], for 32 → [21..31]
                    start = (self.num_layers * 2) // 3
                    return list(range(start, self.num_layers))
                raise ValueError(
                    f"{field_name}: unknown preset {value!r}. "
                    "Allowed: 'all', 'late' or a list of indices."
                )
            lst = list(value)
            bad = [i for i in lst if not (0 <= int(i) < self.num_layers)]
            if bad:
                raise ValueError(
                    f"{field_name}: layers {bad} out of range — the model has {self.num_layers} "
                    f"layers (allowed 0..{self.num_layers - 1})."
                )
            return [int(i) for i in lst]

        # Resolve cross_attn BEFORE target (needed for the late_slice preset).
        ca_explicit = self.cross_attn_layers is not None
        if ca_explicit:
            self.cross_attn_layers = resolve_preset(self.cross_attn_layers, "cross_attn_layers")

        if self.target_layers is None:
            self.target_layers = list(range(self.num_layers))
        elif self.target_layers == "late_slice":
            # Slice-trainer: read layers STRICTLY below the lowest CA layer → their activations
            # are captured from the "bottom" (before injection) and cached → the single-pass
            # slice is correct. (The 'late' preset's catch: it reads ABOVE the lowest CA — there
            # the activations depend on injection, and the slice would be incorrect.)
            if not ca_explicit:
                raise ValueError("target_layers='late_slice' requires explicit cross_attn_layers")
            min_ca = min(self.cross_attn_layers)
            start = self.num_layers // 3
            if min_ca <= start + 1:
                raise ValueError(
                    f"late_slice: cross_attn is too low (min={min_ca}, need > {start + 1}); "
                    "the preset is intended for LATE CA layers")
            self.target_layers = list(range(start, min_ca))  # all indices < min_ca
        else:
            self.target_layers = resolve_preset(self.target_layers, "target_layers")

        if not ca_explicit:
            self.cross_attn_layers = list(self.target_layers)

    def slice_cut_layer(self) -> int:
        """The layer whose OUTPUT is cached for the slice-trainer = the input of the lowest CA layer.

        The slice runs only `layers[min_ca:]`, starting from the cached output of `layer[min_ca-1]`.
        Requires all target (read) layers to be ≤ cut — otherwise the encoder would read
        activations INSIDE the slice (which depend on injection) and the single-pass slice would be incorrect."""
        if not isinstance(self.cross_attn_layers, (list, tuple)):
            raise RuntimeError("resolve_defaults was not called — cross_attn_layers is not resolved")
        cut = min(self.cross_attn_layers) - 1
        bad = [t for t in self.target_layers if t > cut]
        if bad:
            raise ValueError(
                f"slice-trainer: read layers {bad} above cut={cut} (lowest CA={cut + 1}). "
                "Use target_layers='late_slice' or explicit read layers ≤ lowest CA−1.")
        return cut


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


