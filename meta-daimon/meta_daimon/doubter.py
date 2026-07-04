"""Doubter — the original introspection pipeline from the paper
`meta-attention-is-all-you-need`. Cognitive-token generator + 32 BottleneckCrossAttention
+ ReflexionBuffer + two-pass forward with a frozen base.

Calibrated refusal + optional self-correction through a trainable wrapper (~2.3% of the
8B base). On MMLU Phase 2 Selective: sel_acc 89.1% @ 37% coverage (base 64.6%), total_recovery 90.7%.

Architecture:

    Pass 1 (read):
      base.forward(prompt)         → ActivationCollector captures hidden_states
                                     from all target_layers, last token
                                   → encoder(activation_list)
                                   → cognitive_tokens [B, n, hidden]
                                   → ReflexionBuffer.fill(cognitive_tokens)

    Pass 2 (write):
      ActivationCollector.freeze() — do not overwrite the buffer with generated tokens
      base.generate(prompt, ...)    — on each layer the CA hook reads the buffer
                                      h_out = h_residual + tanh(gate) · CA(h, cog_tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from meta_core.buffer import ReflexionBuffer
from meta_daimon.config import DoubterConfig
from meta_core.cross_attention import BottleneckCrossAttention
from meta_core.encoders.base import Encoder
from meta_core.encoders.multi_token import MultiTokenEncoder
from meta_core.encoders.selective import SelectiveEncoder
from meta_core.encoders.transformer import TransformerEncoder
from meta_core.modifiers.base import Modifier

if TYPE_CHECKING:
    from meta_core.pipeline import MetaSpiderPipeline

__all__ = ["Doubter"]


def _build_encoder(
    config: DoubterConfig,
    hidden_dim: int,
    num_layers: int,
    num_cognitive_tokens: int,
) -> Encoder:
    """Construct the encoder from `config.encoder_type`."""
    if config.encoder_type == "selective":
        return SelectiveEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bottleneck_dim=config.encoder_bottleneck,
            gate_init=config.encoder_gate_init,
        )
    if config.encoder_type == "transformer":
        return TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_dim=config.transformer_encoder_dim,
            num_blocks=config.transformer_num_blocks,
            num_heads=config.transformer_num_heads,
            ffn_expansion=config.transformer_ffn_expansion,
            dropout=config.transformer_dropout,
        )
    if config.encoder_type == "multi_token":
        return MultiTokenEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_cognitive_tokens=num_cognitive_tokens,
            bottleneck_dim=config.encoder_bottleneck,
            num_heads=8,
            dropout=0.1,
        )
    raise ValueError(f"Unknown encoder_type: {config.encoder_type!r}")


class Doubter(Modifier):
    """Cognitive-token-based calibrated introspector.

    Attributes:
        config: DoubterConfig.
        encoder: SelectiveEncoder or TransformerEncoder (cognitive tokens generator).
        ca_modules: nn.ModuleDict {str(layer_idx): BottleneckCrossAttention}.
        buffer: ReflexionBuffer (holds the cog tokens between Pass 1 and Pass 2).
    """

    name = "doubter"

    def __init__(self, config: DoubterConfig):
        super().__init__()
        self.config = config
        self.encoder: Optional[Encoder] = None
        self.ca_modules: nn.ModuleDict = nn.ModuleDict()
        self.buffer: Optional[ReflexionBuffer] = None
        self._ca_hook_handles: list = []
        self._cross_attn_layers: list[int] = []

    # ============================================================
    # Lifecycle: on_attach sets up encoder + CA + buffer + hooks
    # ============================================================

    def on_attach(self, pipeline: "MetaSpiderPipeline") -> None:
        """On attach to the pipeline, initializes the wrapper and installs the CA hooks.

        1. Build the encoder with auto-detected hidden_dim / num_layers
        2. Build one BottleneckCrossAttention per cross_attn_layer
        3. Initialize the ReflexionBuffer
        4. Register the CA forward hooks on base.model.layers[i]
        """
        super().on_attach(pipeline)

        if pipeline.config.hidden_dim is None or pipeline.config.num_layers is None:
            raise RuntimeError(
                "MetaSpiderConfig.resolve_defaults() must be called before attach. "
                "Usually MetaSpiderPipeline.from_pretrained() does this."
            )

        hidden_dim = pipeline.config.hidden_dim
        target_layers = list(pipeline.config.target_layers or range(pipeline.config.num_layers))
        n_act_layers = len(target_layers)  # actual number of target layers (for encoder input)
        cross_attn_layers = list(pipeline.config.cross_attn_layers or range(pipeline.config.num_layers))
        self._cross_attn_layers = cross_attn_layers
        self._target_layers = target_layers  # remembered for the checkpoint (see save_checkpoint)

        # num_cognitive_tokens: explicit override via DoubterConfig (for MultiTokenEncoder),
        # otherwise = number of target_layers (Selective/Transformer convention 1 cog per layer).
        if self.config.encoder_type == "multi_token":
            num_cog = self.config.num_cognitive_tokens
        else:
            num_cog = n_act_layers

        # Re-attach (detach → attach, e.g. in BaselineComparison): the wrapper is already
        # built — do NOT rebuild it, otherwise the trained weights get clobbered by random ones.
        # Only restore the hooks and ensure device/dtype.
        if self.encoder is not None and len(self.ca_modules) > 0:
            self._install_ca_hooks(pipeline)
            self._load_pending_state()
            self.set_inference_mode()
            return

        # 1. Encoder
        self.encoder = _build_encoder(
            self.config, hidden_dim, n_act_layers, num_cog,
        )

        # 2. CA modules — one per cross_attn_layer
        self.ca_modules = nn.ModuleDict()
        for layer_idx in cross_attn_layers:
            ca = BottleneckCrossAttention(
                hidden_dim=hidden_dim,
                bottleneck_dim=self.config.ca_bottleneck_dim,
                num_heads=self.config.ca_num_heads,
                num_cognitive_tokens=num_cog,
                dropout=self.config.ca_dropout,
                gate_init=self.config.ca_gate_init,
                token_preference_init=self.config.token_preference_init,
            )
            self.ca_modules[str(layer_idx)] = ca

        # 3. Buffer (device — from the base model)
        device = self._infer_device(pipeline.model)
        self.buffer = ReflexionBuffer(
            num_slots=num_cog,
            hidden_dim=hidden_dim,
            device=str(device),
        )

        # Move the encoder only to device (NOT dtype) — it stays in fp32 so it can
        # accept float activations (per the Phase 1-5 convention in reflexion_model_selective).
        # The CA modules are in the model dtype (for the residual add to bf16 hidden states).
        # Exception — an fp16 base: the CA stay fp32 (AdamW over fp16 params is
        # unstable), and the hook casts the hidden at the boundaries.
        dtype = self._infer_dtype(pipeline.model)
        if isinstance(self.encoder, nn.Module):
            self.encoder = self.encoder.to(device=device)
        if dtype == torch.float16:
            self.ca_modules = self.ca_modules.to(device=device)
        else:
            self.ca_modules = self.ca_modules.to(device=device, dtype=dtype)

        # 4. Install CA forward hooks
        self._install_ca_hooks(pipeline)

        # 5. If it came from from_checkpoint — load the pending state_dicts
        self._load_pending_state()

        # 6. Inference-safe default: eval mode (dropout OFF). nn.Module defaults to
        # training=True, so without this every generate/eval after `from_checkpoint +
        # attach` ran CA attn_dropout stochastically. The Trainer flips .train()/.eval()
        # itself per epoch, so training is unaffected.
        self.set_inference_mode()

    @staticmethod
    def _infer_device(model: Any) -> torch.device:
        """Find the device of the model's first parameter."""
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    @staticmethod
    def _infer_dtype(model: Any) -> torch.dtype:
        """Find the model's compute dtype: the first floating parameter.

        On a quantized base (bnb int8/4bit) the Linear weights are stored in int types —
        we skip them; embeddings/layernorms stay in the compute dtype.
        """
        for p in model.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
        return torch.float32

    def _get_layers(self, model: Any) -> Any:
        """Find the list of transformer layers, family-agnostically (see model_utils)."""
        from meta_core.model_utils import find_decoder_layers
        return find_decoder_layers(model)

    def _install_ca_hooks(self, pipeline: "MetaSpiderPipeline") -> None:
        """Register the forward hooks: CA is applied on each cross_attn_layer."""
        # Guard against duplication: a repeated attach without detach removes the old hooks
        for h in self._ca_hook_handles:
            h.remove()
        self._ca_hook_handles.clear()

        layers = self._get_layers(pipeline.model)
        buffer = self.buffer
        ca_modules = self.ca_modules

        for layer_idx in self._cross_attn_layers:
            ca_module = ca_modules[str(layer_idx)]

            def make_hook(ca, buf):
                def hook(module, inputs, output):
                    if buf is None or not buf.is_filled:
                        return output
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    batch_size = hidden_states.shape[0]
                    # Work in the CA module's dtype: with a bf16/fp32 base it's its own dtype
                    # (casts are no-ops); with an fp16 base the CA is fp32 — cast at the boundaries.
                    ca_dtype = next(ca.parameters()).dtype
                    cog_tokens = buf.expand_batch(batch_size).to(dtype=ca_dtype)
                    modified = ca(hidden_states.to(ca_dtype), cog_tokens)
                    modified = modified.to(hidden_states.dtype)
                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    return modified

                return hook

            handle = layers[layer_idx].register_forward_hook(make_hook(ca_module, buffer))
            self._ca_hook_handles.append(handle)

    def set_inference_mode(self) -> None:
        """Put the wrapper into eval mode (dropout/etc. OFF) — deterministic inference.

        Called automatically at the end of `on_attach`. The Trainer switches the modules
        back to .train() at the start of each epoch and to .eval() for validation.
        """
        if isinstance(self.encoder, nn.Module):
            self.encoder.eval()
        self.ca_modules.eval()

    def on_detach(self) -> None:
        """Remove all CA hooks."""
        for h in self._ca_hook_handles:
            h.remove()
        self._ca_hook_handles.clear()
        super().on_detach()

    def on_pre_forward(self) -> None:
        """Before Pass 1: clear the buffer."""
        if self.buffer is not None:
            self.buffer.clear()

    def on_post_forward(self, activations: dict[int, Any]) -> Optional[dict[str, Any]]:
        """After Pass 1: run the activations through the encoder → fill the buffer.

        Args:
            activations: dict[layer_idx → tensor[B, hidden_dim]] from the base's target_layers.

        Returns:
            None — Doubter emits no control signal; the CA hooks activate automatically
            on Pass 2 as soon as buffer.is_filled.
        """
        if self.encoder is None or self.buffer is None:
            raise RuntimeError("Doubter is not attached to a pipeline (call on_attach).")

        # Sort by layer index and convert to float (for the encoder)
        target_layers = sorted(activations.keys())
        activation_list = [activations[i].float() for i in target_layers]

        cognitive_tokens = self.encoder(activation_list)
        self.buffer.fill(cognitive_tokens)
        return None

    # ============================================================
    # Checkpoint I/O
    # ============================================================

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[DoubterConfig] = None,
        **kwargs: Any,
    ) -> "Doubter":
        """Load the trained encoder + ca_modules from a checkpoint.

        Checkpoint format (framework v1):
            {
                "format_version": "1.0",
                "config": {...},                    # DoubterConfig dict
                "encoder_state": state_dict,
                "ca_state": {str(layer_idx): state_dict, ...},
            }

        Args:
            path: path to the .pt file.
            config: override the config from the checkpoint (optional).

        Note: encoder and ca_modules are created in the pipeline's `on_attach()` — here we only
        remember the states for later loading in `on_attach`.
        """
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no Doubter checkpoint: {path}. Train one first "
                f"(`metaloom train --run-dir <dir>`) or give the correct path to doubter_checkpoint.pt.")
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        if "format_version" not in ckpt:
            raise ValueError(
                "Checkpoint without 'format_version' — possibly the legacy phase1/phase2 format. "
                "Use `archive/src/phase2_selective_llama8b/build_utils.py:load_selective_checkpoint` "
                "to convert."
            )

        if config is None:
            cfg_dict = ckpt.get("config", {})
            config = DoubterConfig(**cfg_dict)

        doubter = cls(config)
        doubter._pending_encoder_state = ckpt.get("encoder_state")
        doubter._pending_ca_state = ckpt.get("ca_state", {})
        # v1.1: the layer indices the checkpoint was trained on (absent in v1.0 — then the
        # only guard is the strict ca_state key match in _load_pending_state).
        doubter._pending_expected_cross = ckpt.get("cross_attn_layers") or None
        doubter._pending_expected_target = ckpt.get("target_layers") or None
        return doubter

    def save_checkpoint(self, path: str) -> None:
        """Save the encoder + ca_modules state_dict + config."""
        if self.encoder is None:
            raise RuntimeError("Nothing to save — the modifier is not attached to a pipeline.")
        ca_state = {k: m.state_dict() for k, m in self.ca_modules.items()}
        cfg_dict = {
            "encoder_type": self.config.encoder_type,
            "encoder_bottleneck": self.config.encoder_bottleneck,
            "encoder_gate_init": self.config.encoder_gate_init,
            "transformer_encoder_dim": self.config.transformer_encoder_dim,
            "transformer_num_blocks": self.config.transformer_num_blocks,
            "transformer_num_heads": self.config.transformer_num_heads,
            "transformer_ffn_expansion": self.config.transformer_ffn_expansion,
            "transformer_dropout": self.config.transformer_dropout,
            "num_cognitive_tokens": self.config.num_cognitive_tokens,
            "ca_bottleneck_dim": self.config.ca_bottleneck_dim,
            "ca_num_heads": self.config.ca_num_heads,
            "ca_dropout": self.config.ca_dropout,
            "ca_gate_init": self.config.ca_gate_init,
            "token_preference_init": self.config.token_preference_init,
            "pool": self.config.pool,
        }
        torch.save(
            {
                "format_version": "1.1",
                "config": cfg_dict,
                # The layer indices the wrapper was trained on (v1.1). The wrapper is
                # calibrated to THESE layers; loading it onto a pipeline with different
                # target/cross_attn layers is verified at attach (see _load_pending_state).
                "target_layers": list(getattr(self, "_target_layers", []) or []),
                "cross_attn_layers": list(self._cross_attn_layers or []),
                "encoder_state": self.encoder.state_dict()
                if isinstance(self.encoder, nn.Module) else None,
                "ca_state": ca_state,
            },
            path,
        )

    def _load_pending_state(self) -> None:
        """Internal: load the pending state from from_checkpoint after on_attach.

        STRICT: the checkpoint's layer set must match the pipeline's. Previously a mismatch
        silently loaded only the intersecting CA layers (trained encoder + randomly-initialized
        CA on the rest) — a half-broken wrapper with no warning.
        """
        pending_enc = getattr(self, "_pending_encoder_state", None)
        pending_ca = getattr(self, "_pending_ca_state", None)

        # v1.1 checkpoints carry their layer indices — verify against the pipeline config.
        exp_cross = getattr(self, "_pending_expected_cross", None)
        exp_target = getattr(self, "_pending_expected_target", None)
        if exp_cross and list(exp_cross) != list(self._cross_attn_layers):
            raise RuntimeError(
                f"Checkpoint/pipeline mismatch: the checkpoint was trained with "
                f"cross_attn_layers={list(exp_cross)}, but the pipeline is configured with "
                f"{list(self._cross_attn_layers)}. Set the same layers in MetaSpiderConfig "
                f"(usually via the run.json manifest).")
        if exp_target and list(exp_target) != list(getattr(self, "_target_layers", []) or []):
            raise RuntimeError(
                f"Checkpoint/pipeline mismatch: the checkpoint was trained with "
                f"target_layers={list(exp_target)}, but the pipeline is configured with "
                f"{list(getattr(self, '_target_layers', []))}. Set the same layers in "
                f"MetaSpiderConfig (usually via the run.json manifest).")

        if pending_enc is not None and isinstance(self.encoder, nn.Module):
            self.encoder.load_state_dict(pending_enc)
            self._pending_encoder_state = None
        if pending_ca:
            ckpt_keys = set(pending_ca.keys())
            module_keys = set(self.ca_modules.keys())
            if ckpt_keys != module_keys:
                raise RuntimeError(
                    f"Checkpoint/pipeline mismatch: ca_state layers {sorted(ckpt_keys, key=int)} "
                    f"!= pipeline cross_attn layers {sorted(module_keys, key=int)}. Refusing a "
                    f"partial load (it would leave the missing layers randomly initialized).")
            for k, state in pending_ca.items():
                self.ca_modules[k].load_state_dict(state)
            self._pending_ca_state = None

    # ============================================================
    # Trainable parameters helper
    # ============================================================

    def get_trainable_parameters(self) -> list:
        """List of parameters for the optimizer (encoder + 32 CA modules)."""
        params: list = []
        if isinstance(self.encoder, nn.Module):
            params.extend(self.encoder.parameters())
        params.extend(self.ca_modules.parameters())
        return params

    # ============================================================
    # Diagnostics
    # ============================================================

    def set_gain(self, gain: float) -> None:
        """v0.2 — set the static injection gain (the uncertainty potentiometer) on EVERY CA layer.

        A single runtime knob over the whole Doubter: gain=1.0 is the trained baseline, >1 amplifies
        doubt (→ more refusal), <1 attenuates, <0 inverts toward confidence. Composes with the trained
        per-layer gates and with AGC. Validated monotonic range ~[0, 1.5] before runaway
        (docs/results/gemma-4-12b/gain-potentiometer.md). Set at inference; not trained.
        """
        for m in self.ca_modules.values():
            m.set_gain(gain)

    def get_gain(self) -> float:
        """Current static injection gain (assumes uniform across layers; returns the first)."""
        for m in self.ca_modules.values():
            return float(m.gain)
        return 1.0

    def get_ca_gate_map(self) -> dict[int, float]:
        """tanh(gate) on each CA layer — the 'cognitive injection map'.

        Stable picture after training (per the article):
            L0-L5    : 0.01-0.04 (tokenization, no meta needed)
            L19-L24  : 0.07-0.09 (reasoning, starts to listen)
            L29      : 0.12-0.19 (LEADER, the decision point)
            L30-L31  : 0.07-0.08 (final processing)
        """
        return {int(k): m.get_gate_value() for k, m in self.ca_modules.items()}

    def get_encoder_gate_map(self) -> dict[int, float]:
        """tanh(gate) on each of the encoder's per-layer projectors — Selective only."""
        if not isinstance(self.encoder, SelectiveEncoder):
            raise RuntimeError(
                "get_encoder_gate_map is available only for SelectiveEncoder."
            )
        return {
            int(k.split("_")[1]): v
            for k, v in self.encoder.get_gate_values().items()
        }

    def get_token_preference_matrix(self) -> dict[int, list[float]]:
        """{layer_idx: [num_cog_tokens]} — the trainable topology
        'which LLM layer looks at which source'.
        """
        return {int(k): m.get_token_preference_values() for k, m in self.ca_modules.items()}
