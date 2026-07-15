"""GoalAnchor — the goal-drift Watchdog LEG: a behavior voice against goal drift.

NOT `daimon_voices.watchdog.Watchdog` (that one is a confidence SENSOR in the Doubter stack whose
probe gates pointwise injection / external actions). GoalAnchor is a full injection voice
with a DIFFERENT lifecycle:

    anchor = encode(GOAL text)          # once — a static anchor in the buffer
    generate(prompt) …                  # trigger-gated CA re-injects the anchor
    generate(another prompt) …          # the SAME anchor persists across generations

vs the Doubter, whose buffer is rebuilt from each prompt's own activations. Hence:
  - `on_pre_forward` does NOT clear the buffer (only resets the trigger);
  - `on_post_forward` is a no-op (the anchor does not come from the prompt);
  - the CA hooks are gated by a TRIGGER — a fixed-K period or a learnable MLP probe on
    `trigger_decision_layer` with cooldown (fire → inject this step, else no-op).

Port of `archive/src/watchdog_v2_llama1b` (v2.1). Validated on Llama-3.2-1B: the latent anchor
matches gold text reminders on the forbid family WITHOUT the goal in the prompt and at ~30%
fewer interventions; within-family generalization 3/3 WIN. Known limits (measured): transfer
to an unseen drift FAMILY fails (the encoder is family-specific); the prefix family was
capability-bound on 1B. See archive/docs/results/watchdog_v2/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from meta_attention.buffer import ReflexionBuffer
from daimon_voices.config import GoalAnchorConfig
from meta_attention.injection import BottleneckCrossAttention
from meta_attention.encoders.transformer import TransformerEncoder
from daimon_voices.voice import Voice

if TYPE_CHECKING:
    from meta_attention.pipeline import MetaAttentionPipeline

__all__ = ["GoalAnchor", "BinaryTrigger", "LearnableTrigger"]


# ============================================================
# Triggers — WHEN to re-inject the anchor
# ============================================================


class BinaryTrigger(nn.Module):
    """Fixed-period trigger: fire every K steps. K<=0 — never (off), K=1 — every step."""

    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.reset()

    def reset(self) -> None:
        self.step_idx = 0

    @torch.no_grad()
    def decide(self, hidden_state: Optional[torch.Tensor] = None) -> bool:
        if self.K <= 0:
            return False
        fire = (self.step_idx % self.K == 0)
        self.step_idx += 1
        return fire


class LearnableTrigger(nn.Module):
    """MLP(hidden@decision_layer) → sigmoid → fire, with cooldown constraints.

    steps_since_last < K_min → no inject (cooldown); ≥ K_max → force inject;
    in between — the MLP decides (threshold 0.5).
    """

    def __init__(self, hidden_dim: int, mlp_hidden: int = 64,
                 K_min: int = 50, K_max: int = 500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K_min = K_min
        self.K_max = K_max
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(-0.4)   # start sparse (~0.4 P(fire) after warmup)
        self.reset()

    def reset(self) -> None:
        self.steps_since_last = 0

    def forward_logit(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Differentiable forward for supervised BCE training. [B,H] or [H] → [B] / scalar."""
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        return self.net(hidden_state.float()).squeeze(-1)

    @torch.no_grad()
    def decide(self, hidden_state: torch.Tensor) -> bool:
        if self.steps_since_last < self.K_min:
            self.steps_since_last += 1
            return False
        if self.steps_since_last >= self.K_max:
            self.steps_since_last = 0
            return True
        logit = self.forward_logit(hidden_state)
        p = torch.sigmoid(logit.reshape(-1)[0]).item()
        fire = p > 0.5
        self.steps_since_last = 0 if fire else self.steps_since_last + 1
        return fire


def _build_trigger(config: GoalAnchorConfig, hidden_dim: int) -> Optional[nn.Module]:
    if config.trigger in ("always", "agent_step"):
        return None       # no per-token trigger; agent_step arms externally (step hooks)
    if config.trigger == "fixed":
        return BinaryTrigger(config.trigger_k)
    if config.trigger == "learnable":
        return LearnableTrigger(hidden_dim, mlp_hidden=config.trigger_mlp_hidden,
                                K_min=config.trigger_k_min, K_max=config.trigger_k_max)
    raise ValueError(f"Unknown trigger: {config.trigger!r}")


# ============================================================
# GoalAnchor voice
# ============================================================


class GoalAnchor(Voice):
    """Static latent goal anchor + trigger-gated CA re-injection against goal drift.

    Usage:
        anchor = GoalAnchor(GoalAnchorConfig(trigger="fixed", trigger_k=100))
        pipeline.attach(anchor)
        anchor.set_anchor("Never use the word 'the'.")   # once
        pipeline.generate("Write an essay about space")   # anchor persists, gated re-inject
    """

    name = "goal_anchor"

    def __init__(self, config: Optional[GoalAnchorConfig] = None):
        super().__init__()
        self.config = config or GoalAnchorConfig()
        self.encoder: Optional[TransformerEncoder] = None
        self.ca_modules: nn.ModuleDict = nn.ModuleDict()
        self.buffer: Optional[ReflexionBuffer] = None
        self.trigger: Optional[nn.Module] = None
        self._ca_hook_handles: list = []
        self._cross_attn_layers: list[int] = []
        self._target_layers: list[int] = []
        # agent_step mode: injection is OFF by default and armed only for the agent-loop
        # decision window (on_step_start/on_step_end from MetaAgent step_hooks).
        self._agent_step_mode: bool = (self.config.trigger == "agent_step")
        self._step_fire: bool = not self._agent_step_mode

    # ---------------- lifecycle ----------------

    def on_attach(self, pipeline: "MetaAttentionPipeline") -> None:
        super().on_attach(pipeline)
        if pipeline.config.hidden_dim is None or pipeline.config.num_layers is None:
            raise RuntimeError("MetaAttentionConfig.resolve_defaults() must run before attach "
                               "(MetaAttentionPipeline.from_pretrained does this).")

        hidden_dim = pipeline.config.hidden_dim
        self._target_layers = list(pipeline.config.target_layers or range(pipeline.config.num_layers))
        self._cross_attn_layers = list(pipeline.config.cross_attn_layers or range(pipeline.config.num_layers))
        n_act = len(self._target_layers)
        num_cog = n_act                    # transformer convention: 1 cog token per read layer

        # Re-attach (detach→attach): don't clobber trained weights, just restore hooks.
        if self.encoder is not None and len(self.ca_modules) > 0:
            self._install_hooks(pipeline)
            self._load_pending_state()
            self.set_inference_mode()
            return

        cfg = self.config
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim, num_layers=n_act,
            encoder_dim=cfg.encoder_dim, num_blocks=cfg.encoder_num_blocks,
            num_heads=cfg.encoder_num_heads, ffn_expansion=cfg.encoder_ffn_expansion,
            dropout=cfg.encoder_dropout,
            use_per_layer_projectors=cfg.use_per_layer_projectors,
            use_layer_pos_embeddings=cfg.use_layer_pos_embeddings,
        )
        self.ca_modules = nn.ModuleDict({
            str(li): BottleneckCrossAttention(
                hidden_dim=hidden_dim, bottleneck_dim=cfg.ca_bottleneck_dim,
                num_heads=cfg.ca_num_heads, num_cognitive_tokens=num_cog,
                dropout=cfg.ca_dropout, gate_init=cfg.ca_gate_init,
                token_preference_init=cfg.token_preference_init,
            )
            for li in self._cross_attn_layers
        })
        device = self._infer_device(pipeline.model)
        dtype = self._infer_dtype(pipeline.model)
        self.encoder = self.encoder.to(device=device)          # fp32 encoder (float activations)
        if dtype == torch.float16:
            self.ca_modules = self.ca_modules.to(device=device)  # keep CA fp32 on fp16 bases
        else:
            self.ca_modules = self.ca_modules.to(device=device, dtype=dtype)
        self.buffer = ReflexionBuffer(num_slots=num_cog, hidden_dim=hidden_dim, device=str(device))
        self.trigger = _build_trigger(cfg, hidden_dim)
        if isinstance(self.trigger, nn.Module):
            self.trigger = self.trigger.to(device)

        self._install_hooks(pipeline)
        self._load_pending_state()
        self.set_inference_mode()

    def set_inference_mode(self) -> None:
        """Eval mode (dropout OFF) — deterministic inference; the Trainer flips it itself."""
        if isinstance(self.encoder, nn.Module):
            self.encoder.eval()
        self.ca_modules.eval()
        if isinstance(self.trigger, nn.Module):
            self.trigger.eval()

    def _install_hooks(self, pipeline: "MetaAttentionPipeline") -> None:
        for h in self._ca_hook_handles:
            h.remove()
        self._ca_hook_handles.clear()
        from meta_attention.model_utils import find_decoder_layers
        layers = find_decoder_layers(pipeline.model)

        # Decision hook: only needed when a TRIGGER is present (fixed/learnable) — it reads the
        # last hidden at trigger_decision_layer to advance/query the trigger. always / agent_step
        # (trigger is None) don't use it: `always` keeps _step_fire=True (set in on_pre_forward),
        # `agent_step` is armed by the MetaAgent step hooks. So the decision-layer range check
        # applies only to triggered modes — an always-anchor works on a model of any depth.
        if self.trigger is not None:
            dec = self.config.trigger_decision_layer
            if dec >= len(layers):
                raise ValueError(f"trigger_decision_layer {dec} out of range "
                                 f"(the model has {len(layers)} layers)")

            def decision_hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                self._step_fire = bool(self.trigger.decide(hs[:, -1, :].detach()[0]))
            self._ca_hook_handles.append(layers[dec].register_forward_hook(decision_hook))

        # CA hooks: inject only when the anchor is set AND the trigger fired this step.
        buffer = self.buffer
        for li in self._cross_attn_layers:
            ca_module = self.ca_modules[str(li)]

            def make_hook(ca, buf):
                def hook(module, inputs, output):
                    if buf is None or not buf.is_filled or not self._step_fire:
                        return output
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    ca_dtype = next(ca.parameters()).dtype
                    cog = buf.expand_batch(hidden_states.shape[0]).to(dtype=ca_dtype)
                    modified = ca(hidden_states.to(ca_dtype), cog).to(hidden_states.dtype)
                    return (modified,) + output[1:] if isinstance(output, tuple) else modified
                return hook

            self._ca_hook_handles.append(layers[li].register_forward_hook(make_hook(ca_module, buffer)))

    def on_detach(self) -> None:
        for h in self._ca_hook_handles:
            h.remove()
        self._ca_hook_handles.clear()
        super().on_detach()

    def on_pre_forward(self) -> None:
        """UNLIKE the Doubter: the anchor PERSISTS across prompts — only the trigger resets."""
        if self.trigger is not None and hasattr(self.trigger, "reset"):
            self.trigger.reset()
        self._step_fire = not self._agent_step_mode   # agent_step: stays off until armed

    # ---------------- agentic decision-window arming (MetaAgent step_hooks) ----------------

    def on_step_start(self, step: int = 0, session: Any = None) -> None:
        """Arm the injection for the agent-loop DECISION window (called by MetaAgent before
        each policy.act). Decide-then-detach: the anchor steers the short decision emission
        (tool call / final line), not the long tool-output digestion."""
        self._step_fire = True

    def on_step_end(self, step: int = 0, session: Any = None) -> None:
        """Disarm after the decision is emitted (called by MetaAgent after policy.act)."""
        if self._agent_step_mode:
            self._step_fire = False

    def on_post_forward(self, activations: dict[int, Any]) -> Optional[dict[str, Any]]:
        """No-op: the anchor comes from the GOAL text (set_anchor), not from the prompt."""
        return None

    # ---------------- anchor ----------------

    def build_anchor(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode the goal into the static anchor: goal ids → frozen base → encoder → buffer."""
        if self._pipeline is None:
            raise RuntimeError("GoalAnchor is not attached to a pipeline.")
        pipe = self._pipeline
        if pipe.collector is None:
            raise RuntimeError("Pipeline without an ActivationCollector.")
        self.buffer.clear()                # our CA hooks no-op during the goal pass
        pipe.collector.clear()
        pipe.collector.unfreeze()
        try:
            with torch.no_grad():
                pipe.model(input_ids=input_ids, attention_mask=attention_mask)
            snap = pipe.collector.get_snapshot()
        finally:
            pipe.collector.freeze()
        acts = [snap[i].float() for i in sorted(self._target_layers) if i in snap]
        cog = self.encoder(acts)
        self.buffer.fill(cog)
        return cog

    def set_anchor(self, goal_text: str) -> torch.Tensor:
        """Convenience: goal text → tokenizer → build_anchor."""
        tok = self._pipeline.tokenizer
        enc = tok(goal_text, return_tensors="pt")
        device = next(self._pipeline.model.parameters()).device
        mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None
        return self.build_anchor(enc.input_ids.to(device), mask)

    def clear_anchor(self) -> None:
        if self.buffer is not None:
            self.buffer.clear()

    def set_trigger(self, trigger: Optional[nn.Module]) -> None:
        """Swap the trigger at runtime. None = always-on."""
        self.trigger = trigger
        if trigger is not None and hasattr(trigger, "reset"):
            trigger.reset()
        self._step_fire = True

    def set_gain(self, gain: float) -> None:
        """The same uncertainty-knob mechanism as the Doubter — one dial over all CA layers."""
        for m in self.ca_modules.values():
            m.set_gain(gain)

    # ---------------- checkpoint I/O (mirrors the Doubter v1.1 contract) ----------------

    @classmethod
    def from_checkpoint(cls, path: str, config: Optional[GoalAnchorConfig] = None,
                        **kwargs: Any) -> "GoalAnchor":
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(f"no GoalAnchor checkpoint: {path}")
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        if ckpt.get("kind") != "goal_anchor":
            raise ValueError("not a GoalAnchor checkpoint (kind != 'goal_anchor'); "
                             "Doubter checkpoints load via Doubter.from_checkpoint.")
        if config is None:
            config = GoalAnchorConfig(**ckpt.get("config", {}))
        anchor = cls(config)
        anchor._pending_encoder_state = ckpt.get("encoder_state")
        anchor._pending_ca_state = ckpt.get("ca_state", {})
        anchor._pending_trigger_state = ckpt.get("trigger_state")
        anchor._pending_expected_cross = ckpt.get("cross_attn_layers") or None
        anchor._pending_expected_target = ckpt.get("target_layers") or None
        return anchor

    def save_checkpoint(self, path: str) -> None:
        if self.encoder is None:
            raise RuntimeError("Nothing to save — the voice is not attached.")
        cfg = self.config
        torch.save({
            "format_version": "1.1",
            "kind": "goal_anchor",
            "config": {f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
            "target_layers": list(self._target_layers),
            "cross_attn_layers": list(self._cross_attn_layers),
            "encoder_state": self.encoder.state_dict(),
            "ca_state": {k: m.state_dict() for k, m in self.ca_modules.items()},
            "trigger_state": (self.trigger.state_dict()
                              if isinstance(self.trigger, LearnableTrigger) else None),
        }, path)

    def _load_pending_state(self) -> None:
        exp_cross = getattr(self, "_pending_expected_cross", None)
        exp_target = getattr(self, "_pending_expected_target", None)
        if exp_cross and list(exp_cross) != list(self._cross_attn_layers):
            raise RuntimeError(f"Checkpoint/pipeline mismatch: cross_attn_layers "
                               f"{list(exp_cross)} != {list(self._cross_attn_layers)}.")
        if exp_target and list(exp_target) != list(self._target_layers):
            raise RuntimeError(f"Checkpoint/pipeline mismatch: target_layers "
                               f"{list(exp_target)} != {list(self._target_layers)}.")
        pend_enc = getattr(self, "_pending_encoder_state", None)
        if pend_enc is not None and isinstance(self.encoder, nn.Module):
            self.encoder.load_state_dict(pend_enc)
            self._pending_encoder_state = None
        pend_ca = getattr(self, "_pending_ca_state", None)
        if pend_ca:
            if set(pend_ca.keys()) != set(self.ca_modules.keys()):
                raise RuntimeError(f"Checkpoint/pipeline mismatch: ca_state layers "
                                   f"{sorted(pend_ca, key=int)} != "
                                   f"{sorted(self.ca_modules, key=int)}. Refusing a partial load.")
            for k, st in pend_ca.items():
                self.ca_modules[k].load_state_dict(st)
            self._pending_ca_state = None
        pend_trig = getattr(self, "_pending_trigger_state", None)
        if pend_trig is not None and isinstance(self.trigger, LearnableTrigger):
            self.trigger.load_state_dict(pend_trig)
            self._pending_trigger_state = None

    # ---------------- misc ----------------

    def get_trainable_parameters(self) -> list:
        params = list(self.encoder.parameters()) if isinstance(self.encoder, nn.Module) else []
        params.extend(self.ca_modules.parameters())
        if isinstance(self.trigger, LearnableTrigger):
            params.extend(self.trigger.parameters())
        return params

    def get_ca_gate_map(self) -> dict[int, float]:
        return {int(k): m.get_gate_value() for k, m in self.ca_modules.items()}

    @staticmethod
    def _infer_device(model: Any) -> torch.device:
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    @staticmethod
    def _infer_dtype(model: Any) -> torch.dtype:
        for p in model.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
        return torch.float32
