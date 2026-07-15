"""Chronographer — the episodic-memory LEG: a daimon voice that remembers the PAST.

    pipe.attach(chrono)
    chrono.record_episode(input_ids, mask, meta)   # after a turn: episode → K memory tokens → bank
    generate(prompt) ...                            # memory CA injects the WHOLE bank every forward
    chrono.record_episode(...)                      # the bank ACCUMULATES across the session

Third lifecycle type in the family: the Doubter rebuilds its buffer from each prompt, the
GoalAnchor encodes the goal once and holds it, the Chronographer accumulates a rolling bank
(novelty write-gate + consolidate/centroid/fifo eviction). Against context rot on long
sessions: clean eval (train questions excluded) Δ +0.083 overall on Llama-1B
(`docs/results/llama-1b/chronographer-clean-eval.md`).

Unlike the Doubter/GoalAnchor this voice does NOT use the library buffer/CA machinery:
its trained modules (EmotionalEncoder + EpisodeCompressor + per-layer
MemoryBottleneckCrossAttention) predate the framework and are kept verbatim in
`daimon_voices._chrono` so the legacy `content_pipeline.pt` checkpoint loads unchanged.
It registers its own forward hooks (the Voice contract explicitly allows this).
Research-only variants (gate reader, entropy gate, signal capture) stayed in the archive.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn as nn

from daimon_voices.voice import Voice
from meta_attention.model_utils import find_decoder_layers, infer_hidden_dim

from daimon_voices.config import ChronographerConfig
from daimon_voices._chrono import (EmotionalEncoder, EpisodeCompressor,
                                 MemoryBottleneckCrossAttention, PersistentMemoryBank)

__all__ = ["Chronographer"]

CHECKPOINT_FORMAT_VERSION = "1.1"


class Chronographer(Voice):
    """Rolling episodic memory: record_episode() → bank → memory-CA injection each forward."""

    name = "chronographer"

    def __init__(self, config: Optional[ChronographerConfig] = None) -> None:
        super().__init__()
        self.config = config or ChronographerConfig()
        # Modules are built at attach (need the base's hidden size / device / dtype).
        self.encoder: Optional[EmotionalEncoder] = None
        self.compressor: Optional[EpisodeCompressor] = None
        self.memory_ca_layers: Optional[nn.ModuleDict] = None
        self.bank: Optional[PersistentMemoryBank] = None
        self._pending_state: Optional[dict] = None      # from_checkpoint before attach
        self._hook_handles: list = []
        self._collected: dict[int, dict[str, torch.Tensor]] = {}
        self._collecting = False
        self._current_memory: Optional[torch.Tensor] = None
        self._goal_vec: Optional[torch.Tensor] = None    # tandem-v1: goal latent for the compressor
        self._live_tokens: list = []                     # trainable record path (grad-alive mirror)
        self._gain: float = 1.0                          # the daimon fader (runtime scale)
        self._hidden_dim: int = 0
        self._dtype = torch.float32

    # ---------------- lifecycle ----------------

    def on_attach(self, pipeline: Any) -> None:
        super().on_attach(pipeline)
        base = pipeline.model
        cfg = self.config
        self._hidden_dim = infer_hidden_dim(base)   # incl. вложенный text_config (gemma-4)
        self._dtype = next(base.parameters()).dtype
        device = next(base.parameters()).device
        # device_map-раскладка (модель на нескольких GPU): каждый memory-CA живёт на девайсе
        # СВОЕГО слоя; энкодер/компрессор/банк — на главном (первый параметр базы).
        _layers = find_decoder_layers(base)
        _layer_dev = {li: next(_layers[li].parameters()).device for li in cfg.cross_attn_layers}
        in_dim = cfg.projector_input_dim or 2 * self._hidden_dim

        self.encoder = EmotionalEncoder(
            num_layers=cfg.num_layers, input_dim=in_dim,
            bottleneck_dim=cfg.encoder_bottleneck, gate_init=cfg.encoder_gate_init,
        ).to(device)
        self.compressor = EpisodeCompressor(
            num_layers=cfg.num_layers, bottleneck_dim=cfg.encoder_bottleneck,
            hidden_dim=self._hidden_dim, K=cfg.K_memory_tokens,
            n_heads=cfg.compressor_n_heads,
            goal_cond_k=getattr(cfg, "goal_cond_k", None),   # ChronoAnchorConfig only
        ).to(device)
        self.memory_ca_layers = nn.ModuleDict()
        for li in cfg.cross_attn_layers:
            ca = MemoryBottleneckCrossAttention(
                hidden_dim=self._hidden_dim, bottleneck_dim=cfg.memory_ca_bottleneck,
                num_heads=cfg.memory_ca_num_heads, dropout=cfg.memory_ca_dropout,
                gate_init=cfg.memory_ca_gate_init,
                use_adaptive_gate=cfg.use_adaptive_gates,
                adaptive_gate_hidden=cfg.adaptive_gate_hidden,
                adaptive_use_bank_size=cfg.adaptive_use_bank_size,
            ).to(device=_layer_dev[li], dtype=self._dtype)
            # scalar gate stays float32 (checkpoint compatibility, tanh stability)
            ca.gate = nn.Parameter(torch.tensor([cfg.memory_ca_gate_init],
                                                dtype=torch.float32, device=_layer_dev[li]))
            if cfg.use_adaptive_gates and ca.adaptive_gate is not None:
                ca.adaptive_gate = ca.adaptive_gate.float().to(device)
            self.memory_ca_layers[str(li)] = ca
        self.bank = PersistentMemoryBank(
            max_episodes=cfg.bank_max_episodes, K_per_episode=cfg.K_memory_tokens,
            hidden_dim=self._hidden_dim, device=device,
            novelty_threshold=cfg.novelty_threshold, eviction_mode=cfg.eviction_mode,
        )
        self.bank._auto_sync_target = self.set_memory

        if self._pending_state is not None:
            self._load_states(self._pending_state)
            self._pending_state = None
        self._install_hooks(base)
        self.set_inference_mode()

    def on_detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        super().on_detach()

    def on_pre_forward(self) -> None:
        """No-op: the bank PERSISTS across forwards — that is the whole point."""

    def on_post_forward(self, activations: dict[int, Any]) -> Optional[dict[str, Any]]:
        """No-op: episodes are recorded explicitly via `record_episode` (the harness/agent
        decides what counts as an episode), not from every Pass-1."""
        return None

    # ---------------- hooks ----------------

    def _install_hooks(self, base: nn.Module) -> None:
        layers = find_decoder_layers(base)
        cfg = self.config
        # memory-CA injection + attention tracking (consolidate eviction needs it)
        for li in cfg.cross_attn_layers:
            ca = self.memory_ca_layers[str(li)]

            def make_ca_hook(ca_module):
                def hook(module, inputs, output):
                    if self._current_memory is None or self._current_memory.size(0) == 0:
                        return output
                    hs = output[0] if isinstance(output, tuple) else output
                    # CA may run at a different dtype than the base (fp32 CA on an fp16 base
                    # for from-scratch training — the GoalAnchor convention); cast both ways.
                    ca_dtype = next(ca_module.parameters()).dtype
                    mem = self._current_memory.to(device=hs.device, dtype=ca_dtype)
                    modified = ca_module(hs.to(ca_dtype), mem,
                                         bank_size=len(self.bank)).to(hs.dtype)
                    if self._gain != 1.0:
                        modified = hs + self._gain * (modified - hs)
                    attn = getattr(ca_module, "_last_attn_weights", None)
                    if len(self.bank) > 0 and attn is not None:
                        with torch.no_grad():
                            n, k = len(self.bank), self.bank.K
                            if attn.size(-1) == n * k:
                                self.bank.record_attention(
                                    attn.mean(dim=(0, 1, 2)).reshape(n, k).mean(dim=-1))
                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    return modified
                return hook

            self._hook_handles.append(layers[li].register_forward_hook(make_ca_hook(ca)))
        # activation taps (record path): last + mean per layer
        for li in cfg.activation_layers:
            def make_tap(idx):
                def hook(module, inputs, output):
                    if not self._collecting:
                        return
                    hs = output[0] if isinstance(output, tuple) else output
                    self._collected[idx] = {"last": hs[:, -1, :].detach(),
                                            "mean": hs.mean(dim=1).detach()}
                return hook
            self._hook_handles.append(layers[li].register_forward_hook(make_tap(li)))

    # ---------------- record path ----------------

    def record_episode(self, input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       meta: Optional[dict] = None, force: bool = False,
                       clean_snap: bool = False, trainable: bool = False) -> bool:
        """Episode → encoder → compressor → bank.add (novelty-gated). Returns written?

        By default the snapshot is taken WITH memory injection active (steady-state
        semantics: the episode encodes the model's state-of-mind given what it already
        remembers). clean_snap=True disables memory for a raw snapshot.

        trainable=True keeps the encoder/compressor grad graph alive (the base forward
        is no-grad either way — the taps detach) so a downstream LM loss can train the
        compressor THROUGH the injected memory tokens (tandem-v1 behavioral training).
        """
        if trainable:
            return self._record_episode(input_ids, attention_mask, meta, force, clean_snap,
                                        trainable=True)
        with torch.no_grad():
            return self._record_episode(input_ids, attention_mask, meta, force, clean_snap)

    def _record_episode(self, input_ids, attention_mask, meta, force, clean_snap,
                        trainable: bool = False) -> bool:
        if self.encoder is None:
            raise RuntimeError("Chronographer is not attached to a pipeline.")
        self._collected.clear()
        saved = self._current_memory
        if clean_snap:
            self._current_memory = None
        self._collecting = True
        try:
            with torch.no_grad():
                self._pipeline.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self._collecting = False
            if clean_snap:
                self._current_memory = saved
        enc_dev = next(self.encoder.parameters()).device
        acts = torch.stack(
            [torch.cat([self._collected[i]["last"][0], self._collected[i]["mean"][0]],
                       dim=-1).to(enc_dev)
             for i in sorted(self.config.activation_layers)], dim=0)
        per_layer = self.encoder(acts.unsqueeze(0).float())
        memory_tokens = self.compressor(per_layer, goal_vec=self._goal_vec)   # [1, K, hidden]
        written = self.bank.add(memory_tokens.squeeze(0).to(self._dtype),
                                meta or {}, force=force)
        # The bank detaches on add (persistent store). For training, mirror the episode in a
        # LIVE list and inject that instead, so the LM loss backprops into the compressor.
        # Assumes the training discipline: force-add, no eviction (bank_max ≥ session length).
        if trainable and written:
            self._live_tokens.append(memory_tokens.squeeze(0).to(self._dtype))
            self.set_memory(torch.cat(self._live_tokens, dim=0))
        return written

    # ---------------- replay controls ----------------

    def set_memory(self, memory_tokens: Optional[torch.Tensor]) -> None:
        """Override the injected memory (None = off). bank.add re-syncs it automatically."""
        self._current_memory = memory_tokens

    def set_goal_condition(self, goal: Optional[torch.Tensor]) -> None:
        """Tandem-v1 (mechanism 1): condition episode COMPRESSION on the active goal.

        `goal` — the goal latent in the base's hidden space, two forms:
          [hidden]      — mean-pooled vector → additive path (goal_proj, v1.2-v1.5);
          [N, hidden]   — the GoalAnchor's cog tokens AS IS → per-token attention path
                          (v1.7: mean-pooling loses half the signal for graded content).
        Affects only FUTURE record_episode calls; None switches conditioning off.
        With zero-init projections (untrained) both paths are exact no-ops."""
        if goal is None:
            self._goal_vec = None
            return
        g = goal.detach().float()
        if g.dim() == 3:
            g = g.squeeze(0)
        self._goal_vec = g

    def clear_bank(self) -> None:
        self.bank.clear()
        self._current_memory = None
        self._live_tokens = []

    def set_gain(self, gain: float) -> None:
        """The daimon fader: runtime multiplier over the memory-CA contribution
        (trained gates untouched). 1.0 = as trained, 0.0 = memory silent."""
        self._gain = float(gain)

    def set_inference_mode(self) -> None:
        for m in (self.encoder, self.compressor, self.memory_ca_layers):
            if m is not None:
                m.eval()

    # ---------------- persistence ----------------

    def _load_states(self, ck: dict) -> None:
        self.encoder.load_state_dict(ck["encoder_state"])
        # strict=False: pre-tandem checkpoints carry no goal_proj keys (zero-init = no-op)
        self.compressor.load_state_dict(ck["compressor_state"], strict=False)
        for k, state in ck["memory_ca_states"].items():
            # strict=False: legacy checkpoints carry no adaptive_gate keys
            self.memory_ca_layers[str(k)].load_state_dict(state, strict=False)

    def save_checkpoint(self, path: str) -> None:
        if self.encoder is None:
            raise RuntimeError("Nothing to save: not attached/built.")
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "chronographer",
            "config": asdict(self.config),
            "encoder_state": self.encoder.state_dict(),
            "compressor_state": self.compressor.state_dict(),
            "memory_ca_states": {k: v.state_dict() for k, v in self.memory_ca_layers.items()},
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str,
                        config: Optional[ChronographerConfig] = None) -> "Chronographer":
        """Load trained weights. Accepts both the framework format (kind="chronographer",
        config embedded) and the LEGACY `content_pipeline.pt` (bare states — the shape
        then comes from `config` or the 1B defaults). Modules are built lazily at attach."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("kind") not in (None, "chronographer"):
            raise ValueError(f"not a Chronographer checkpoint (kind={ck.get('kind')!r})")
        if config is None and isinstance(ck.get("config"), dict):
            config = ChronographerConfig(**{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in ck["config"].items()})
        mod = cls(config)
        missing = [k for k in ("encoder_state", "compressor_state", "memory_ca_states")
                   if k not in ck]
        if missing:
            raise ValueError(f"checkpoint lacks {missing}")
        mod._pending_state = ck
        return mod
