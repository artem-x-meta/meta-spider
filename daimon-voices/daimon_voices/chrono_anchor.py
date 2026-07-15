"""ChronoAnchor — the goal-conditioned episodic memory: ONE organ, not two voices.

The union of the GoalAnchor (holds the FUTURE: the goal) and the Chronographer (holds the
PAST: the episodes). Instead of two cross-attention tracts summing on the residual, the goal
becomes a PRIVILEGED INPUT of the memory organ: it conditions episode COMPRESSION, so every
memory token carries a piece of the active goal.

    anchor = ChronoAnchor.from_checkpoint("chrono_anchor.pt")
    pipe.attach(anchor)
    anchor.set_goal("Stay under $60; never buy the allergenic color.")   # once per session
    agent = MetaAgent(policy, tools, step_hooks=[anchor])                # episodes per step
    agent.run("Find me a desk lamp")                                     # goal steers decisions

Validated (Llama-3.2-1B, `docs/results/llama-1b/tandem-one-organ-scratch.md`): the goal exists
ONLY as a latent (never in the prompt) and still drives the decision — 1.000 vs the 0.500
structural ceiling of the same weights without the goal, 0.972 under unseen adversarial lures.
On gemma-4-12b the channel is selective (5/11 goal types) but beats the TEXT goal where the
model's RLHF prior overrides an instruction (`docs/results/gemma-12b/…`).

The training recipe that makes it work (all four are load-bearing, measured):
  1. CENTER the goal latent (subtract the train-goal mean) — otherwise the constant component
     eats the channel and conditioning degenerates into a global bias;
  2. FOCUS the LM loss on the decision tokens — otherwise the goal-dependent gradient drowns;
  3. DIVERSE goal phrasings — otherwise the channel memorizes surface form;
  4. train to the loss plateau — a diverse task first dips, then generalizes.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch

from daimon_voices.chronographer import Chronographer
from daimon_voices.config import ChronoAnchorConfig

__all__ = ["ChronoAnchor"]

CHECKPOINT_FORMAT_VERSION = "1.0"


class ChronoAnchor(Chronographer):
    """Chronographer whose compressor is conditioned on the session GOAL.

    Adds to the memory organ: goal encoding from the FROZEN base's own activations (no separate
    trained anchor encoder is needed — the base already encodes the goal, `set_goal`), the
    centering vector (learned at train time, travels in the checkpoint), and the agentic
    lifecycle (`on_step_end` records the session-so-far as an episode).
    """

    name = "chrono_anchor"

    def __init__(self, config: Optional[ChronoAnchorConfig] = None) -> None:
        cfg = config or ChronoAnchorConfig()
        super().__init__(cfg)
        self.config: ChronoAnchorConfig = cfg
        self.goal_center: Optional[torch.Tensor] = None   # [hidden] or [n_layers, hidden]
        self.goal_text: Optional[str] = None
        self._pending_center: Optional[torch.Tensor] = None

    # ---------------- goal ----------------

    @torch.no_grad()
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        """Goal text → latent, read from the FROZEN base's own activations.

        pool="last"   : [n_layers, hidden] — the last token per read layer (validated on 1B);
        pool="tokens" : [n_layers*T, hidden] — every goal token on the read layers (for
                        compositional goals, where a single summary vector merges the
                        distinguishing clause with the task description).
        Centering (if a center is loaded/learned) is applied here — the raw goal latent is
        dominated by a large constant component that would otherwise swamp the channel.
        """
        if self._pipeline is None:
            raise RuntimeError("ChronoAnchor is not attached to a pipeline.")
        pipe = self._pipeline
        tok = pipe.tokenizer
        device = next(pipe.model.parameters()).device
        enc = tok(goal_text, return_tensors="pt")
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None
        layers = sorted(self.config.activation_layers)

        if self.config.goal_pool == "tokens":
            out = pipe.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states                      # [0] = embeddings; layer i → hs[i+1]
            v = torch.cat([hs[i + 1][0].float() for i in layers], dim=0)
        else:
            if pipe.collector is None:
                raise RuntimeError("Pipeline without an ActivationCollector.")
            pipe.collector.clear()
            pipe.collector.unfreeze()
            try:
                pipe.model(input_ids=ids, attention_mask=mask)
                snap = pipe.collector.get_snapshot()
            finally:
                pipe.collector.freeze()
            v = torch.stack([snap[i][0].float() for i in layers], dim=0)

        if self.goal_center is not None:
            v = v - self.goal_center.to(v.device)
        return v

    def set_goal(self, goal_text: str) -> torch.Tensor:
        """Set the session goal (encode → condition future episode compression).

        The goal NEVER enters the prompt: it lives only as a latent condition. Episodes already
        in the bank were compressed under the previous goal — `clear_bank()` first if the goal
        changes mid-session and old episodes should be re-encoded."""
        vec = self.encode_goal(goal_text)
        self.set_goal_condition(vec)
        self.goal_text = goal_text
        return vec

    def clear_goal(self) -> None:
        self.set_goal_condition(None)
        self.goal_text = None

    @torch.no_grad()
    def fit_goal_center(self, goal_texts: list[str]) -> torch.Tensor:
        """Learn the centering vector from the TRAIN goals (call once, before training).

        The center travels in the checkpoint and is applied at inference — no leakage: it is a
        property of the goal distribution the wrapper was trained on, not of the eval set.
        """
        saved, self.goal_center = self.goal_center, None      # encode raw
        try:
            vecs = [self.encode_goal(g) for g in goal_texts]
        finally:
            self.goal_center = saved
        if self.config.goal_pool == "tokens":
            center = torch.stack([v.mean(dim=0) for v in vecs]).mean(dim=0)   # [hidden]
        else:
            center = torch.stack(vecs).mean(dim=0)                            # [n_layers, hidden]
        self.goal_center = center
        return center

    # ---------------- agentic lifecycle (MetaAgent step_hooks) ----------------

    def on_step_start(self, step: int = 0, session: Any = None) -> None:
        """No-op: memory injects on every forward — the goal steers the whole step."""

    def on_step_end(self, step: int = 0, session: Any = None) -> None:
        """Record the session-so-far as an episode (goal-conditioned compression).

        Called by MetaAgent after each decision+observation. The bank ACCUMULATES: the model's
        state-of-mind after every step, each memory token already carrying the goal.
        """
        if session is None or self.encoder is None:
            return
        text = session.render() if hasattr(session, "render") else str(session)
        if not text.strip():
            return
        tok = self._pipeline.tokenizer
        device = next(self._pipeline.model.parameters()).device
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=self.config.max_episode_tokens)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None
        self.record_episode(ids, mask, meta={"step": step, "goal": self.goal_text}, force=True)

    # ---------------- persistence ----------------

    def save_checkpoint(self, path: str) -> None:
        if self.encoder is None:
            raise RuntimeError("Nothing to save: not attached/built.")
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "chrono_anchor",
            "config": asdict(self.config),
            "encoder_state": self.encoder.state_dict(),
            "compressor_state": self.compressor.state_dict(),
            "memory_ca_states": {k: v.state_dict() for k, v in self.memory_ca_layers.items()},
            "goal_center": (self.goal_center.cpu() if self.goal_center is not None else None),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str,
                        config: Optional[ChronoAnchorConfig] = None) -> "ChronoAnchor":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        kind = ck.get("kind")
        if kind not in (None, "chrono_anchor", "chronographer"):
            raise ValueError(f"not a ChronoAnchor checkpoint (kind={kind!r})")
        if config is None and isinstance(ck.get("config"), dict):
            known = ChronoAnchorConfig.__dataclass_fields__
            config = ChronoAnchorConfig(**{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in ck["config"].items() if k in known})
        mod = cls(config)
        missing = [k for k in ("encoder_state", "compressor_state", "memory_ca_states")
                   if k not in ck]
        if missing:
            raise ValueError(f"checkpoint lacks {missing}")
        mod._pending_state = ck
        mod._pending_center = ck.get("goal_center")
        return mod

    def on_attach(self, pipeline: Any) -> None:
        super().on_attach(pipeline)
        if self._pending_center is not None:
            device = next(pipeline.model.parameters()).device
            self.goal_center = self._pending_center.to(device).float()
            self._pending_center = None
