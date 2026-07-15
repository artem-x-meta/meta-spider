"""ChronoAnchorTrainer — the measured recipe for training the goal-conditioned memory organ.

Four ingredients, all load-bearing (each was measured; dropping one breaks the channel):

  1. **CENTER the goal latent** (`fit_goal_center` over the train goals). Without it the raw
     goal vector is dominated by a constant component: the conditioning projection learns a
     global bias instead of a per-goal condition, and the channel dies (Llama-1B: 0.486 ≈ the
     blind ceiling, vs 1.000 with centering).
  2. **FOCUS the LM loss on the DECISION tokens** (`decision_markers` per sample, weight ×5).
     The goal decides 1-2 tokens out of ~15 in a gold answer; without re-weighting the
     goal-dependent gradient drowns in boilerplate (in-family 0.931 → held-phrasings 0.639
     without focus, 0.875 with).
  3. **DIVERSE goal phrasings** in the training set (≥ ~25 per goal type). Narrow phrasings →
     the channel memorizes surface form and fails on unseen wording (held-phrasing 0.25 → 1.00).
  4. **Train to the loss plateau.** A diverse task first DIPS below the narrow one, then
     overtakes it — stopping early reads as "diversity hurts" (2 epochs 0.806 → 4 epochs 1.000).

The training signal is a plain LM loss on a gold continuation that OBEYS the goal — behavioral,
not recall: the memory channel must learn OBLIGATION, and obligation is only learnable from
"what the right action was", never from "what the episode contained" (measured: recall-trained
memory carries the fact but not the constraint).

    trainer = ChronoAnchorTrainer(pipeline, anchor)
    trainer.fit(samples, epochs=4, lr=5e-5)     # samples: ChronoAnchorSample
    anchor.save_checkpoint("chrono_anchor.pt")  # the goal center travels inside
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

__all__ = ["ChronoAnchorSample", "ChronoAnchorTrainer"]


@dataclass
class ChronoAnchorSample:
    """One training session.

    goal:             the goal text — NEVER enters the prompt, only the latent channel.
    episodes:         texts recorded into the bank, in order (agent steps / transcript prefixes).
    prompt:           the final prompt the model answers (the decision point).
    gold:             the gold answer that OBEYS the goal (behavioral target).
    decision_markers: substrings of `gold` that carry the decision (the tier word, the price,
                      the chosen option) — these tokens get the loss weight. Empty = uniform.
    """

    goal: str
    episodes: list[str]
    prompt: str
    gold: str
    decision_markers: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class ChronoAnchorTrainer:
    """Trains ONLY the memory organ (encoder + compressor + memory-CA); the base stays frozen.

    Gradients flow through the frozen base into the wrapper (the base acts as a proxy loss) —
    the same two-pass discipline as the rest of the framework.
    """

    def __init__(self, pipeline: Any, anchor: Any, *,
                 decision_weight: float = 5.0,
                 max_length: int = 8192,
                 render: Optional[Callable[[str], str]] = None):
        # NB: not every tokenizer takes add_special_tokens / return_offsets_mapping (slow ones,
        # test fakes) — every call below degrades gracefully instead of hard-failing.
        self.pipeline = pipeline
        self.anchor = anchor
        self.decision_weight = decision_weight
        self.max_length = max_length
        self.render = render or (lambda text: text)
        self.tok = pipeline.tokenizer
        self.device = next(pipeline.model.parameters()).device

    # ---------------- loss weighting ----------------

    def _tok_ids(self, text: str, *, special: bool = True) -> torch.Tensor:
        try:
            enc = self.tok(text, return_tensors="pt", add_special_tokens=special,
                           truncation=True, max_length=self.max_length)
        except TypeError:                       # tokenizer without add_special_tokens
            enc = self.tok(text, return_tensors="pt")
        return enc.input_ids

    def _offsets(self, text: str) -> Optional[list]:
        """Char spans per token, or None if the tokenizer cannot produce them (slow/fake)."""
        try:
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        except (TypeError, NotImplementedError, ValueError):
            return None
        offs = enc.get("offset_mapping") if isinstance(enc, dict) else getattr(
            enc, "offset_mapping", None)
        return list(offs) if offs is not None else None

    def _decision_weights(self, gold_text: str, n_tokens: int,
                          markers: list[str]) -> torch.Tensor:
        """Weight ×decision_weight on tokens overlapping the decision markers.

        Precise path: char offsets from a fast tokenizer. Fallback (no offsets): weight the
        tokens of each marker phrase by matching their id-subsequence in the gold ids.
        """
        w = torch.ones(n_tokens)
        if not markers:
            return w
        spans = []
        low = gold_text.lower()
        for m in markers:
            i = low.find(m.lower())
            if i >= 0:
                spans.append((i, i + len(m)))
        if not spans:
            return w

        offsets = self._offsets(gold_text)
        if offsets is not None:
            for i, (a, b) in enumerate(offsets[:n_tokens]):
                if any(a < end and b > start for start, end in spans):
                    w[i] = self.decision_weight
            return w

        gold_ids = self._tok_ids(gold_text, special=False)[0].tolist()
        for m in markers:                       # id-subsequence match
            m_ids = self._tok_ids(m, special=False)[0].tolist()
            if not m_ids:
                continue
            for i in range(len(gold_ids) - len(m_ids) + 1):
                if gold_ids[i:i + len(m_ids)] == m_ids:
                    for j in range(i, min(i + len(m_ids), n_tokens)):
                        w[j] = self.decision_weight
        return w

    # ---------------- one session ----------------

    def _record_session(self, sample: ChronoAnchorSample, trainable: bool) -> None:
        self.anchor.clear_bank()
        self.anchor.set_goal(sample.goal)                 # centered latent → conditions compression
        for text in sample.episodes:
            enc = self.tok(self.render(text), return_tensors="pt", truncation=True,
                           max_length=self.max_length)
            ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device) if hasattr(enc, "attention_mask") else None
            self.anchor.record_episode(ids, mask, meta={"train": True}, force=True,
                                       trainable=trainable)

    def step_loss(self, sample: ChronoAnchorSample) -> torch.Tensor:
        """Behavioral LM loss on the goal-obeying gold, weighted on the decision tokens."""
        self._record_session(sample, trainable=True)
        eos = getattr(self.tok, "eos_token", "") or ""
        p_ids = self._tok_ids(self.render(sample.prompt))
        g_ids = self._tok_ids(sample.gold + eos, special=False)
        ids = torch.cat([p_ids, g_ids], dim=1).to(self.device)
        n_p, n_g = p_ids.size(1), g_ids.size(1)

        try:    # only the gold tail's logits — a 262k-vocab model would OOM on the full ones
            out = self.pipeline.model(input_ids=ids, logits_to_keep=n_g + 1)
        except TypeError:
            out = self.pipeline.model(input_ids=ids)
        if hasattr(out, "logits"):
            logits = out.logits.float()
        else:   # a bare-hidden forward (test fakes / custom heads) → score through lm_head
            hidden = out[0] if isinstance(out, tuple) else out
            logits = self.pipeline.model.lm_head(hidden).float()
        lg = logits[0, -(n_g + 1):-1] if logits.size(1) == n_g + 1 else logits[0, n_p - 1:-1]
        target = ids[0, n_p:].to(lg.device)
        ce = F.cross_entropy(lg, target, reduction="none")
        w = self._decision_weights(sample.gold + eos, ce.size(0),
                                   sample.decision_markers).to(ce.device)
        return (w * ce).sum() / w.sum()

    # ---------------- fit ----------------

    def fit(self, samples: list[ChronoAnchorSample], *, epochs: int = 4, lr: float = 5e-5,
            grad_clip: float = 1.0, shuffle_seed: int = 303,
            log_every: int = 40, on_log: Optional[Callable[[str], None]] = None) -> list[float]:
        """Train the organ. Fits the goal center from the TRAIN goals first (no leakage)."""
        if self.anchor.encoder is None:
            raise RuntimeError("Attach the ChronoAnchor to the pipeline before training.")
        log = on_log or (lambda m: None)

        goals = sorted({s.goal for s in samples})
        self.anchor.fit_goal_center(goals)               # ingredient 1 (necessary)
        log(f"goal center fitted over {len(goals)} unique goals")

        params = (list(self.anchor.encoder.parameters())
                  + list(self.anchor.compressor.parameters())
                  + list(self.anchor.memory_ca_layers.parameters()))
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr)
        fp16_base = next(self.pipeline.model.parameters()).dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=fp16_base and self.device.type == "cuda")
        rng = random.Random(shuffle_seed)

        for m in (self.anchor.encoder, self.anchor.compressor, self.anchor.memory_ca_layers):
            m.train()
        losses: list[float] = []
        order = list(samples)
        for ep in range(epochs):
            rng.shuffle(order)
            for i, sample in enumerate(order):
                loss = self.step_loss(sample)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
                if log_every and ((i + 1) % log_every == 0 or i == 0):
                    tail = losses[-log_every:]
                    log(f"[ep{ep} {i + 1}/{len(order)}] loss={loss.item():.3f} "
                        f"(ma={sum(tail) / len(tail):.3f})")
        self.anchor.set_inference_mode()
        self.anchor.clear_bank()
        return losses
