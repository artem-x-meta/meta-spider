"""BottleneckCrossAttention — the meta-attention head for injecting cognitive tokens.

One module is registered on each target layer of the base model (32 for 8B, 16 for 1B)
via a forward hook. Q comes from the layer's compressed hidden states (after down_proj into
the bottleneck), K/V — from the cognitive tokens buffer. The output is projected back to
hidden_dim and mixed into the residual via a tanh gate.

Features:
  - bottleneck 4096→256→4096 (8B) or 2048→128→2048 (1B) — half the cost of full-rank
  - tanh(gate) init=0.3 — linear zone, gradients flow; init=2.0 = frozen
  - token_preference [num_cog_tokens] — a per-source bias shared across heads
    "each LLM CA layer chooses which cognitive tokens to attend to"
  - gate and token_preference are kept in float32 for precision
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["BottleneckCrossAttention"]


class BottleneckCrossAttention(nn.Module):
    """Meta-attention head: cognitive tokens → correction of an LLM layer.

    Args:
        hidden_dim: the base model's hidden state dimensionality.
        bottleneck_dim: attention compression (default 256 for 8B, 128 for 1B).
        num_heads: number of heads (bottleneck_dim must be divisible by num_heads).
        num_cognitive_tokens: size of the cognitive tokens buffer (= encoder's num_layers).
        dropout: attention dropout (default 0.1).
        gate_init: initial value of the gate scalar (default 0.3 — linear zone of tanh).
        token_preference_init: initial value of the per-source bias (default 0.0).
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 256,
        num_heads: int = 4,
        num_cognitive_tokens: int = 32,
        dropout: float = 0.1,
        gate_init: float = 0.3,
        token_preference_init: float = 0.0,
    ):
        super().__init__()
        assert bottleneck_dim % num_heads == 0, (
            f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.num_cognitive_tokens = num_cognitive_tokens
        self.head_dim = bottleneck_dim // num_heads

        # Pre-norm before compression
        self.norm = nn.LayerNorm(hidden_dim)
        # Compress hidden states into the bottleneck
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        # Query from the compressed space
        self.q_proj = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        # Key/Value from the cognitive tokens (full hidden_dim → bottleneck)
        self.k_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        # Back to full hidden_dim
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # gate is always float32 for precision (per Phase 2 finding)
        self.gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
        # token_preference shared across heads, float32
        self.token_preference = nn.Parameter(
            torch.full((num_cognitive_tokens,), token_preference_init, dtype=torch.float32)
        )

        self._last_attn_weights: torch.Tensor | None = None

        # AGC (opt-in, default OFF — does not touch the validated regime): negative feedback
        # against injection RUNAWAY. CA is by construction a positive amplifier (query=live
        # residual attends more strongly to the cognitive tokens as drift grows → injection
        # grows → more drift → "doubt" accumulates into paranoia/a loop). AGC dampens the
        # injection as the residual saturates with the injected direction → "doubt" stays
        # ~uniform (a setpoint).
        self.agc_enabled: bool = False
        # The tank metaphor: inflow (injection) is constant in magnitude (out_norm≈const), but
        # the LEVEL of doubt in the activations grows over time (a slow drain) → level ∝
        # generation step. decay mode: α is strong at the start (fill the tank to target),
        # settles toward floor (top up exactly against the drain → level is UNIFORM, the signal
        # is NOT muted to zero).
        self.agc_mode: str = "decay"     # 'decay' (level∝step) | 'measure' (hidden-measure, diag)
        self.agc_floor: float = 0.3      # the maintenance level of α (NOT zero — keep the signal)
        self.agc_tau: float = 16.0       # decay scale (steps until reaching floor)
        self.agc_measure: str = "drift"  # for 'measure' mode: 'drift'|'out_norm'|'proj'
        self.agc_target: float = 0.15
        self.agc_scale: float = 0.10
        self._gen_step: int = 0          # generation step (reset on prefill seq>1)
        self._last_agc_alpha: float | None = None
        self._last_agc_align: float | None = None
        self._agc_diag: dict | None = None

        # v0.2 — STATIC GAIN (the uncertainty potentiometer). A runtime multiplier on the
        # injection strength: gain=1.0 → trained baseline; >1 amplify (more doubt → more refusal);
        # <1 attenuate; <0 invert (push toward confidence). Composes WITH gate and AGC. NOT a
        # trained parameter — a control knob set at inference. Validated monotonic range ~[0, 1.5]
        # before positive-feedback runaway (see docs/results/gemma-4-12b/gain-potentiometer.md);
        # combine with AGC near the ceiling for stability.
        self.gain: float = 1.0

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.down_proj, self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cognitive_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention with cognitive tokens to hidden_states.

        Args:
            hidden_states: [B, seq_len, hidden_dim] — output of an LLM layer.
            cognitive_tokens: [B, num_cog_tokens, hidden_dim] — from the ReflexionBuffer.

        Returns:
            hidden_states + tanh(gate) · BottleneckCrossAttention(h, cog_tokens)
        """
        residual = hidden_states
        h = self.norm(hidden_states)

        batch_size, seq_len, _ = h.shape
        num_cog = cognitive_tokens.shape[1]

        # Compress hidden states into the bottleneck
        h_compressed = self.down_proj(h)            # [B, seq, bottleneck]

        Q = self.q_proj(h_compressed)               # [B, seq, bottleneck]
        K = self.k_proj(cognitive_tokens)           # [B, num_cog, bottleneck]
        V = self.v_proj(cognitive_tokens)           # [B, num_cog, bottleneck]

        # Multi-head: split into num_heads × head_dim
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_cog, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_cog, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, seq|num_cog, head_dim]

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # [B, num_heads, seq, num_cog]

        # Token preference bias: shared across heads, broadcast over position
        preference_bias = self.token_preference[:num_cog].to(attn_weights.dtype)
        attn_weights = attn_weights + preference_bias[None, None, None, :]

        attn_weights = torch.softmax(attn_weights, dim=-1)
        self._last_attn_weights = attn_weights  # before dropout, for diagnostics
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq, head_dim]
        # Back: [B, seq, bottleneck]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.bottleneck_dim)

        # Project back to full dimensionality
        output = self.up_proj(attn_output)

        # gate in float32; cast to the output dtype
        gate_value = torch.tanh(self.gate).to(output.dtype)
        if self.agc_enabled and self.agc_mode == "decay":
            # Level in the tank ∝ generation step (constant inflow). reset on prefill (seq>1 =
            # start of a new generation). α: 1 at the start (fill to target) → floor (keep the
            # level uniform against the drain, do NOT mute). A scalar per step, the same across
            # positions.
            self._gen_step = 0 if seq_len > 1 else self._gen_step + 1
            decay = math.exp(-self._gen_step / max(self.agc_tau, 1e-6))
            alpha = self.agc_floor + (1.0 - self.agc_floor) * decay
            self._last_agc_alpha = float(alpha)
            return residual + (self.gain * alpha * gate_value) * output
        if self.agc_enabled:
            # measure mode (diagnostics / a future adaptive probe-meter of the level). Candidate
            # measures: drift/out_norm/proj — all turned out NOT to grow (content drowns the doubt
            # component); the exact level would come from a trained probe (ConfidenceHead) — deferred.
            res_f, out_f = residual.float(), output.float()
            cog_dir = nn.functional.normalize(
                cognitive_tokens.float().mean(dim=1, keepdim=True), dim=-1)  # [B,1,hidden]
            drift = (res_f * cog_dir).sum(-1)                                # [B, seq]
            out_norm = out_f.norm(dim=-1)                                    # [B, seq]
            proj = (res_f * out_f).sum(-1) / (out_f.pow(2).sum(-1) + 1e-6)   # [B, seq]
            measure = {"drift": drift, "out_norm": out_norm, "proj": proj}[self.agc_measure]
            alpha = ((self.agc_target - measure) / self.agc_scale).clamp(0.0, 1.0)  # [B, seq]
            self._last_agc_align = float(measure.mean())
            self._last_agc_alpha = float(alpha.mean())
            self._agc_diag = {"drift": float(drift.mean()), "out_norm": float(out_norm.mean()),
                              "proj": float(proj.mean())}
            return residual + (self.gain * alpha.unsqueeze(-1).to(output.dtype)) * gate_value * output
        return residual + (self.gain * gate_value) * output

    # --- Control (v0.2) ---

    def set_gain(self, gain: float) -> None:
        """Set the static injection gain (the uncertainty potentiometer). See __init__: gain=1.0
        is the trained baseline, >1 amplifies, <1 attenuates, <0 inverts toward confidence."""
        self.gain = float(gain)

    # --- Diagnostics ---

    def get_gate_value(self) -> float:
        """tanh(gate) — the current 'volume' of injection in this layer."""
        return torch.tanh(self.gate).item()

    def get_raw_gate_value(self) -> float:
        """Raw gate without tanh — useful for tracking the zone (0.3 vs 2.0)."""
        return self.gate.item()

    def get_gate_grad_norm(self) -> float:
        """Gate gradient norm — for diagnosing dead gradients."""
        if self.gate.grad is not None:
            return self.gate.grad.norm().item()
        return 0.0

    def get_token_preference_values(self) -> list[float]:
        """token_preference bias over the cognitive tokens (raw, without softmax)."""
        return self.token_preference.detach().tolist()

    def get_preference_top_k(self, k: int = 3) -> dict:
        """Top-k and bottom-k preferred cognitive tokens."""
        prefs = self.token_preference.detach()
        top_vals, top_idx = torch.topk(prefs, k)
        bot_vals, bot_idx = torch.topk(prefs, k, largest=False)
        return {
            "top": [(idx.item(), val.item()) for idx, val in zip(top_idx, top_vals)],
            "bottom": [(idx.item(), val.item()) for idx, val in zip(bot_idx, bot_vals)],
        }
