"""SelectiveEncoder — the original feedforward encoder from the paper
`meta-attention-is-all-you-need`. Per-layer projectors + per-layer encoder gates +
shared output projector → multi-token cognitive tokens for cross-attention injection.

Project calibration record (Phase 2 Selective on Llama-8B MMLU):
  - selective accuracy 89.1% @ 37% coverage (base 64.6%)
  - total_recovery 90.7%
  (the old refusal_prec 99.84% was a metric artifact; honest ≈0.50 vs the oracle)

Pipeline:

    layer i activation [B, hidden_dim]
        → projector_i (LayerNorm + Linear → bottleneck_dim + GELU)
        → * tanh(layer_gate_i)                       # per-layer scalar gate (float32)
        → stack over all layers                       [B, n_layers, bottleneck_dim]
        → output_proj (LN + Linear→hidden + GELU + Linear→hidden)
        → output_norm (LayerNorm)
        → cognitive tokens                            [B, n_layers, hidden_dim]

No self-attention between the layers' cognitive tokens — each projector
works independently. Inter-layer relationships materialize in `BottleneckCrossAttention`
via the `token_preference` bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from meta_core.encoders.base import Encoder

__all__ = ["SelectiveEncoder"]


class SelectiveEncoder(nn.Module, Encoder):
    """Feedforward encoder with per-layer gates → cognitive tokens.

    Args:
        hidden_dim: the base model's hidden state dimensionality (e.g. 4096 for 8B).
        num_layers: number of target layers (= number of cognitive tokens).
        bottleneck_dim: per-layer compression (default 256). The bottleneck filters noise.
        gate_init: initial value of the gate scalar in the linear zone of tanh (default 0.3 —
            `tanh'(0.3)=0.91`, gradient flows; init=2.0 freezes the gates forever).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 32,
        bottleneck_dim: int = 256,
        gate_init: float = 0.3,
    ):
        nn.Module.__init__(self)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.gate_init = gate_init

        # Per-layer projectors (one per target layer)
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

        # Per-layer scalar gates (float32 for precision)
        self.layer_gates = nn.ParameterList([
            nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
            for _ in range(num_layers)
        ])

        # Shared output projector into the base's embedding space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, activation_list: list[torch.Tensor]) -> torch.Tensor:
        """Run the activations through the Selective encoder.

        Args:
            activation_list: list of num_layers tensors [B, hidden_dim].

        Returns:
            cognitive_tokens: [B, num_layers, hidden_dim].
        """
        assert len(activation_list) == self._num_layers, (
            f"Expected {self._num_layers} activations, got {len(activation_list)}"
        )

        projected = []
        for activation, projector, gate in zip(
            activation_list, self.layer_projectors, self.layer_gates
        ):
            proj = projector(activation)
            gate_value = torch.tanh(gate).to(proj.dtype)
            projected.append(gate_value * proj)

        # [B, num_layers, bottleneck_dim]
        stacked = torch.stack(projected, dim=1)
        # [B, num_layers, hidden_dim]
        cognitive_tokens = self.output_proj(stacked)
        cognitive_tokens = self.output_norm(cognitive_tokens)
        return cognitive_tokens

    # --- Diagnostics ---

    def get_gate_values(self) -> dict[str, float]:
        """tanh(gate) for each layer — a map of 'how much the layer uses the channel'."""
        return {
            f"layer_{i}": torch.tanh(self.layer_gates[i]).item()
            for i in range(self._num_layers)
        }

    def get_gate_diagnostics(self) -> dict[str, dict[str, float]]:
        """Full gate diagnostics: raw, tanh, grad_norm — per layer."""
        result = {}
        for i in range(self._num_layers):
            gate = self.layer_gates[i]
            result[f"layer_{i}"] = {
                "raw": gate.item(),
                "tanh": torch.tanh(gate).item(),
                "grad_norm": gate.grad.norm().item() if gate.grad is not None else 0.0,
            }
        return result

    def get_active_layers(self, threshold: float = 0.1) -> list[int]:
        """Indices of layers with |tanh(gate)| > threshold."""
        return [
            i for i in range(self._num_layers)
            if abs(torch.tanh(self.layer_gates[i]).item()) > threshold
        ]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers
