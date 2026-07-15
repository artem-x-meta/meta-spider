"""EmotionalEncoder: адаптация ConcatEncoder (Phase 5 Variant B) под Llama-3.2-1B.

Per-layer projectors на (last+mean concat) → bottleneck токены.
Output идёт в EpisodeCompressor (он проектирует в hidden_dim), поэтому
output_proj/output_norm СЮДА не входят — это разделение ответственности:
encoder = extraction, compressor = aggregation + projection.
"""

import torch
import torch.nn as nn


class EmotionalEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 16,
        input_dim: int = 4096,
        bottleneck_dim: int = 384,
        gate_init: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, bottleneck_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

        self.layer_gates = nn.ParameterList([
            nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
            for _ in range(num_layers)
        ])

    def forward(self, activations) -> torch.Tensor:
        """
        Аргументы:
            activations: либо list of num_layers тензоров [batch, input_dim],
                         либо stacked tensor [batch, num_layers, input_dim].
        Возвращает:
            per_layer_tokens: [batch, num_layers, bottleneck_dim]
        """
        if isinstance(activations, torch.Tensor):
            assert activations.dim() == 3, f"Expected 3D, got {activations.shape}"
            assert activations.size(1) == self.num_layers, \
                f"Expected {self.num_layers} layers, got {activations.size(1)}"
            activations = [activations[:, i, :] for i in range(self.num_layers)]

        projected = []
        for act, projector, gate in zip(activations, self.layer_projectors, self.layer_gates):
            proj = projector(act)
            gate_value = torch.tanh(gate).to(proj.dtype)
            projected.append(gate_value * proj)
        return torch.stack(projected, dim=1)  # [batch, num_layers, bottleneck_dim]

    def get_encoder_gate_values(self) -> dict:
        return {
            f"layer_{i}": torch.tanh(self.layer_gates[i]).item()
            for i in range(self.num_layers)
        }

    def get_encoder_gate_diagnostics(self) -> dict:
        result = {}
        for i in range(self.num_layers):
            gate = self.layer_gates[i]
            result[f"layer_{i}"] = {
                "raw": gate.item(),
                "tanh": torch.tanh(gate).item(),
                "grad_norm": gate.grad.norm().item() if gate.grad is not None else 0.0,
            }
        return result

    def get_active_layers(self, threshold: float = 0.1) -> list:
        return [
            i for i in range(self.num_layers)
            if abs(torch.tanh(self.layer_gates[i]).item()) > threshold
        ]

    def load_pretrained_projectors(self, projector_states: dict):
        """Загрузить веса предобученных projectors из 00_pretrain_projectors.py."""
        for i in range(self.num_layers):
            if i in projector_states:
                self.layer_projectors[i].load_state_dict(projector_states[i])
