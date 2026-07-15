"""AdaptiveMemoryGate: input-dependent gate для MemoryBottleneckCrossAttention.

Вместо обучаемого scalar gate (как в текущей версии) gate вычисляется per-token
из hidden state. Опционально — bank-size bias, позволяющий gate автоматически
ослабляться при росте bank'а (борется с overdose effect, выявленным в Test 3/8).

Аналог Phase 4 (input-dependent encoder gates) для memory CA pathway.
"""

import math
import torch
import torch.nn as nn


class AdaptiveMemoryGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        gate_hidden: int = 64,
        use_bank_size: bool = True,
        bank_size_scale_init: float = 0.0,
    ):
        """
        Args:
            hidden_dim: размерность current hidden state (вход для gate prediction)
            gate_hidden: размерность скрытого слоя gate_net (маленький MLP)
            use_bank_size: добавлять ли learnable bias от log(bank_size+1)
            bank_size_scale_init: начальное значение для bank_size scale (0 = не влияет на старте)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_bank_size = use_bank_size

        # Маленький MLP: hidden_state → scalar (raw gate logit, до tanh)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        # Инициализация: финальный bias = atanh(0.3) ≈ 0.31 — стартуем в той же
        # рабочей зоне, что и static gate_init=0.3
        with torch.no_grad():
            self.gate_net[-1].weight.zero_()
            self.gate_net[-1].bias.fill_(math.atanh(0.3))

        if use_bank_size:
            self.bank_size_scale = nn.Parameter(
                torch.tensor(bank_size_scale_init, dtype=torch.float32)
            )

    def forward(self, hidden_states: torch.Tensor, bank_size: int = 0) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_dim]
            bank_size: int (количество эпизодов в bank на текущий forward)
        Returns:
            gate_values: [batch, seq, 1] — per-token gate (в float32 для точности)
        """
        # Кастуем во float32 для стабильности (как gate parameter в базе)
        h = hidden_states.float()
        raw = self.gate_net(h).squeeze(-1)  # [batch, seq]

        if self.use_bank_size and bank_size > 0:
            bank_term = self.bank_size_scale * math.log(bank_size + 1)
            raw = raw + bank_term

        return torch.tanh(raw).unsqueeze(-1)  # [batch, seq, 1]

    def get_diagnostics(self, hidden_states: torch.Tensor, bank_size: int = 0) -> dict:
        """Диагностика: распределение gate values на данном вводе."""
        with torch.no_grad():
            g = self.forward(hidden_states, bank_size).squeeze(-1)  # [batch, seq]
            return {
                "mean": float(g.mean()),
                "std": float(g.std()),
                "min": float(g.min()),
                "max": float(g.max()),
                "bank_size": bank_size,
                "bank_size_scale": float(self.bank_size_scale) if self.use_bank_size else 0.0,
            }
