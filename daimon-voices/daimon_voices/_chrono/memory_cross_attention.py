"""MemoryBottleneckCrossAttention: fork BottleneckCrossAttention для dynamic bank.

Отличие от базового BottleneckCA:
- **Нет fixed `token_preference[num_tokens]`** — bank растёт/уменьшается между сессиями,
  fixed-size bias не имеет смысла. Attention сам определяет вес каждого memory token
  через softmax(Q·Kᵀ/√d_head) по содержимому.
- Memory tokens приходят как [n_memory, hidden_dim] (плоский bank), без batch dim;
  forward expand'ит до batch'а.
"""

import math
import torch
import torch.nn as nn

from daimon_voices._chrono.adaptive_memory_gate import AdaptiveMemoryGate


class MemoryBottleneckCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_init: float = 0.3,
        use_adaptive_gate: bool = False,
        adaptive_gate_hidden: int = 64,
        adaptive_use_bank_size: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads
        self.use_adaptive_gate = use_adaptive_gate
        assert bottleneck_dim % num_heads == 0, "bottleneck_dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.q_proj = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        # Static scalar gate (legacy). Если use_adaptive_gate=True — игнорируется в forward.
        self.gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))

        if use_adaptive_gate:
            self.adaptive_gate = AdaptiveMemoryGate(
                hidden_dim=hidden_dim,
                gate_hidden=adaptive_gate_hidden,
                use_bank_size=adaptive_use_bank_size,
            )
        else:
            self.adaptive_gate = None

        self._last_attn_weights = None
        self._init_weights()

    def _init_weights(self):
        for proj in [self.down_proj, self.q_proj, self.k_proj, self.v_proj, self.up_proj]:
            nn.init.normal_(proj.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor, memory_tokens: torch.Tensor,
                bank_size: int = 0) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] — текущий вход LLM слоя
            memory_tokens: [n_memory, hidden_dim] (плоский bank) ИЛИ [batch, n_memory, hidden_dim]
            bank_size: int — для adaptive gate, эпизодов в bank (=n_memory / K_per_episode).
                       Используется только если use_adaptive_gate=True и adaptive_use_bank_size=True.
        Returns:
            hidden_states + gated memory contribution
        """
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape

        # memory может прийти как 2D (плоский bank) — расширим до batch'а
        if memory_tokens.dim() == 2:
            memory_tokens = memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        n_memory = memory_tokens.size(1)

        h = self.norm(hidden_states)
        h_compressed = self.down_proj(h)

        Q = self.q_proj(h_compressed)  # [batch, seq_len, bottleneck_dim]
        K = self.k_proj(memory_tokens)  # [batch, n_memory, bottleneck_dim]
        V = self.v_proj(memory_tokens)  # [batch, n_memory, bottleneck_dim]

        # Multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_memory, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_memory, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # БЕЗ token_preference bias (см. docstring)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        self._last_attn_weights = attn_weights
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.bottleneck_dim)

        output = self.up_proj(attn_output)  # [batch, seq, hidden_dim]

        if self.use_adaptive_gate and self.adaptive_gate is not None:
            # gate_values: [batch, seq, 1] — per-token gating
            gate_values = self.adaptive_gate(hidden_states, bank_size=bank_size)
            gate_values = gate_values.to(output.dtype)
            return residual + gate_values * output
        else:
            # Legacy: scalar gate (как в исходной версии)
            gate_value = torch.tanh(self.gate).to(output.dtype)
            return residual + gate_value * output

    def get_gate_value(self) -> float:
        """Возвращает scalar gate (legacy). Для adaptive gate использовать
        adaptive_gate.get_diagnostics() с реальным input."""
        return torch.tanh(self.gate).item()

    def get_raw_gate_value(self) -> float:
        return self.gate.item()

    def get_gate_grad_norm(self) -> float:
        if self.gate.grad is not None:
            return self.gate.grad.norm().item()
        return 0.0
