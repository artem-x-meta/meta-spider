"""EpisodeCompressor: сжимает per-layer encoder tokens в K memory tokens.

Архитектура — attention pool с K learnable queries (обобщение MultiPositionProjector
из Phase 5 Variant C: там K=1 query на K=4 позиции, тут K=4 queries на 16 слоёв).

Input:  [batch, num_layers=16, bottleneck=384]
Output: [batch, K=4, hidden_dim=2048]

Тандем-v1 (механизм 1): опциональное GOAL-КОНДИЦИОНИРОВАНИЕ — последние goal_cond_k
queries получают добавку goal_proj(goal_vec) (цель из ког-токенов Якоря). goal_proj
zero-init → без цели и на старых чекпоинтах поведение побитово прежнее; двухкомпонентный
банк [K-k безусловных; k целевых] — страховка от пере-кондиционирования (дизайн тандема).
"""

from typing import Optional

import torch
import torch.nn as nn


class EpisodeCompressor(nn.Module):
    def __init__(
        self,
        num_layers: int = 16,
        bottleneck_dim: int = 384,
        hidden_dim: int = 2048,
        K: int = 4,
        n_heads: int = 4,
        goal_cond_k: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.goal_cond_k = K // 2 if goal_cond_k is None else goal_cond_k

        # K learnable queries в bottleneck space
        self.queries = nn.Parameter(torch.randn(K, bottleneck_dim) * 0.02)

        # Цель (hidden_dim, из ког-токенов Якоря) → добавка к последним goal_cond_k queries.
        # Zero-init: exact identity к безусловному компрессору, пока не обучено.
        self.goal_proj = nn.Linear(hidden_dim, bottleneck_dim)
        nn.init.zeros_(self.goal_proj.weight)
        nn.init.zeros_(self.goal_proj.bias)

        # PER-TOKEN путь (тандем-v1.7): целевые queries ВНИМАЮТ всем ког-токенам цели —
        # mean-pool убивает половину сигнала (числовой бюджет: R2 0.88 полные / 0.46 mean,
        # tandem-v16-numeric-budget-negative.md). Zero-init out_proj → no-op пока не обучено.
        self.goal_kv_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.goal_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=2, batch_first=True)
        nn.init.zeros_(self.goal_attn.out_proj.weight)
        nn.init.zeros_(self.goal_attn.out_proj.bias)

        # Cross-attention: queries × per-layer tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Project K compressed tokens из bottleneck в hidden_dim.
        # NB: убрали финальный LayerNorm и упростили stack — раньше was
        # [LayerNorm + Linear + GELU + Linear + LayerNorm], что давало
        # constant output (γ→0, β=const → memory tokens идентичны для всех
        # эпизодов, диагностировано в 99_diag_memory_collapse.py).
        # Теперь только Linear projection — позволяет variance проходить.
        self.out_proj = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, per_layer_tokens: torch.Tensor,
                goal_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Аргументы:
            per_layer_tokens: [batch, num_layers, bottleneck_dim]
            goal_vec: [hidden_dim] или [batch, hidden_dim] — латент цели (опционально)
        Возвращает:
            memory_tokens: [batch, K, hidden_dim]
        """
        batch_size = per_layer_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, K, bottleneck]
        if goal_vec is not None and self.goal_cond_k > 0:
            kc = self.goal_cond_k
            if goal_vec.dim() == 2:
                # per-token путь: [N, hidden] — целевые queries внимают всем ког-токенам цели
                kv = self.goal_kv_proj(goal_vec.to(self.goal_kv_proj.weight.dtype))
                kv = kv.unsqueeze(0).expand(batch_size, -1, -1)     # [batch, N, bottleneck]
                qc = q[:, self.K - kc:, :]
                att, _ = self.goal_attn(qc, kv, kv)
                q = torch.cat([q[:, :self.K - kc], qc + att], dim=1)
            else:
                # аддитивный путь: [hidden] (mean-pooled) — v1.2-v1.5
                gv = goal_vec.unsqueeze(0).expand(batch_size, -1)
                g = self.goal_proj(gv.to(self.goal_proj.weight.dtype))  # [batch, bottleneck]
                mask = torch.zeros(self.K, 1, device=q.device, dtype=q.dtype)
                mask[self.K - kc:] = 1.0
                q = q + mask.unsqueeze(0) * g.unsqueeze(1)
        # MultiheadAttention требует float-совместимости; per_layer_tokens приходит float32
        attn_out, _ = self.cross_attn(q, per_layer_tokens, per_layer_tokens)
        # attn_out: [batch, K, bottleneck]
        return self.out_proj(attn_out)  # [batch, K, hidden_dim]

    def get_query_norms(self) -> list:
        """Диагностика: норма каждой из K query — если все равны, queries неинформативны."""
        return [float(torch.norm(self.queries[k])) for k in range(self.K)]
