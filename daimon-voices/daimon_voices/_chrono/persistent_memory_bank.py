"""PersistentMemoryBank: cross-session store для K memory tokens на эпизод.

Лёгкий контейнер, не nn.Module (без параметров; growing tensor + metadata list).

Write-gate: novelty-based фильтр (max cos sim к existing tokens < threshold) —
от bank-overflow в длинных сессиях (Test 3 выявил: bank без фильтра растёт в шум,
chrono деградирует к концу с 120+ memory tokens).

Overflow handling режимы:
  - "consolidate" (default — soft compression): найти 2 least-attended эпизода →
    attention-weighted merge → заменить парой одним merged эпизодом. Память НЕ
    теряется полностью, только становится менее подробной (sleep-consolidation метафора).
  - "centroid": вытеснить эпизод ближайший к centroid (least unique).
  - "fifo": вытеснить старейший (legacy).

Attention tracking (для consolidate): на каждом forward'е base модели каждый
memory CA слой через hook вызывает bank.record_attention(per_episode_avg).
Bank поддерживает running sum/count для compute среднего usage.
"""

from pathlib import Path
import torch
import torch.nn.functional as F


class PersistentMemoryBank:
    def __init__(
        self,
        max_episodes: int = 50,
        K_per_episode: int = 4,
        hidden_dim: int = 2048,
        device: str = "cuda",
        novelty_threshold: float = 0.95,
        eviction_mode: str = "consolidate",
    ):
        self.max_episodes = max_episodes
        self.K = K_per_episode
        self.hidden_dim = hidden_dim
        self.device = device
        self.novelty_threshold = novelty_threshold
        self.eviction_mode = eviction_mode

        self.tokens = torch.zeros(0, K_per_episode, hidden_dim, device=device)
        self.metadata = []

        # auto-sync callback: при successful add вызывается с get_all_tokens()
        self._auto_sync_target = None

        # Attention tracking per episode (для "consolidate" eviction).
        # running sum + count → mean attention weight per episode за всё время в bank.
        # Reset при merge — для новых эпизодов counter снова с нуля.
        self.attention_sum = torch.zeros(0, device=device)
        self.attention_count = torch.zeros(0, dtype=torch.long, device=device)

        # Статистика для диагностики
        self.stats = {
            "n_added": 0,
            "n_rejected_duplicate": 0,
            "n_evicted_centroid": 0,
            "n_evicted_fifo": 0,
            "n_consolidated": 0,
        }

    # --- Внутренние утилиты ---

    def _flat_view(self) -> torch.Tensor:
        """[N, K*hidden_dim] — каждый эпизод как один вектор для сравнения."""
        return self.tokens.reshape(len(self.metadata), -1)

    def _max_sim_to_bank(self, episode_tokens: torch.Tensor) -> float:
        """Max cos sim между новым эпизодом и каждым из существующих."""
        if len(self.metadata) == 0:
            return 0.0
        ep_flat = episode_tokens.reshape(-1).unsqueeze(0)  # [1, K*hidden]
        bank_flat = self._flat_view()                       # [N, K*hidden]
        sims = F.cosine_similarity(ep_flat, bank_flat, dim=-1)
        return float(sims.max())

    def _least_novel_index(self) -> int:
        """Индекс эпизода, ближайшего к centroid (наименее уникальный)."""
        bank_flat = self._flat_view()
        centroid = bank_flat.mean(dim=0, keepdim=True)
        dists = (bank_flat - centroid).norm(dim=-1)
        return int(dists.argmin())

    def _append(self, episode_tokens: torch.Tensor, meta: dict):
        self.tokens = torch.cat([self.tokens, episode_tokens.unsqueeze(0)], dim=0)
        self.metadata.append(meta)
        # Расширить attention tracking для нового эпизода
        self.attention_sum = torch.cat([
            self.attention_sum, torch.zeros(1, device=self.device),
        ])
        self.attention_count = torch.cat([
            self.attention_count, torch.zeros(1, dtype=torch.long, device=self.device),
        ])
        self.stats["n_added"] += 1

    def _drop_index(self, idx: int):
        keep = torch.ones(self.tokens.size(0), dtype=torch.bool, device=self.device)
        keep[idx] = False
        self.tokens = self.tokens[keep]
        self.attention_sum = self.attention_sum[keep]
        self.attention_count = self.attention_count[keep]
        self.metadata.pop(idx)

    # --- Attention tracking + consolidation ---

    def record_attention(self, per_episode_attn: torch.Tensor):
        """Update attention stats после одного memory CA forward'а.
        per_episode_attn: [N_episodes] — средняя attention weight per episode."""
        if per_episode_attn.size(0) != len(self.metadata):
            return  # mismatch — пропускаем (например после merge но до auto-sync)
        per_episode_attn = per_episode_attn.detach().to(self.device).float()
        self.attention_sum += per_episode_attn
        self.attention_count += 1

    def get_attention_mean(self) -> torch.Tensor:
        """Mean attention weight per episode (running average)."""
        return self.attention_sum / self.attention_count.clamp(min=1).float()

    def _least_attended_pair(self) -> tuple:
        """Индексы 2 наименее attended эпизодов (через current rolling mean).
        Если у эпизодов нет attention data (count=0) — они first candidates."""
        mean_attn = self.get_attention_mean()
        # Episodes без attention data → синтетически low score (берутся первыми на merge)
        no_data_penalty = torch.where(
            self.attention_count == 0,
            torch.full_like(mean_attn, -1.0),
            mean_attn,
        )
        sorted_idx = torch.argsort(no_data_penalty)
        return int(sorted_idx[0]), int(sorted_idx[1])

    def consolidate(self) -> bool:
        """Найти 2 least-attended → attention-weighted merge → replace в bank.

        Merge mechanics (v2 из chronographer-consolidation-via-merge.md):
          - weights = softmax of (attention_a, attention_b) → low attention → low weight,
            но **не ноль** (gist двух забытых эпизодов сохраняется как averaged signature)
          - merged_tokens = w_a × tokens[a] + w_b × tokens[b]
          - merged metadata = combined (subjects union, max significance)
          - drop a, b → append merged

        Returns True если consolidation реально произошла.
        """
        if len(self.metadata) < 2:
            return False
        idx_a, idx_b = self._least_attended_pair()
        if idx_a == idx_b:
            return False

        # Attention-weighted мерж — но с floor (softmax над raw, чтобы оба эпизода имели голос)
        mean_a = float(self.get_attention_mean()[idx_a])
        mean_b = float(self.get_attention_mean()[idx_b])
        # Если оба ≈ 0 (no data), berем 50/50. Иначе softmax по raw mean.
        if mean_a + mean_b < 1e-9:
            w_a = w_b = 0.5
        else:
            raw = torch.tensor([mean_a, mean_b])
            soft = torch.softmax(raw, dim=0)
            w_a, w_b = float(soft[0]), float(soft[1])

        merged_tokens = w_a * self.tokens[idx_a] + w_b * self.tokens[idx_b]
        meta_a = self.metadata[idx_a]
        meta_b = self.metadata[idx_b]
        merged_meta = {
            "merged": True,
            "subjects": list(set([meta_a.get("subject", "?"), meta_b.get("subject", "?")])),
            "probe_score": max(meta_a.get("probe_score", 0.5), meta_b.get("probe_score", 0.5)),
            "merge_history": (meta_a.get("merge_history", []) + [idx_a]
                              + meta_b.get("merge_history", []) + [idx_b])[:8],
            "n_merged_in": meta_a.get("n_merged_in", 1) + meta_b.get("n_merged_in", 1),
        }

        # drop indices в порядке убывания (чтобы оставшийся не сместился)
        i_hi, i_lo = (max(idx_a, idx_b), min(idx_a, idx_b))
        self._drop_index(i_hi)
        self._drop_index(i_lo)
        self._append(merged_tokens, merged_meta)
        self.stats["n_consolidated"] += 1
        return True

    # --- Public API ---

    def add(self, episode_tokens: torch.Tensor, meta: dict, force: bool = False) -> bool:
        """Добавить эпизод с novelty-фильтром.

        Returns:
            True если эпизод реально записан (либо append, либо после eviction)
            False если отвергнут как близкий дубль (novelty < threshold)

        force=True пропускает novelty-check (для tests/диагностики).
        """
        if episode_tokens.dim() == 3 and episode_tokens.size(0) == 1:
            episode_tokens = episode_tokens.squeeze(0)
        assert episode_tokens.shape == (self.K, self.hidden_dim), \
            f"Expected [{self.K}, {self.hidden_dim}], got {episode_tokens.shape}"
        episode_tokens = episode_tokens.detach().to(self.device)

        # 1. Novelty check (если не force)
        if not force and len(self.metadata) > 0:
            max_sim = self._max_sim_to_bank(episode_tokens)
            if max_sim >= self.novelty_threshold:
                self.stats["n_rejected_duplicate"] += 1
                return False

        # 2. Overflow handling
        if len(self.metadata) >= self.max_episodes:
            if self.eviction_mode == "consolidate":
                self.consolidate()  # soft compression: merge 2 least-attended → 1
            elif self.eviction_mode == "centroid":
                evict_idx = self._least_novel_index()
                self._drop_index(evict_idx)
                self.stats["n_evicted_centroid"] += 1
            else:  # "fifo"
                self._drop_index(0)
                self.stats["n_evicted_fifo"] += 1

        # 3. Append + auto-sync
        self._append(episode_tokens, meta)
        if self._auto_sync_target is not None:
            self._auto_sync_target(self.get_all_tokens())
        return True

    def _evict_fifo(self):
        """Legacy метод (для совместимости со старым кодом если где-то остался)."""
        if len(self.metadata) > 0:
            self._drop_index(0)
            self.stats["n_evicted_fifo"] += 1

    def get_all_tokens(self) -> torch.Tensor:
        """Flatten для CA injection: [N*K, hidden_dim]."""
        if self.tokens.size(0) == 0:
            return torch.zeros(0, self.hidden_dim, device=self.device)
        return self.tokens.reshape(-1, self.hidden_dim)

    def get_subset_tokens(self, indices: list) -> torch.Tensor:
        """[len(indices)*K, hidden_dim] для условий random_K / chronographer_K."""
        if len(indices) == 0:
            return torch.zeros(0, self.hidden_dim, device=self.device)
        sub = self.tokens[indices]  # [n, K, hidden_dim]
        return sub.reshape(-1, self.hidden_dim)

    def top_k_by_significance(self, K: int) -> list:
        """Top-K эпизодов по |probe_score - 0.5|."""
        sigs = [abs(m.get("probe_score", 0.5) - 0.5) for m in self.metadata]
        if len(sigs) <= K:
            return list(range(len(sigs)))
        return sorted(range(len(sigs)), key=lambda i: -sigs[i])[:K]

    def random_subset(self, K: int, rng=None) -> list:
        """Случайный subset длины K для baseline сравнения."""
        import numpy as np
        if rng is None:
            rng = np.random.default_rng()
        n = len(self.metadata)
        if n <= K:
            return list(range(n))
        return rng.choice(n, size=K, replace=False).tolist()

    def __len__(self):
        return len(self.metadata)

    def clear(self):
        self.tokens = torch.zeros(0, self.K, self.hidden_dim, device=self.device)
        self.metadata = []
        self.attention_sum = torch.zeros(0, device=self.device)
        self.attention_count = torch.zeros(0, dtype=torch.long, device=self.device)
        # stats не обнуляем — они кумулятивные для всей сессии model'и

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": self.tokens.cpu(),
            "metadata": self.metadata,
            "max_episodes": self.max_episodes,
            "K": self.K,
            "hidden_dim": self.hidden_dim,
            "novelty_threshold": self.novelty_threshold,
            "eviction_mode": self.eviction_mode,
            "stats": self.stats,
        }, path)

    @classmethod
    def load(cls, path, device: str = "cuda"):
        d = torch.load(path, weights_only=False)
        bank = cls(
            max_episodes=d["max_episodes"],
            K_per_episode=d["K"],
            hidden_dim=d["hidden_dim"],
            device=device,
            novelty_threshold=d.get("novelty_threshold", 0.95),
            eviction_mode=d.get("eviction_mode", "centroid"),
        )
        bank.tokens = d["tokens"].to(device)
        bank.metadata = d["metadata"]
        bank.stats = d.get("stats", bank.stats)
        return bank
