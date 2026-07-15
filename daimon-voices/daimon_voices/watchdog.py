"""Watchdog — сенсор неуверенности на когнитивном токене (одна из «ног» daimon).

Читает когнитивный токен в точке решения (обученный энкодер + линейная проба → P(uncertain))
и ГЕЙТИТ этим скором действие. ГЛАВНОЕ применение — не read-only:

1. **Чинит инъекцию на длинной (кодовой) генерации.** Постоянная инъекция Doubter'а портит
   длинный вывод (−3 на ODEX-15: сигнал уровня решения шумит на каждом токене исполнения и
   компаундится — «модулятор РЕШЕНИЯ, не генерации»). Гейтинг лечит: инъекция ТОЛЬКО в окне
   решения (gain 1.5 → модель эмитит tool_call → код при gain 0 чистый) = **+1, 0 потерь** —
   точечная инъекция реабилитирована. Здесь реализован сенсор; гейтинг-петли — в харнессах
   (`lab/experiments/rag-coding/`); trigger-механика в core — см. GoalAnchor (та же идея).
2. Read-only гейтинг внешнего действия (docs-lookup / refuse / escalate) без инъекции вовсе:
   +1, 0 потерь на том же ODEX.

Валидация сенсора: cog-probe @ L32 slot, in-domain AUC ~0.715, cog > сырой активации
(`docs/results/qwen-14b/cogprobe-watchdog.md`, `injection-forms-comparison.md`).
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence


def _to_numpy(x):
    if hasattr(x, "detach"):            # torch.Tensor
        return x.detach().float().cpu().numpy()
    import numpy as np
    return np.asarray(x, dtype=float)


def _fit_logreg(Xs, y, C: float = 1.0):
    """Логистическая регрессия: sklearn если есть, иначе numpy-GD (без жёсткой зависимости)."""
    import numpy as np
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xs, y)
        return clf.coef_[0].astype(float), float(clf.intercept_[0])
    except Exception:
        # numpy fallback: полный батч GD с L2 (1/C)
        n, d = Xs.shape
        w = np.zeros(d); b = 0.0; lr = 0.1; lam = 1.0 / max(C, 1e-6)
        for _ in range(3000):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            gw = Xs.T @ g / n + lam * w / n
            gb = float(g.mean())
            w -= lr * gw; b -= lr * gb
        return w, b


class ConfidenceProbe:
    """Стандартизованная логистическая проба на ОДНОМ слоте когнитивного токена (или flatten):
    P(uncertain) = sigmoid(((x - mean) / scale) · coef + intercept)."""

    def __init__(self, slot: int, coef, mean, scale, intercept: float,
                 layer: Optional[int] = None, flatten: bool = False):
        import numpy as np
        self.slot = int(slot)
        self.flatten = bool(flatten)
        self.layer = layer
        self.coef = np.asarray(coef, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.intercept = float(intercept)

    def _feat(self, cog):
        """cog: [num_cog, hidden] (батч уже снят). Возвращает фичу: один слот или flatten."""
        import numpy as np
        cog = np.asarray(cog, dtype=float)
        return cog.reshape(-1) if self.flatten else cog[self.slot]

    def proba(self, cog) -> float:
        import numpy as np
        z = (self._feat(cog) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(float(z @ self.coef) + self.intercept))))

    @classmethod
    def fit(cls, cogs: Sequence[Any], labels: Sequence[float], slot: int = 0,
            flatten: bool = False, C: float = 1.0, layer: Optional[int] = None) -> "ConfidenceProbe":
        """Обучить пробу. cogs: список [num_cog, hidden]; labels: 0/1 (1 = uncertain/wrong)."""
        import numpy as np
        feats = []
        for c in cogs:
            c = _to_numpy(c)
            feats.append(c.reshape(-1) if flatten else c[slot])
        X = np.stack(feats).astype(float)
        y = np.asarray(labels, dtype=float)
        mean = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-8
        coef, intercept = _fit_logreg((X - mean) / scale, y, C)
        return cls(slot=slot, coef=coef, mean=mean, scale=scale, intercept=intercept,
                   layer=layer, flatten=flatten)

    def to_dict(self) -> dict:
        return {"slot": self.slot, "flatten": self.flatten, "layer": self.layer,
                "coef": self.coef.tolist(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceProbe":
        return cls(slot=d["slot"], coef=d["coef"], mean=d["mean"], scale=d["scale"],
                   intercept=d["intercept"], layer=d.get("layer"), flatten=d.get("flatten", False))


class Watchdog:
    """Энкодер (читает активации target-слоёв → когнитивный токен) + ConfidenceProbe.

    score()/is_uncertain() — READ-ONLY: инъекции нет, побочной порчи генерации нет. Решение о действии
    (lookup/refuse) принимает вызывающий код по скору. Энкодер можно взять из обученного Doubter
    (`watchdog = Watchdog(doubter.encoder, probe)`) — тот же сигнал, но без CA.
    """

    def __init__(self, encoder: Any, probe: ConfidenceProbe):
        self.encoder = encoder
        self.probe = probe

    def _cog(self, activation_list):
        cog = self.encoder(activation_list)          # [B, num_cog, hidden]
        return _to_numpy(cog)[0]                      # снимаем батч → [num_cog, hidden]

    def score(self, activation_list) -> float:
        """P(uncertain) в точке решения. activation_list — как для энкодера (target-слои)."""
        return self.probe.proba(self._cog(activation_list))

    def is_uncertain(self, activation_list, threshold: float = 0.5) -> bool:
        return self.score(activation_list) >= threshold

    @classmethod
    def fit(cls, encoder: Any, activation_lists: Sequence[Any], labels: Sequence[float],
            slot: int = 0, flatten: bool = False, C: float = 1.0) -> "Watchdog":
        """Собрать cog-токены энкодером по списку активаций и обучить пробу (1 = uncertain/wrong)."""
        cogs = [_to_numpy(encoder(a))[0] for a in activation_lists]
        probe = ConfidenceProbe.fit(cogs, labels, slot=slot, flatten=flatten, C=C)
        return cls(encoder, probe)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.probe.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, encoder: Any) -> "Watchdog":
        with open(path, encoding="utf-8") as f:
            return cls(encoder, ConfidenceProbe.from_dict(json.load(f)))
