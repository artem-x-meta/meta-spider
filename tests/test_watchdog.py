"""Watchdog (read-only confidence sensor) — CPU smoke tests, без GPU/HF.

Проверяем: ConfidenceProbe.fit разделяет два кластера (uncertain vs certain); Watchdog прогоняет
активации через энкодер→пробу; threshold-гейт; save/load round-trip.
"""
import numpy as np
import torch

from daimon_voices.watchdog import ConfidenceProbe, Watchdog


def _clusters(n=40, hidden=8, num_cog=4, slot=0, seed=0):
    """Два кластера cog-токенов [num_cog,hidden]: label 1 (uncertain) сдвинут по слоту от label 0."""
    rng = np.random.default_rng(seed)
    cogs, labels = [], []
    for i in range(n):
        c = rng.normal(0, 0.3, size=(num_cog, hidden))
        lab = i % 2
        c[slot] += (2.0 if lab else -2.0)        # сигнал в нужном слоте
        cogs.append(c); labels.append(lab)
    return cogs, labels


def test_confidence_probe_fit_separates():
    cogs, labels = _clusters()
    probe = ConfidenceProbe.fit(cogs, labels, slot=0)
    # uncertain-подобный (+2 по слоту) → высокий P; certain-подобный (−2) → низкий
    hi = probe.proba(np.tile([[2.0]], (4, 8)))
    lo = probe.proba(np.tile([[-2.0]], (4, 8)))
    assert hi > 0.5 > lo, f"проба не разделяет: hi={hi} lo={lo}"


def test_probe_to_from_dict_roundtrip():
    cogs, labels = _clusters()
    probe = ConfidenceProbe.fit(cogs, labels, slot=0)
    d = probe.to_dict()
    probe2 = ConfidenceProbe.from_dict(d)
    x = np.tile([[1.5]], (4, 8))
    assert abs(probe.proba(x) - probe2.proba(x)) < 1e-9


class _FakeEncoder:
    """Возвращает cog [1, num_cog, hidden] = переданный numpy-массив (батч добавлен)."""
    def __call__(self, activation):
        return torch.tensor(np.asarray(activation, dtype=float)).unsqueeze(0)


def test_watchdog_score_and_gate(tmp_path):
    cogs, labels = _clusters()
    enc = _FakeEncoder()
    wd = Watchdog.fit(enc, cogs, labels, slot=0)
    # активация uncertain-кластера → uncertain; certain → нет
    unc = np.zeros((4, 8)); unc[0] += 2.0
    cer = np.zeros((4, 8)); cer[0] -= 2.0
    assert wd.score(unc) > 0.5 > wd.score(cer)
    assert wd.is_uncertain(unc, threshold=0.5)
    assert not wd.is_uncertain(cer, threshold=0.5)
    # save/load round-trip
    p = tmp_path / "wd.json"
    wd.save(str(p))
    wd2 = Watchdog.load(str(p), enc)
    assert abs(wd.score(unc) - wd2.score(unc)) < 1e-9


def test_watchdog_readonly_no_injection():
    """Watchdog не трогает модель: у него нет ca_modules/attach — это чистый сенсор."""
    cogs, labels = _clusters()
    wd = Watchdog.fit(_FakeEncoder(), cogs, labels, slot=0)
    assert not hasattr(wd, "ca_modules")
    assert hasattr(wd, "score") and hasattr(wd, "probe")
