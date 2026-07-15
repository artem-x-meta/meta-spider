"""ChronoAnchorTrainer — рецепт обучения органа памяти (CPU, FakeLM).

Проверяем не «сходится ли лосс» (на фейке это ни о чём), а ИНГРЕДИЕНТЫ рецепта:
центр цели фитится по ТРЕЙН-целям и едет в чекпоинт; лосс взвешен на токенах решения;
учится только обвязка (база заморожена); эпизоды пишутся с живым градом.
"""
import torch

from daimon_voices import ChronoAnchor, ChronoAnchorConfig
from daimon_loom.training import ChronoAnchorSample, ChronoAnchorTrainer
from tests.test_cli import _fake_pipe


def _anchor(pipe):
    n = pipe.model.config.num_hidden_layers
    a = ChronoAnchor(ChronoAnchorConfig(
        num_layers=n, activation_layers=tuple(range(n)), cross_attn_layers=tuple(range(n)),
        encoder_bottleneck=16, K_memory_tokens=4, compressor_n_heads=2,
        memory_ca_bottleneck=8, memory_ca_num_heads=2,
        bank_max_episodes=8, novelty_threshold=2.0, eviction_mode="fifo", goal_cond_k=2))
    pipe.attach(a)
    return a


def _samples():
    return [
        ChronoAnchorSample(
            goal="Stay under $60 in total.",
            episodes=["user: find a lamp", "tool: three lamps found"],
            prompt="Which one? [1] premium $150 [2] standard $45",
            gold="The standard edition at $45.",
            decision_markers=["standard", "$45"]),
        ChronoAnchorSample(
            goal="Only the top tier will do; cost is irrelevant.",
            episodes=["user: find a lamp", "tool: three lamps found"],
            prompt="Which one? [1] premium $150 [2] standard $45",
            gold="The PREMIUM edition at $150.",
            decision_markers=["PREMIUM", "$150"]),
    ]


def test_fit_learns_center_and_trains_only_the_organ(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    base_before = [p.detach().clone() for p in pipe.model.parameters()]
    ca_before = [p.detach().clone() for p in anchor.memory_ca_layers.parameters()]

    logs = []
    losses = ChronoAnchorTrainer(pipe, anchor).fit(
        _samples(), epochs=2, lr=1e-3, log_every=1, on_log=logs.append)

    assert len(losses) == 4                                   # 2 сэмпла × 2 эпохи
    assert anchor.goal_center is not None                     # ингредиент 1: центр зафитен
    assert any("goal center fitted over 2" in m for m in logs)
    for p, before in zip(pipe.model.parameters(), base_before):
        assert torch.allclose(p, before)                      # база ЗАМОРОЖЕНА
    assert any(not torch.allclose(p, b)
               for p, b in zip(anchor.memory_ca_layers.parameters(), ca_before))  # орган учится
    assert not anchor.compressor.training                     # после fit — eval-режим
    assert len(anchor.bank) == 0                              # банк вычищен


def test_decision_tokens_get_the_weight(fake_lm_factory):
    """Ингредиент 2: токены решения весят ×5, остальные ×1."""
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    tr = ChronoAnchorTrainer(pipe, anchor, decision_weight=5.0)
    gold = "The standard edition at $45."
    n = tr._tok_ids(gold, special=False).shape[1]
    w = tr._decision_weights(gold, n, ["standard", "$45"])
    assert w.max().item() == 5.0 and w.min().item() == 1.0
    assert (w == 5.0).sum().item() >= 2                        # оба маркера покрыты
    flat = tr._decision_weights(gold, n, [])                   # без маркеров — равномерно
    assert torch.allclose(flat, torch.ones(n))


def test_step_loss_backprops_into_the_compressor(fake_lm_factory):
    """Град доходит до целевого пути компрессора — цепь «цель → память → решение» жива."""
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    tr = ChronoAnchorTrainer(pipe, anchor)
    anchor.fit_goal_center([s.goal for s in _samples()])
    loss = tr.step_loss(_samples()[0])
    loss.backward()
    assert anchor.compressor.queries.grad is not None
    assert anchor.compressor.goal_attn.in_proj_weight.grad is not None


def test_trained_organ_roundtrips_with_its_center(fake_lm_factory, tmp_path):
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    ChronoAnchorTrainer(pipe, anchor).fit(_samples(), epochs=1, lr=1e-3, log_every=0)
    p = str(tmp_path / "ca.pt")
    anchor.save_checkpoint(p)

    a2 = ChronoAnchor.from_checkpoint(p)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(a2)
    assert torch.allclose(a2.goal_center.cpu(), anchor.goal_center.cpu(), atol=1e-5)
    for k, v in anchor.compressor.state_dict().items():
        assert torch.allclose(v, a2.compressor.state_dict()[k], atol=1e-6)
