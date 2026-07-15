"""ChronoAnchor — the ONE organ: goal-conditioned episodic memory + the agentic lifecycle.

The union of GoalAnchor (future) and Chronographer (past): the goal is not a second voice on
the residual but a PRIVILEGED INPUT of the memory organ — it conditions episode compression.
"""
import pytest
import torch

from daimon_agent import MetaAgent, Session
from daimon_agent.tools import Tool
from daimon_agent.action import AgentAction
from daimon_agent.policy import Policy
from daimon_voices import ChronoAnchor, ChronoAnchorConfig
from tests.test_cli import _fake_pipe


def _cfg(n_layers, pool="last"):
    return ChronoAnchorConfig(
        num_layers=n_layers,
        activation_layers=tuple(range(n_layers)),
        cross_attn_layers=tuple(range(n_layers)),
        encoder_bottleneck=16, K_memory_tokens=4, compressor_n_heads=2,
        memory_ca_bottleneck=8, memory_ca_num_heads=2,
        bank_max_episodes=8, novelty_threshold=2.0, eviction_mode="fifo",
        goal_pool=pool, goal_cond_k=2,
    )


def _pipe_and_anchor(fake_lm_factory, pool="last"):
    pipe = _fake_pipe(fake_lm_factory)
    n = pipe.model.config.num_hidden_layers
    anchor = ChronoAnchor(_cfg(n, pool))
    pipe.attach(anchor)
    return pipe, anchor


GOALS = ["Stay under $60 in total.", "Never touch the sources; report only.",
         "Only peer-reviewed sources may be cited."]


# ───────────────────────── goal channel ─────────────────────────

def test_goal_is_read_from_frozen_base_and_conditions_compressor(fake_lm_factory):
    """set_goal кодирует цель активациями базы (без отдельного якоря) и кондиционирует сжатие."""
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    n = pipe.model.config.num_hidden_layers
    vec = anchor.set_goal(GOALS[0])
    assert vec.shape == (n, pipe.model.config.hidden_size)     # [n_layers, hidden]
    assert anchor.goal_text == GOALS[0]
    assert anchor._goal_vec is not None
    assert anchor.compressor.goal_cond_k == 2                  # из конфига

    ids = pipe.tokenizer("episode", return_tensors="pt").input_ids
    with torch.no_grad():
        anchor.compressor.goal_attn.out_proj.weight.normal_(std=0.05)   # «обучили» goal-путь
    anchor.record_episode(ids, None, {}, force=True)
    goal_mem = anchor._current_memory.clone()
    anchor.clear_goal()
    anchor.clear_bank()
    anchor.record_episode(ids, None, {}, force=True)
    k, kc = anchor.config.K_memory_tokens, anchor.compressor.goal_cond_k
    plain = anchor._current_memory
    assert torch.allclose(plain[:k - kc], goal_mem[:k - kc])    # безусловные токены — те же
    assert not torch.allclose(plain[k - kc:], goal_mem[k - kc:])  # целевые — несут цель


def test_goal_pool_tokens_gives_per_token_latent(fake_lm_factory):
    pipe, anchor = _pipe_and_anchor(fake_lm_factory, pool="tokens")
    n = pipe.model.config.num_hidden_layers
    vec = anchor.encode_goal(GOALS[0])
    n_tok = pipe.tokenizer(GOALS[0], return_tensors="pt").input_ids.shape[1]
    assert vec.shape == (n * n_tok, pipe.model.config.hidden_size)


def test_goal_center_removes_constant_component(fake_lm_factory):
    """Центрирование (необходимый ингредиент рецепта): после fit центр вычитается из цели."""
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    raw = anchor.encode_goal(GOALS[0])
    center = anchor.fit_goal_center(GOALS)
    assert center.shape == raw.shape
    centered = anchor.encode_goal(GOALS[0])
    assert torch.allclose(centered, raw - center, atol=1e-5)
    assert centered.norm() < raw.norm()          # константа снята


# ───────────────────────── agentic lifecycle ─────────────────────────

class _ToolThenFinal(Policy):
    """Две ступени: сначала тул, потом финал (минимальная агентная сессия)."""

    def __init__(self):
        self.calls = 0

    def act(self, session):
        self.calls += 1
        if self.calls == 1:
            return AgentAction(kind="tool", tool="search", args={"q": "lamp"})
        return AgentAction(kind="final", content="done")


def test_step_hooks_record_one_episode_per_step(fake_lm_factory):
    """MetaAgent(step_hooks=[anchor]): банк накапливает по эпизоду на шаг — родной цикл."""
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    anchor.set_goal(GOALS[0])
    agent = MetaAgent(_ToolThenFinal(), step_hooks=[anchor], max_steps=4)
    agent.tools.register(Tool(name="search", description="search",
                              fn=lambda q: f"results for {q}", arg="q"))
    res = agent.run("find a desk lamp")
    assert res.answer == "done" and res.steps == 2
    assert len(anchor.bank) == 2                      # по эпизоду на шаг
    assert anchor._current_memory.size(0) == 2 * anchor.config.K_memory_tokens
    metas = [m.get("goal") for m in anchor.bank.metadata]
    assert metas == [GOALS[0], GOALS[0]]             # эпизоды помечены активной целью


def test_bank_persists_across_generations_but_clears_on_demand(fake_lm_factory):
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    anchor.set_goal(GOALS[1])
    s = Session()
    s.user("hi")
    anchor.on_step_end(1, s)
    assert len(anchor.bank) == 1
    anchor.on_pre_forward()                          # банк переживает форварды
    assert len(anchor.bank) == 1
    anchor.clear_bank()
    assert len(anchor.bank) == 0 and anchor._current_memory is None


def test_empty_session_records_nothing(fake_lm_factory):
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    anchor.on_step_end(1, Session())                 # пустая сессия — не пишем
    assert len(anchor.bank) == 0
    anchor.on_step_end(1, None)                      # без сессии — no-op
    assert len(anchor.bank) == 0


# ───────────────────────── checkpoint ─────────────────────────

def test_checkpoint_roundtrip_carries_goal_center(fake_lm_factory, tmp_path):
    """Центр цели ЕДЕТ В ЧЕКПОИНТЕ (свойство трейн-распределения, не евала — без ликеджа)."""
    pipe, anchor = _pipe_and_anchor(fake_lm_factory)
    center = anchor.fit_goal_center(GOALS)
    p = str(tmp_path / "chrono_anchor.pt")
    anchor.save_checkpoint(p)
    ck = torch.load(p, weights_only=False)
    assert ck["kind"] == "chrono_anchor" and ck["goal_center"] is not None

    a2 = ChronoAnchor.from_checkpoint(p)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(a2)
    assert a2.goal_center is not None
    assert torch.allclose(a2.goal_center.cpu(), center.cpu(), atol=1e-5)
    assert a2.config.goal_cond_k == 2 and a2.config.goal_pool == "last"
    a2.set_goal(GOALS[0])                            # рабочий после загрузки
    s = Session(); s.user("x")
    a2.on_step_end(1, s)
    assert len(a2.bank) == 1


def test_wrong_kind_rejected(tmp_path):
    p = str(tmp_path / "bad.pt")
    torch.save({"kind": "doubter", "encoder_state": {}}, p)
    with pytest.raises(ValueError, match="kind"):
        ChronoAnchor.from_checkpoint(p)
