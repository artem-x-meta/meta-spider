"""OpenAI-совместимый сервер — шов к внешним агентным рантаймам (Hermes Agent, τ²-bench).

Внешний харнесс ходит по HTTP и не может хостить модификатор из forward-прохода → сервим МЫ.
Тесты гоняют ядро сервера напрямую (без HTTP): сессии, латентная цель, банк эпизодов.
"""
import pytest
import torch

from daimon_agent.serve import (ChronoAnchorControl, LatentControl,
                              LlamaCppAnchorControl, DaimonServer, _TextSession,
                              from_pipeline)
from daimon_voices import ChronoAnchor, ChronoAnchorConfig
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


MSG = [{"role": "user", "content": "buy me a desk lamp"}]


def test_chat_is_openai_shaped(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    srv = from_pipeline(pipe, model_name="daimon")
    out = srv.chat(MSG)
    assert out["object"] == "chat.completion" and out["model"] == "daimon"
    assert out["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(out["choices"][0]["message"]["content"], str)
    assert srv.models()["data"][0]["id"] == "daimon"


def test_goal_lives_in_the_latent_not_in_the_messages(fake_lm_factory):
    """Цель ставится боковым каналом (/v1/meta/goal) и НЕ появляется в сообщениях."""
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    srv = from_pipeline(pipe, anchor)
    srv.set_goal("Stay under $60.", session="s1")
    assert anchor.goal_text == "Stay under $60."
    assert anchor._goal_vec is not None

    out = srv.chat(MSG, session="s1")
    assert out["meta"]["goal_set"] is True
    joined = " ".join(m["content"] for m in MSG)
    assert "60" not in joined                       # цели нет в промпте — только в латенте


def test_episodes_accumulate_per_turn(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    srv = from_pipeline(pipe, anchor)
    srv.set_goal("Read-only: never modify.", session="s1")
    srv.chat(MSG, session="s1")
    srv.chat(MSG + [{"role": "assistant", "content": "searching"},
                    {"role": "user", "content": "and?"}], session="s1")
    assert len(anchor.bank) == 2                    # по эпизоду на ход (цикл ChronoAnchor)
    assert srv.sessions["s1"].turns == 2


def test_sessions_isolate_goal_and_bank(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    srv = from_pipeline(pipe, anchor)
    srv.set_goal("Goal A", session="a")
    srv.chat(MSG, session="a")
    bank_a = len(anchor.bank)
    srv.set_goal("Goal B", session="b")             # переключение: банк чистится, цель — новая
    assert anchor.goal_text == "Goal B" and len(anchor.bank) == 0
    srv.chat(MSG, session="b")
    assert srv.sessions["a"].goal == "Goal A" and srv.sessions["b"].goal == "Goal B"
    assert bank_a == 1


def test_gain_and_reset(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    anchor = _anchor(pipe)
    srv = from_pipeline(pipe, anchor)
    srv.set_goal("G", session="s")
    srv.chat(MSG, session="s")
    assert srv.set_gain(0.5)["gain"] == 0.5 and anchor._gain == 0.5
    srv.reset(session="s")
    assert len(anchor.bank) == 0 and srv.sessions["s"].turns == 0


def test_knobs_need_a_control(fake_lm_factory):
    srv = from_pipeline(_fake_pipe(fake_lm_factory))              # без модификатора
    with pytest.raises(RuntimeError, match="LatentControl"):
        srv.set_goal("x")
    with pytest.raises(RuntimeError, match="LatentControl"):
        srv.set_gain(1.0)


# ───────────── llama.cpp-нога: якорь ставится при СПАВНЕ процесса (META_ANCHOR) ─────────────

def test_llamacpp_control_respawns_on_goal_and_gain():
    """CPU-нога: у C++ нет рантайм-сеттера — смена цели/гейна = новый процесс с META_ANCHOR."""
    spawns = []

    def factory(goal, gain):
        spawns.append((goal, gain))
        return type("B", (), {"generate": lambda self, p: type(
            "O", (), {"text": f"[goal={goal} gain={gain}] {p[:10]}", "signal": None})()})()

    ctl = LlamaCppAnchorControl(factory)
    assert isinstance(ctl, LatentControl)
    assert spawns == [(None, 1.0)]                       # первый спавн — без якоря
    ctl.set_goal("Never exceed $60.")
    assert spawns[-1] == ("Never exceed $60.", 1.0)      # цель — СЫРЬЁМ, при спавне
    ctl.set_gain(1.5)
    assert spawns[-1] == ("Never exceed $60.", 1.5)      # META_GAIN → респавн
    ctl.on_turn("transcript", 1)                         # эпизоды — на стороне C++-драйвера
    assert ctl.stats()["leg"] == "llama.cpp"

    srv = DaimonServer(ctl.backend, control=ctl, record_episodes=False)
    srv.set_goal("Read-only.", session="s")
    out = srv.chat([{"role": "user", "content": "hi"}], session="s")
    assert "goal=Read-only." in out["choices"][0]["message"]["content"]
    assert out["meta"]["goal_set"] is True


def test_text_session_shape():
    assert _TextSession("abc").render() == "abc"
