"""Chronographer (the episodic-memory leg): rolling-bank lifecycle, injection, checkpoint.

Ported from archive/src/phase_chronograph_llama1b. Third lifecycle type: the Doubter
rebuilds per prompt, the GoalAnchor encodes once, the Chronographer ACCUMULATES.
"""
import pytest
import torch

from daimon_voices import Chronographer, ChronographerConfig
from tests.test_cli import _fake_pipe


def _mini_cfg(n_layers):
    return ChronographerConfig(
        num_layers=n_layers,
        activation_layers=tuple(range(n_layers)),
        cross_attn_layers=tuple(range(n_layers)),
        encoder_bottleneck=16, K_memory_tokens=2, compressor_n_heads=2,
        memory_ca_bottleneck=8, memory_ca_num_heads=2,
        bank_max_episodes=3, novelty_threshold=2.0,  # >1 = novelty-гейт всегда пропускает
        eviction_mode="fifo",
    )


def _pipe_and_chrono(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    n_layers = pipe.model.config.num_hidden_layers
    ch = Chronographer(_mini_cfg(n_layers))
    pipe.attach(ch)
    return pipe, ch


def _ids(pipe, text="episode text"):
    enc = pipe.tokenizer(text, return_tensors="pt")
    return enc.input_ids, getattr(enc, "attention_mask", None)


# ───────────────────────── lifecycle ─────────────────────────

def test_attach_builds_modules_in_eval_mode(fake_lm_factory):
    _, ch = _pipe_and_chrono(fake_lm_factory)
    assert ch.encoder is not None and not ch.encoder.training
    assert not ch.compressor.training and not ch.memory_ca_layers.training
    assert len(ch.bank) == 0 and ch._current_memory is None


def test_record_episode_accumulates_and_syncs(fake_lm_factory):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    assert ch.record_episode(ids, mask, {"turn": 0}) is True
    assert len(ch.bank) == 1
    # auto-sync: bank.add обновил инъектируемую память
    assert ch._current_memory is not None
    assert ch._current_memory.size(0) == ch.config.K_memory_tokens
    ch.record_episode(*_ids(pipe, "second episode"), meta={"turn": 1}, force=True)
    assert len(ch.bank) == 2
    assert ch._current_memory.size(0) == 2 * ch.config.K_memory_tokens


def test_bank_persists_across_pre_forward(fake_lm_factory):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ch.record_episode(*_ids(pipe), meta={})
    ch.on_pre_forward()                       # Doubter чистит буфер; Хронограф — НЕТ
    assert len(ch.bank) == 1 and ch._current_memory is not None


def test_injection_changes_hidden_and_gain_zero_disables(fake_lm_factory):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)

    def out_hidden():
        with torch.no_grad():
            out = pipe.model(input_ids=ids, attention_mask=mask)
        return (out.logits if hasattr(out, "logits") else out[0]).clone()

    clean = out_hidden()                       # банк пуст → чистая база
    ch.record_episode(ids, mask, {})
    injected = out_hidden()                    # банк непуст → инъекция активна
    assert not torch.allclose(clean, injected)
    ch.set_gain(0.0)                           # фейдер в ноль = снова база
    assert torch.allclose(clean, out_hidden())
    ch.set_gain(1.0)
    ch.clear_bank()                            # пустой банк = снова база
    assert torch.allclose(clean, out_hidden())


def test_bank_eviction_cap(fake_lm_factory):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    for i in range(5):                         # cap = 3 (fifo)
        ch.record_episode(*_ids(pipe, f"ep {i}"), meta={"turn": i}, force=True)
    assert len(ch.bank) == ch.config.bank_max_episodes


def test_detach_removes_hooks(fake_lm_factory):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    ch.record_episode(ids, mask, {})

    def fwd():
        with torch.no_grad():
            out = pipe.model(input_ids=ids)
        return (out.logits if hasattr(out, "logits") else out).clone()

    injected = fwd()
    pipe.detach(ch)
    after = fwd()
    assert not torch.allclose(injected, after)  # хуки сняты → инъекции нет


# ───────────────────────── checkpoint contract ─────────────────────────

def test_checkpoint_roundtrip(fake_lm_factory, tmp_path):
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    p = str(tmp_path / "chrono.pt")
    ch.save_checkpoint(p)
    ck = torch.load(p, weights_only=False)
    assert ck["kind"] == "chronographer" and ck["format_version"] == "1.1"

    ch2 = Chronographer.from_checkpoint(p)     # конфиг из чекпоинта
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(ch2)
    # веса совпали
    for k in ch.encoder.state_dict():
        assert torch.allclose(ch.encoder.state_dict()[k], ch2.encoder.state_dict()[k])
    ids, mask = _ids(pipe2)
    assert ch2.record_episode(ids, mask, {}) is True


def test_legacy_checkpoint_format(fake_lm_factory, tmp_path):
    """content_pipeline.pt-стиль: голые state'ы без kind/config → конфиг явно."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    p = str(tmp_path / "legacy.pt")
    torch.save({
        "encoder_state": ch.encoder.state_dict(),
        "compressor_state": ch.compressor.state_dict(),
        "memory_ca_states": {k: v.state_dict() for k, v in ch.memory_ca_layers.items()},
    }, p)
    ch2 = Chronographer.from_checkpoint(p, config=ch.config)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(ch2)
    assert ch2.record_episode(*_ids(pipe2), meta={}) is True


def test_wrong_kind_rejected(tmp_path):
    p = str(tmp_path / "bad.pt")
    torch.save({"kind": "doubter", "encoder_state": {}}, p)
    with pytest.raises(ValueError, match="kind"):
        Chronographer.from_checkpoint(p)


# ───────────────────────── tandem-v1: goal conditioning ─────────────────────────

def test_goal_condition_zero_init_is_noop(fake_lm_factory):
    """Zero-init goal_proj: с целью и без — память побитово одинакова."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    ch.record_episode(ids, mask, {}, force=True)
    plain = ch._current_memory.clone()
    ch.clear_bank()
    ch.set_goal_condition(torch.randn(4, pipe.model.config.hidden_size))
    ch.record_episode(ids, mask, {}, force=True)
    assert torch.allclose(plain, ch._current_memory)


def test_goal_condition_changes_memory_when_trained(fake_lm_factory):
    """Ненулевой goal_proj: 1-dim цель (аддитивный путь) меняет ТОЛЬКО целевые токены."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    with torch.no_grad():
        ch.compressor.goal_proj.weight.normal_(std=0.05)
    ch.record_episode(ids, mask, {}, force=True)
    plain = ch._current_memory.clone()
    ch.clear_bank()
    ch.set_goal_condition(torch.randn(pipe.model.config.hidden_size))
    ch.record_episode(ids, mask, {}, force=True)
    cond = ch._current_memory
    k, kc = ch.config.K_memory_tokens, ch.compressor.goal_cond_k
    assert torch.allclose(plain[:k - kc], cond[:k - kc])          # безусловные не тронуты
    assert not torch.allclose(plain[k - kc:], cond[k - kc:])      # целевые — изменились
    ch.set_goal_condition(None)                                    # выключение
    ch.clear_bank()
    ch.record_episode(ids, mask, {}, force=True)
    assert torch.allclose(plain, ch._current_memory)


def test_pre_tandem_checkpoint_loads_without_goal_keys(fake_lm_factory, tmp_path):
    """Чекпоинт без goal_proj (до тандема) грузится; zero-init сохраняет поведение."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    p = str(tmp_path / "old.pt")
    state = {k: v for k, v in ch.compressor.state_dict().items() if "goal_proj" not in k}
    torch.save({
        "encoder_state": ch.encoder.state_dict(),
        "compressor_state": state,
        "memory_ca_states": {k: v.state_dict() for k, v in ch.memory_ca_layers.items()},
    }, p)
    ch2 = Chronographer.from_checkpoint(p, config=ch.config)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(ch2)
    assert torch.allclose(ch2.compressor.goal_proj.weight,
                          torch.zeros_like(ch2.compressor.goal_proj.weight))
    assert ch2.record_episode(*_ids(pipe2), meta={}) is True


def test_goal_condition_per_token_path(fake_lm_factory):
    """[N, hidden]-цель → attention-путь: zero-init = no-op; после «обучения» меняет
    ТОЛЬКО целевые токены; 1-dim путь не задет."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    hid = pipe.model.config.hidden_size
    ch.record_episode(ids, mask, {}, force=True)
    plain = ch._current_memory.clone()
    goal_tokens = torch.randn(6, hid)
    ch.clear_bank()
    ch.set_goal_condition(goal_tokens)                 # 2-dim → как есть (не пулится)
    assert ch._goal_vec.dim() == 2
    ch.record_episode(ids, mask, {}, force=True)
    assert torch.allclose(plain, ch._current_memory)   # zero-init out_proj = no-op
    with torch.no_grad():
        ch.compressor.goal_attn.out_proj.weight.normal_(std=0.05)
    ch.clear_bank()
    ch.record_episode(ids, mask, {}, force=True)
    cond = ch._current_memory
    k, kc = ch.config.K_memory_tokens, ch.compressor.goal_cond_k
    assert torch.allclose(plain[:k - kc], cond[:k - kc])
    assert not torch.allclose(plain[k - kc:], cond[k - kc:])


def test_record_episode_trainable_keeps_grad(fake_lm_factory):
    """trainable=True: у памяти живой граф до компрессора (иначе — отвязана)."""
    pipe, ch = _pipe_and_chrono(fake_lm_factory)
    ids, mask = _ids(pipe)
    ch.record_episode(ids, mask, {}, force=True)
    assert not ch._current_memory.requires_grad
    ch.clear_bank()
    ch.record_episode(ids, mask, {}, force=True, trainable=True)
    assert ch._current_memory.requires_grad
    ch._current_memory.sum().backward()
    assert ch.compressor.queries.grad is not None
