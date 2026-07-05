"""GoalAnchor (the goal-drift Watchdog leg): lifecycle, triggers, checkpoint contract.

Ported from archive/src/watchdog_v2_llama1b (v2.1). Key lifecycle contrast vs the Doubter:
the anchor is built from the GOAL text and PERSISTS across generations.
"""
import pytest
import torch

from meta_core import (
    BinaryTrigger, GoalAnchor, GoalAnchorConfig, LearnableTrigger,
    MetaSpiderConfig, MetaSpiderPipeline,
)
from tests.test_cli import _fake_pipe


def _mini_anchor(trigger="always", **kw):
    return GoalAnchor(GoalAnchorConfig(
        encoder_dim=32, encoder_num_blocks=1, encoder_num_heads=2,
        ca_bottleneck_dim=16, ca_num_heads=2,
        trigger=trigger, trigger_decision_layer=1, **kw))


# ───────────────────────── triggers ─────────────────────────

def test_binary_trigger_period():
    t = BinaryTrigger(K=3)
    fires = [t.decide() for _ in range(7)]
    assert fires == [True, False, False, True, False, False, True]
    t2 = BinaryTrigger(K=0)
    assert [t2.decide() for _ in range(3)] == [False, False, False]


def test_learnable_trigger_cooldown_and_force():
    t = LearnableTrigger(hidden_dim=8, K_min=2, K_max=4)
    h = torch.zeros(8)
    # cooldown: first K_min steps never fire
    assert t.decide(h) is False
    assert t.decide(h) is False
    # ≥ K_max forces a fire regardless of the MLP
    t.steps_since_last = 4
    assert t.decide(h) is True
    assert t.steps_since_last == 0


# ───────────────────────── lifecycle ─────────────────────────

def test_attach_eval_mode_and_anchor_persists(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor()
    pipe.attach(a)
    assert not a.ca_modules.training and not a.encoder.training  # A1 discipline

    a.set_anchor("never use the word the")
    assert a.buffer.is_filled
    # the Doubter clears its buffer in on_pre_forward; the GoalAnchor must NOT:
    a.on_pre_forward()
    assert a.buffer.is_filled, "the anchor must persist across generations"
    # generation works end-to-end with the anchor set
    out = pipe.generate("write about space", max_new_tokens=3, apply_chat_template=False)
    assert isinstance(out, str)
    assert a.buffer.is_filled, "generate must not consume the anchor"
    a.clear_anchor()
    assert not a.buffer.is_filled


def test_trigger_gates_injection(fake_lm_factory):
    """With K=0 (never fire) the CA must be a no-op; with always-on it must modify hiddens."""
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor(trigger="fixed", trigger_k=0)    # never fire
    pipe.attach(a)
    a.set_anchor("goal")
    ids = torch.randint(1, 50, (1, 6))
    with torch.no_grad():
        out_gated = pipe.model(input_ids=ids)
    pipe.detach(a)
    with torch.no_grad():
        out_clean = pipe.model(input_ids=ids)
    assert torch.allclose(out_gated, out_clean), "K=0 trigger must make CA a no-op"

    b = _mini_anchor(trigger="always")
    pipe.attach(b)
    b.set_anchor("goal")
    with torch.no_grad():
        out_inj = pipe.model(input_ids=ids)
    assert not torch.allclose(out_inj, out_clean), "always-on anchor must modify the forward"


def test_decision_layer_out_of_range_raises(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)          # 4 layers
    a = _mini_anchor()
    a.config.trigger_decision_layer = 99
    with pytest.raises(ValueError, match="out of range"):
        pipe.attach(a)


# ───────────────────────── checkpoint contract ─────────────────────────

def test_checkpoint_roundtrip_and_kind_guard(tmp_path, fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor(trigger="learnable")
    pipe.attach(a)
    path = str(tmp_path / "anchor.pt")
    a.save_checkpoint(path)

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["kind"] == "goal_anchor"
    assert ckpt["cross_attn_layers"] == [0, 1, 2, 3]
    assert ckpt["trigger_state"] is not None      # learnable trigger weights saved

    a2 = GoalAnchor.from_checkpoint(path)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(a2)                               # strict verify passes on same layout
    assert isinstance(a2.trigger, LearnableTrigger)

    # a Doubter checkpoint must be rejected loudly
    from meta_core import Doubter, DoubterConfig
    d = Doubter(DoubterConfig(encoder_type="selective", encoder_bottleneck=16,
                              ca_bottleneck_dim=16, ca_num_heads=2))
    pipe3 = _fake_pipe(fake_lm_factory)
    pipe3.attach(d)
    dpath = str(tmp_path / "doubter.pt")
    d.save_checkpoint(dpath)
    with pytest.raises(ValueError, match="goal_anchor"):
        GoalAnchor.from_checkpoint(dpath)


def test_checkpoint_layer_mismatch_raises(tmp_path, fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor()
    pipe.attach(a)
    path = str(tmp_path / "anchor.pt")
    a.save_checkpoint(path)

    from tests.conftest import FakeTokenizer
    m = fake_lm_factory(hidden_dim=64, num_layers=4)
    cfg = MetaSpiderConfig(model_name="fake", device="cpu", dtype="float32",
                           target_layers=[0, 1, 2, 3], cross_attn_layers=[0, 1])
    pipe2 = MetaSpiderPipeline.from_pretrained(cfg, model=m, tokenizer=FakeTokenizer())
    a2 = GoalAnchor.from_checkpoint(path)
    with pytest.raises(RuntimeError, match="mismatch"):
        pipe2.attach(a2)


# ───────────────────────── agentic mode: decide-then-detach ─────────────────────────

def test_agent_step_mode_off_until_armed(fake_lm_factory):
    """agent_step: injection is OFF by default; armed only inside the decision window."""
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor(trigger="agent_step")
    pipe.attach(a)
    a.set_anchor("stay on task")
    ids = torch.randint(1, 50, (1, 6))

    with torch.no_grad():
        out_idle = pipe.model(input_ids=ids)          # not armed → no-op
    a.on_step_start(1)
    with torch.no_grad():
        out_armed = pipe.model(input_ids=ids)         # decision window → injected
    a.on_step_end(1)
    with torch.no_grad():
        out_after = pipe.model(input_ids=ids)         # disarmed again
    pipe.detach(a)
    with torch.no_grad():
        out_clean = pipe.model(input_ids=ids)

    assert torch.allclose(out_idle, out_clean)
    assert not torch.allclose(out_armed, out_clean), "armed window must inject the anchor"
    assert torch.allclose(out_after, out_clean)
    # on_pre_forward (a new generate) must NOT re-enable injection in agent_step mode
    pipe.attach(a)
    a.on_pre_forward()
    assert a._step_fire is False


def test_metaagent_step_hooks_arm_disarm(fake_lm_factory):
    """MetaAgent arms the anchor exactly around policy.act (the decision window)."""
    from meta_agent import MetaAgent
    from meta_agent.action import AgentAction

    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor(trigger="agent_step")
    pipe.attach(a)
    a.set_anchor("goal")

    seen = []

    class SpyPolicy:
        def act(self, session):
            seen.append(a._step_fire)                  # inside the decision window
            return AgentAction(kind="final", content="done")

    agent = MetaAgent(SpyPolicy(), step_hooks=[a])
    result = agent.run("do the thing")
    assert result.answer == "done"
    assert seen == [True], "the anchor must be armed during policy.act"
    assert a._step_fire is False, "and disarmed right after the decision"


def test_set_gain_reaches_all_ca(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    a = _mini_anchor()
    pipe.attach(a)
    a.set_gain(1.5)
    assert all(m.gain == 1.5 for m in a.ca_modules.values())
    assert pipe.set_gain(0.5) >= 1                 # the pipeline-level mixer reaches it too
    assert all(m.gain == 0.5 for m in a.ca_modules.values())
