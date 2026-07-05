"""Drift sessions (agentic GoalAnchor v3 data): builder, off-task grading, drift-onset labels,
and the key integration — the EXISTING Trainer trains a GoalAnchor end-to-end (LM-CE) on FakeLM.
"""
import json

import torch

from meta_core import GoalAnchor, GoalAnchorConfig
from meta_loom.data import drift_sessions as DS
from tests.test_cli import _fake_pipe


# ───────────────────────── builder ─────────────────────────

def test_build_drift_sessions_shape_and_purity(fake_tokenizer):
    items = DS.build_drift_sessions(fake_tokenizer, per_family=3, seed=1)
    assert len(items) == 3 * len(DS.CONSTRAINT_FAMILIES)
    fams = {it["family"] for it in items}
    assert fams == set(DS.CONSTRAINT_FAMILIES)
    for it in items:
        assert it["goal_text"] and it["prompt"] and it["target"]
        # pure-latent default: the constraint must NOT leak into the prompt
        assert "CONSTRAINT" not in it["prompt"], "goal_in_prompt=False must keep the prompt clean"
    # and the text-arm variant puts it back
    with_goal = DS.build_drift_sessions(fake_tokenizer, per_family=1, goal_in_prompt=True, seed=1)
    assert all("CONSTRAINT" in it["prompt"] for it in with_goal)


def test_builder_deterministic(fake_tokenizer):
    a = DS.build_drift_sessions(fake_tokenizer, per_family=2, seed=7)
    b = DS.build_drift_sessions(fake_tokenizer, per_family=2, seed=7)
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]


# ───────────────────────── off-task grading ─────────────────────────

def test_is_off_task_families():
    assert DS.is_off_task("forbid_tool", {"forbidden_tool": "email"},
                          '<tool_call>{"name": "email", "arguments": {}}</tool_call>')
    assert not DS.is_off_task("forbid_tool", {"forbidden_tool": "email"},
                              '<tool_call>{"name": "search", "arguments": {}}</tool_call>')
    assert DS.is_off_task("budget", {"budget": 100},
                          '{"name": "buy", "arguments": {"item": "x", "price": 300}}')
    assert not DS.is_off_task("budget", {"budget": 100},
                              '{"name": "buy", "arguments": {"item": "x", "price": 40}}')
    assert DS.is_off_task("confirm_first", {"needs_confirm_before": "buy"},
                          '{"name": "buy", "arguments": {"item": "x", "price": 10}}')
    assert not DS.is_off_task("confirm_first", {"needs_confirm_before": "buy", "confirmed": True},
                              '{"name": "buy", "arguments": {"item": "x", "price": 10}}')
    assert DS.is_off_task("forbid_word", {"forbidden_word": "amazing"}, "This is Amazing value!")
    assert not DS.is_off_task("forbid_word", {"forbidden_word": "amazing"}, "This is great value.")


def test_drift_onset_labels_fire_before_offtask():
    """The v2 fix: labels mark the run-up to an off-task action, not a periodic clock."""
    off = [False, False, True, False, False, True]
    assert DS.drift_onset_labels(off, horizon=1) == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert DS.drift_onset_labels([False] * 4) == [0.0] * 4          # no drift → never fire


# ───────────────────────── Trainer trains GoalAnchor (LM-CE) ─────────────────────────

def test_trainer_trains_goal_anchor_end_to_end(fake_lm_factory, fake_tokenizer):
    """The v3 objective: injector learned via the SAME LM-CE Trainer as the Doubter
    (targets = correct next action), with GOAL activations as the anchor input."""
    from meta_loom.training.collector import DatasetSample
    from meta_loom.training.trainer import Trainer, TrainerConfig

    pipe = _fake_pipe(fake_lm_factory)
    anchor = GoalAnchor(GoalAnchorConfig(
        encoder_dim=32, encoder_num_blocks=1, encoder_num_heads=2,
        ca_bottleneck_dim=16, ca_num_heads=2,
        trigger="always",                      # training regime (v2 Step-4a): inject on every step
        trigger_decision_layer=1))
    pipe.attach(anchor)

    sessions = DS.build_drift_sessions(fake_tokenizer, per_family=2, seed=3)
    goal_acts = {}                             # collect ONCE per goal, reuse across samples
    samples, targets = [], []
    for it in sessions:
        if it["goal_text"] not in goal_acts:
            goal_acts[it["goal_text"]] = DS.collect_goal_activations(pipe, it["goal_text"])
        samples.append(DatasetSample(
            input_text=it["prompt"][:200], ground_truth=json.dumps(it["spec"]),
            activations=goal_acts[it["goal_text"]]))
        targets.append((it["target"], it["family"]))

    before = [p.detach().clone() for p in anchor.get_trainable_parameters()][:3]
    trainer = Trainer(anchor, pipe, TrainerConfig(
        epochs=1, batch_size=2, grad_accumulation=2,
        pretrain_projectors=False, max_seq_len=96))
    hist = trainer.train(samples, targets_by_sample=targets)

    assert len(hist["train_loss"]) == 1 and hist["train_loss"][0] > 0
    after = [p.detach() for p in anchor.get_trainable_parameters()][:3]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), \
        "the anchor's encoder/CA must receive gradient updates through the frozen base"
