"""build-anchor (GoalAnchor factory): orchestration on FakeLM with an injected miner.

The teacher-mining step needs a real LM (valid Python that passes tests) + HF datasets, so it is
not unit-tested here; we inject `mine_fn` to return fixed self-distilled pairs and check the
train→save path (slice-train a GoalAnchor, write goal_anchor.pt + run.json) runs on CPU.
"""
import json

import torch

from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_loom.cli import build_anchor as ba


def _anchor_fake_pipe(fake_lm_factory):
    """Fake pipe whose CA lives on the TOP layers → a real slice cut (cut = min(CA)-1 = 1)."""
    from tests.conftest import FakeTokenizer
    m = fake_lm_factory(hidden_dim=64, num_layers=4)
    cfg = MetaAttentionConfig(model_name="fake", device="cpu", dtype="float32",
                           target_layers=[2, 3], cross_attn_layers=[2, 3])
    return MetaAttentionPipeline.from_pretrained(cfg, model=m, tokenizer=FakeTokenizer())


def _fixed_pairs(pipeline, *, families, n_specs, verbose=True):
    """Injected miner: a few deterministic self-distilled pairs (no generation / no network)."""
    spec_text = "TASK: add two numbers.\nREQUIREMENTS:\n1. name `f`.\n2. no print()."
    pairs = []
    for i in range(6):
        pairs.append({
            "messages": [{"role": "system", "content": "You are a precise coding assistant."},
                         {"role": "user", "content": f"Write f (case {i})."}],
            "target": "```python\ndef f(a, b):\n    return a + b\n```",
            "spec_text": spec_text, "step": i % 3, "task_id": i})
    return pairs


def test_build_anchor_stage_trains_and_saves(tmp_path, fake_lm_factory):
    pipe = _anchor_fake_pipe(fake_lm_factory)
    run_dir = tmp_path / "anchor_run"
    ckpt = ba.build_anchor_stage(
        str(run_dir), "fake", families=["func_name", "no_print"], mine_specs=6, epochs=1,
        slice_train=False, pipeline=pipe, mine_fn=_fixed_pairs, verbose=False)

    assert ckpt.exists() and ckpt.name == "goal_anchor.pt"
    c = torch.load(str(ckpt), weights_only=False)
    assert c["kind"] == "goal_anchor"
    assert c["cross_attn_layers"] == [2, 3]                 # slice cut = 1
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["voice"] == "goal_anchor"
    assert manifest["families"] == ["func_name", "no_print"]
    assert manifest["n_pairs"] == 6


def test_build_anchor_rejects_empty_mine(tmp_path, fake_lm_factory):
    pipe = _anchor_fake_pipe(fake_lm_factory)
    with __import__("pytest").raises(RuntimeError, match="no pairs"):
        ba.build_anchor_stage(str(tmp_path / "r"), "fake", mine_specs=1, epochs=1,
                              pipeline=pipe, mine_fn=lambda *a, **k: [], verbose=False)


def test_default_families_are_diverse():
    # 10 families across avoid/require/structural/naming/typing (diversity = transfer)
    assert len(ba.DEFAULT_FAMILIES) == 10
    assert "func_name" in ba.DEFAULT_FAMILIES and "type_hints" in ba.DEFAULT_FAMILIES
