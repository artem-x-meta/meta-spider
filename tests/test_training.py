"""Level 3 smoke tests: losses + collector + Trainer.

Use FakeLM + FakeTokenizer from conftest.py.
"""

import random
import pytest
import torch

from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_voices import Doubter, DoubterConfig
from daimon_loom import ActivationDatasetCollector, DatasetSample, Trainer, TrainerConfig, build_correction_target, build_target_by_action, make_labels_with_prompt_mask


# ============================================================
# losses.py — target builders
# ============================================================


def test_build_correction_target_confirm():
    """pass1_correct=True → confirm action + a copy of ground_truth."""
    target, action = build_correction_target("B) 4 Hz", pass1_correct=True)
    assert action == "confirm"
    assert target == " B) 4 Hz"


def test_build_correction_target_correct():
    """pass1_correct=False + ratio=1.0 → always correct."""
    rng = random.Random(42)
    target, action = build_correction_target(
        "B) 4 Hz", pass1_correct=False,
        correction_ratio=1.0, rng=rng,
    )
    assert action == "correct"
    assert "B) 4 Hz" in target
    assert "Wait" in target


def test_build_correction_target_refuse():
    """pass1_correct=False + ratio=0.0 → always refuse."""
    rng = random.Random(42)
    target, action = build_correction_target(
        "B) 4 Hz", pass1_correct=False,
        correction_ratio=0.0, rng=rng,
    )
    assert action == "refuse"
    assert "not confident" in target.lower()


def test_build_agentic_target_code_when_confident():
    """pass1_correct=True → 'code' action: start a code block, NOT a tool call."""
    from daimon_loom import build_agentic_target

    target, action = build_agentic_target("how to parse a date", pass1_correct=True)
    assert action == "code"
    assert "```python" in target
    assert "tool_call" not in target


def test_build_agentic_target_lookup_when_uncertain_is_parseable():
    """pass1 wrong → 'lookup' action emitting a tool call the agentic harness can parse."""
    import json
    import re

    from daimon_loom import build_agentic_target

    target, action = build_agentic_target("strftime milliseconds format", pass1_correct=False)
    assert action == "lookup"
    # same regex the native ODEX harness uses to detect a docs_lookup call
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", target, re.S)
    assert m is not None, target
    call = json.loads(m.group(1))
    assert call["name"] == "docs_lookup"
    assert call["arguments"]["query"] == "strftime milliseconds format"


def test_build_agentic_target_lookup_query_is_json_safe():
    """A query with quotes/newlines/backslashes still yields valid JSON (no broken tool call)."""
    import json
    import re

    from daimon_loom import build_agentic_target

    target, _ = build_agentic_target('save to "C:\\tmp"\nwith json', pass1_correct=False)
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", target, re.S)
    assert m is not None
    json.loads(m.group(1))  # must not raise


def test_build_target_by_action_explicit():
    """build_target_by_action creates a target from an explicit action type."""
    assert build_target_by_action("B) 4 Hz", "confirm") == " B) 4 Hz"
    assert "Wait" in build_target_by_action("B) 4 Hz", "correct")
    assert "not confident" in build_target_by_action("B) 4 Hz", "refuse").lower()
    with pytest.raises(ValueError):
        build_target_by_action("X", "unknown")  # type: ignore


def test_make_labels_with_prompt_mask():
    """labels: the first prompt_len positions → -100, pad → -100."""
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 0, 0]])  # 0 = pad
    labels = make_labels_with_prompt_mask(input_ids, prompt_len=3, pad_token_id=0)
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100
    assert labels[0, 2] == -100
    assert labels[0, 3] == 8     # target start
    assert labels[0, 4] == 9
    assert labels[0, 5] == -100  # padding
    assert labels[0, 6] == -100


# ============================================================
# ActivationDatasetCollector
# ============================================================


def test_collector_collects_samples(fake_lm, fake_tokenizer):
    """ActivationDatasetCollector.collect returns a list[DatasetSample] with activations."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    collector = ActivationDatasetCollector(pipeline, max_new_tokens=2)

    questions = ["What is 2+2?", "Capital of France?"]
    truths = ["4", "Paris"]
    samples = collector.collect(questions, truths, verbose=False)

    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, DatasetSample)
        assert s.input_text != ""
        assert s.ground_truth in truths
        assert set(s.activations.keys()) == {0, 1, 2, 3}
        for act in s.activations.values():
            assert act.shape == (1, 64) or act.shape == (64,)


def test_collector_save_load_round_trip(tmp_path, fake_lm, fake_tokenizer):
    """save → load returns identical samples."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    collector = ActivationDatasetCollector(pipeline, max_new_tokens=2)
    samples = collector.collect(["Hello"], ["world"], verbose=False)

    path = tmp_path / "ds.pt"
    ActivationDatasetCollector.save(samples, str(path))
    loaded = ActivationDatasetCollector.load(str(path))

    assert len(loaded) == 1
    assert loaded[0].input_text == samples[0].input_text
    assert loaded[0].ground_truth == samples[0].ground_truth
    for k in samples[0].activations:
        assert torch.allclose(loaded[0].activations[k], samples[0].activations[k])


# ============================================================
# Trainer — optimizer + one train step
# ============================================================


def _build_pipeline_with_doubter(fake_lm, fake_tokenizer):
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    return pipeline, doubter


def _make_fake_samples(n: int = 3) -> list[DatasetSample]:
    samples = []
    for i in range(n):
        samples.append(DatasetSample(
            input_text=f"Q{i}",
            ground_truth=f"A{i}",
            activations={k: torch.randn(64) for k in range(4)},
            pass1_text="",
            pass1_correct=(i % 2 == 0),
        ))
    return samples


def test_trainer_build_optimizer_5_groups(fake_lm, fake_tokenizer):
    """build_optimizer creates 5 param groups (encoder, CA, enc_gates, CA gates, prefs)."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(epochs=1, batch_size=1))
    optimizer = trainer.build_optimizer()

    # SelectiveEncoder has layer_gates → 5 groups
    assert len(optimizer.param_groups) == 5
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    # First 2 groups — base LR; the next 3 — ×5
    assert lrs[0] == lrs[1] == 2e-4
    assert lrs[2] == lrs[3] == lrs[4] == 2e-4 * 5.0


def test_trainer_train_step_produces_loss(fake_lm, fake_tokenizer):
    """_train_step returns a scalar loss."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=1, batch_size=2, grad_accumulation=1, pretrain_projectors=False,
    ))
    samples = _make_fake_samples(2)
    targets = [(" A0", "confirm"), (" A1", "confirm")]
    loss = trainer._train_step(samples, targets)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_trainer_train_full_loop(fake_lm, fake_tokenizer):
    """trainer.train runs several epochs, returns history."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=2, batch_size=2, grad_accumulation=1,
        pretrain_projectors=False, warmup_ratio=0.0,
    ))
    samples = _make_fake_samples(4)
    history = trainer.train(samples)
    assert len(history["train_loss"]) == 2
    assert all(l > 0 for l in history["train_loss"])
    assert len(history["ca_gate_map"]) == 2


def test_trainer_with_validation_and_early_stop(fake_lm, fake_tokenizer):
    """Validation + early stop work."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=3, batch_size=2, grad_accumulation=1,
        pretrain_projectors=False, warmup_ratio=0.0, early_stop_patience=1,
    ))
    samples = _make_fake_samples(4)
    val_samples = _make_fake_samples(2)
    history = trainer.train(samples, val_samples=val_samples)
    assert len(history["val_loss"]) >= 1
    assert "best_val_loss" in history


def test_trainer_pretrain_projectors(fake_lm, fake_tokenizer):
    """pretrain_projectors returns per-layer accuracy."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=1, pretrain_epochs=2, pretrain_batch_size=2,
    ))
    samples = _make_fake_samples(8)
    results = trainer.pretrain_projectors(samples, verbose=False)
    assert len(results) == 4  # 4 layers
    for acc in results.values():
        assert 0.0 <= acc <= 1.0


def test_trainer_with_transformer_encoder_skips_pretrain(fake_lm, fake_tokenizer):
    """pretrain_projectors does not fail on a TransformerEncoder (skip)."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="transformer",
        transformer_encoder_dim=32, transformer_num_blocks=1, transformer_num_heads=2,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    trainer = Trainer(doubter, pipeline, TrainerConfig(epochs=1))
    samples = _make_fake_samples(2)
    results = trainer.pretrain_projectors(samples, verbose=False)
    assert results == {}  # skipped


def test_trainer_checkpoint_round_trip(tmp_path, fake_lm, fake_tokenizer):
    """save_checkpoint + load_checkpoint restores encoder + CA weights."""
    pipeline, doubter = _build_pipeline_with_doubter(fake_lm, fake_tokenizer)
    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=1, batch_size=2, grad_accumulation=1,
        pretrain_projectors=False, warmup_ratio=0.0,
    ))
    # One train step so the weights change
    samples = _make_fake_samples(2)
    trainer.train(samples)
    snapshot_encoder = {k: v.clone() for k, v in doubter.encoder.state_dict().items()}

    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path), epoch=1)

    # Zero out the weights to confirm that load restores them
    for p in doubter.encoder.parameters():
        p.data.zero_()
    trainer.load_checkpoint(str(path))
    for k, v in doubter.encoder.state_dict().items():
        assert torch.allclose(v, snapshot_encoder[k]), f"encoder param {k} not restored"


# ============================================================
# Batched activation collection — equivalence with one-by-one
# ============================================================


def _make_clean_pipeline(fake_lm, fake_tokenizer):
    from meta_attention import MetaAttentionConfig, MetaAttentionPipeline

    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    return MetaAttentionPipeline.from_pretrained(
        cfg, model=fake_lm, tokenizer=fake_tokenizer,
    )


def test_batched_collect_equals_single(fake_lm, fake_tokenizer):
    """Activations and answers from batched collection are identical to one-by-one.

    Questions of different lengths → real left-padding in the batch. FakeLM is
    position-independent, so with correct indexing of the last
    token the result should match bit-for-bit; with right-padding
    the last position would be a pad token and the activations would diverge.
    """
    import torch

    from daimon_loom import ActivationDatasetCollector

    questions = ["short", "a much longer question text", "mid length q", "x"]
    truths = ["a", "b", "c", "d"]

    pipe = _make_clean_pipeline(fake_lm, fake_tokenizer)

    single = ActivationDatasetCollector(pipe, max_new_tokens=3, batch_size=1)
    samples_single = single.collect(questions, truths, verbose=False)

    batched = ActivationDatasetCollector(pipe, max_new_tokens=3, batch_size=4)
    samples_batched = batched.collect(questions, truths, verbose=False)

    assert len(samples_single) == len(samples_batched) == 4
    for s1, s2 in zip(samples_single, samples_batched):
        assert s1.input_text == s2.input_text
        assert s1.activations.keys() == s2.activations.keys()
        for layer in s1.activations:
            assert s1.activations[layer].shape == s2.activations[layer].shape
            assert torch.allclose(
                s1.activations[layer], s2.activations[layer], atol=1e-5,
            ), f"activations mismatch at layer {layer}"


def test_batched_collect_restores_padding_side(fake_lm, fake_tokenizer):
    """The collector temporarily sets left-padding and restores the original side."""
    from daimon_loom import ActivationDatasetCollector

    fake_tokenizer.padding_side = "right"
    pipe = _make_clean_pipeline(fake_lm, fake_tokenizer)
    coll = ActivationDatasetCollector(pipe, max_new_tokens=2, batch_size=2)
    coll.collect(["q1", "q2 longer"], ["a", "b"], verbose=False)
    assert fake_tokenizer.padding_side == "right"


def test_default_check_handles_markdown_and_aliases():
    """Canary v8 lesson: chat models bold their answers (**Karl Marx**), a naive
    substring doesn't match; the check must go over all aliases."""
    from daimon_loom import ActivationDatasetCollector

    check = ActivationDatasetCollector._default_check
    pred = "The Communist Manifesto was written by **Karl Marx** and **Friedrich Engels**."
    assert check(pred, ["Karl Marx and Friedrich Engels"])
    assert check(pred, ["wrong answer", "Karl Marx and Friedrich Engels"])
    assert not check(pred, ["Adam Smith"])
    assert not check("", ["anything"])


def test_collect_with_alias_lists(fake_lm, fake_tokenizer):
    """ground_truths as list[list[str]]: canonical in ground_truth, all of them in aliases."""
    from daimon_loom import ActivationDatasetCollector

    pipe = _make_clean_pipeline(fake_lm, fake_tokenizer)
    coll = ActivationDatasetCollector(pipe, max_new_tokens=2, batch_size=2)
    samples = coll.collect(
        ["q one", "q two"],
        [["Alpha", "alpha variant"], ["Beta"]],
        verbose=False,
    )
    assert samples[0].ground_truth == "Alpha"
    assert samples[0].aliases == ["Alpha", "alpha variant"]
    assert samples[1].aliases == ["Beta"]


def test_dataset_save_load_aliases_roundtrip(tmp_path, fake_lm, fake_tokenizer):
    """aliases survive save/load; format 1.0 without aliases reads back (None)."""
    from daimon_loom import ActivationDatasetCollector

    pipe = _make_clean_pipeline(fake_lm, fake_tokenizer)
    coll = ActivationDatasetCollector(pipe, max_new_tokens=2, batch_size=2)
    samples = coll.collect(["q1"], [["A", "B"]], verbose=False)

    p = tmp_path / "ds.pt"
    ActivationDatasetCollector.save(samples, str(p))
    loaded = ActivationDatasetCollector.load(str(p))
    assert loaded[0].aliases == ["A", "B"]
