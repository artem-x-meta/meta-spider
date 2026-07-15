"""v0.3.1 arbiter-audit fixes: eval-mode at attach, strict checkpoint load, EOS supervision,
loud misconfig guards, MCQ letter extraction, tool-call format detection, abstain affordance.
"""
import json

import pytest
import torch

from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_voices import Doubter, DoubterConfig
from meta_attention.buffer import ReflexionBuffer
from meta_attention.reader import ActivationCollector
from tests.test_cli import _fake_pipe


def _mini_doubter():
    return Doubter(DoubterConfig(encoder_type="selective", encoder_bottleneck=16,
                                 ca_bottleneck_dim=16, ca_num_heads=2))


# ───────────────────── A1: inference is eval-mode (dropout OFF) ─────────────────────

def test_attach_sets_eval_mode(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    d = _mini_doubter()
    pipe.attach(d)
    assert not d.ca_modules.training, "CA modules must be in eval mode after attach (dropout off)"
    assert not d.encoder.training, "encoder must be in eval mode after attach"
    # the Trainer's epoch loop can still flip them back:
    d.ca_modules.train()
    assert d.ca_modules.training


def test_reattach_sets_eval_mode(fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)
    d = _mini_doubter()
    pipe.attach(d)
    d.ca_modules.train()               # simulate training having flipped the mode
    pipe.detach(d)
    pipe.attach(d)                     # the re-attach path (BaselineComparison detach→attach)
    assert not d.ca_modules.training


# ───────────────────── A2: strict checkpoint↔pipeline layer match ─────────────────────

def test_checkpoint_saves_layers_and_roundtrips(tmp_path, fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)          # layers [0,1,2,3]
    d = _mini_doubter()
    pipe.attach(d)
    path = str(tmp_path / "d.pt")
    d.save_checkpoint(path)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["cross_attn_layers"] == [0, 1, 2, 3]
    assert ckpt["target_layers"] == [0, 1, 2, 3]
    # round-trip onto an identically-configured pipeline works
    d2 = Doubter.from_checkpoint(path)
    pipe2 = _fake_pipe(fake_lm_factory)
    pipe2.attach(d2)                             # no raise


def test_checkpoint_layer_mismatch_raises(tmp_path, fake_lm_factory):
    pipe = _fake_pipe(fake_lm_factory)          # cross_attn [0,1,2,3]
    d = _mini_doubter()
    pipe.attach(d)
    path = str(tmp_path / "d.pt")
    d.save_checkpoint(path)

    m = fake_lm_factory(hidden_dim=64, num_layers=4)
    from tests.conftest import FakeTokenizer
    cfg = MetaAttentionConfig(model_name="fake", device="cpu", dtype="float32",
                           target_layers=[0, 1, 2, 3], cross_attn_layers=[0, 1])  # mismatch
    pipe2 = MetaAttentionPipeline.from_pretrained(cfg, model=m, tokenizer=FakeTokenizer())
    d2 = Doubter.from_checkpoint(path)
    with pytest.raises(RuntimeError, match="mismatch"):
        pipe2.attach(d2)


# ───────────────────── A3: EOS supervised (labels mask by position, not value) ─────────────────────

def test_labels_mask_keeps_eos():
    from daimon_loom.training.losses import make_labels_with_prompt_mask
    pad = 0
    # seq: [prompt 5, 6] [target 7, eos=pad_id? no: eos==pad VALUE 0] + padding 0 0
    # with pad==eos, the target-final eos token has the SAME id as padding.
    input_ids = torch.tensor([[5, 6, 7, 0, 0, 0]])          # 7=target token, first 0=real EOS
    attn = torch.tensor([[1, 1, 1, 1, 0, 0]])               # real length 4 (incl. the EOS)
    labels = make_labels_with_prompt_mask(input_ids, prompt_len=2, pad_token_id=pad,
                                          attention_mask=attn)
    assert labels[0].tolist() == [-100, -100, 7, 0, -100, -100], \
        "the real EOS (id==pad) inside the attended region must stay supervised"
    # fallback without a mask keeps the old (value-based) behavior
    labels_old = make_labels_with_prompt_mask(input_ids, prompt_len=2, pad_token_id=pad)
    assert labels_old[0].tolist() == [-100, -100, 7, -100, -100, -100]


def test_trainer_appends_eos(fake_lm_factory):
    """_train_step must train on target+eos (the wrapper learns to STOP after the phrase)."""
    from daimon_loom.training.collector import DatasetSample
    from daimon_loom.training.trainer import Trainer, TrainerConfig

    pipe = _fake_pipe(fake_lm_factory)
    d = _mini_doubter()
    pipe.attach(d)
    tok = pipe.tokenizer
    acts = {i: torch.zeros(64) for i in range(4)}
    s = DatasetSample(input_text="q1", ground_truth="a1", activations=acts)
    tr = Trainer(d, pipe, TrainerConfig(epochs=1, batch_size=1, grad_accumulation=1,
                                        pretrain_projectors=False))
    captured = {}
    cls = type(tok)
    orig = cls.__call__

    def spy(self, text, **kw):
        if isinstance(text, list):
            captured["full_texts"] = text
        return orig(self, text, **kw)

    cls.__call__ = spy          # instance attr won't do: obj(...) resolves on the TYPE
    try:
        tr._train_step([s], [("target", "confirm")])
    finally:
        cls.__call__ = orig
    assert captured["full_texts"][0].endswith(tok.eos_token), \
        "the training text must end with the tokenizer's EOS"


# ───────────────────── A5: loud misconfig guards ─────────────────────

def test_collector_bad_layer_raises(fake_lm):
    with pytest.raises(ValueError, match="out of range"):
        ActivationCollector(fake_lm, target_layers=[0, 99])


def test_expand_batch_mismatch_raises():
    buf = ReflexionBuffer(num_slots=2, hidden_dim=4)
    buf.fill(torch.zeros(3, 2, 4))              # 3 prompts in the buffer
    with pytest.raises(RuntimeError, match="batch mismatch"):
        buf.expand_batch(2)                      # forward batch of 2 → misconfiguration
    assert buf.expand_batch(3).shape[0] == 3     # matching batch is fine
    buf.fill(torch.zeros(1, 2, 4))
    assert buf.expand_batch(5).shape[0] == 5     # broadcast of a single prompt is fine


# ───────────────────── B4: MCQ letter extraction (oracle skew) ─────────────────────

def test_mcq_checker_extracts_choice():
    from daimon_loom.data.dataset import check_answer_correctness, extract_mcq_letter
    # explicit statements
    assert extract_mcq_letter("The answer is B") == "B"
    assert extract_mcq_letter("Answer: (C)") == "C"
    assert extract_mcq_letter("B) because ...") == "B"
    # the old false positive: mentioned ≠ chosen
    assert extract_mcq_letter("Between B and C it is hard to say") is None
    assert check_answer_correctness("Between B and C it is hard to say", ["B"]) is False
    assert check_answer_correctness("The answer is B", ["B"]) is True
    assert check_answer_correctness("B", ["B"]) is True
    # non-letter short truths keep the word-boundary behavior
    assert check_answer_correctness("the value is 42 exactly", ["42"]) is True
    # long truths: normalized substring as before
    assert check_answer_correctness("**Karl Marx** wrote it", ["Karl Marx"]) is True


# ───────────────────── B1: tool-call target format ─────────────────────

def test_tool_call_formats_and_detection():
    from daimon_loom.data.agentic_mix import (
        TOOL_CALL_FORMATS, detect_tool_call_format, tool_call_text,
    )

    class Tok:
        def __init__(self, tpl): self.chat_template = tpl

    assert detect_tool_call_format(Tok("... <tool_call> ...")) == "qwen"
    assert detect_tool_call_format(Tok("... <|tool_call|> ...")) == "granite"
    assert detect_tool_call_format(Tok("... <|python_tag|> ...")) == "llama"
    assert detect_tool_call_format(Tok("plain template")) == "qwen"  # fallback + warning

    q = tool_call_text("search", {"q": 1}, fmt="qwen")
    assert q.startswith("<tool_call>") and '"arguments"' in q
    g = tool_call_text("search", {"q": 1}, fmt="granite")
    assert g.startswith("<|tool_call|>[")
    l = tool_call_text("search", {"q": 1}, fmt="llama")
    assert '"parameters"' in l and "<tool_call>" not in l
    with pytest.raises(ValueError):
        tool_call_text("x", {}, fmt="nope")
    assert set(TOOL_CALL_FORMATS) == {"qwen", "granite", "llama"}


# ───────────────────── B3: abstain affordance (fair baseline) ─────────────────────

def test_abstain_affordance_suffix():
    from daimon_loom.cli.collect import (
        ABSTAIN_AFFORDANCE_SUFFIX, MCQ_SUFFIX, resolve_prompt_flags,
    )
    # off by default — old behavior intact
    suffix, _ = resolve_prompt_flags(False, True, None)
    assert suffix == MCQ_SUFFIX
    # on: appended after the answer-format instruction, reaches BOTH arms via input_text
    suffix, _ = resolve_prompt_flags(False, True, None, abstain_affordance=True)
    assert suffix == MCQ_SUFFIX + ABSTAIN_AFFORDANCE_SUFFIX
    suffix, _ = resolve_prompt_flags(False, False, None, abstain_affordance=True)
    assert suffix == ABSTAIN_AFFORDANCE_SUFFIX
    # the wording must trip the refusal detector
    from daimon_loom.evaluation.harness import REFUSAL_PHRASES
    assert any(p in ABSTAIN_AFFORDANCE_SUFFIX.lower() for p in REFUSAL_PHRASES)


def test_check_gsm8k_answer_sentence_period():
    """Регекс чисел не должен захватывать точку конца предложения (находка второго Fable:
    «The answer is 5.» парсился как «5.» и молча занижал оракул на фразовых ответах)."""
    from daimon_loom.data.dataset import check_gsm8k_answer
    assert check_gsm8k_answer("The answer is 5.", "5")
    assert check_gsm8k_answer("So the total is 1,234.", "1234")
    assert check_gsm8k_answer("It equals 3.14, done.", "3.14")     # честная десятичная жива
    assert check_gsm8k_answer("Balance: -7.", "-7")
    assert check_gsm8k_answer("#### 42.", "42")                     # страховка и в ####-ветке
    assert not check_gsm8k_answer("The answer is 6.", "5")
