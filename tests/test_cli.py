"""metaloom CLI tests: _common, dispatcher, and the chain collect→train→eval on FakeLM.

The key point — the manifest (run.json) threads the stages together: train/eval do NOT re-specify the architecture.
"""
import json

import pytest
import torch

from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_loom import ActivationDatasetCollector
from daimon_loom.cli import _common as C
from daimon_loom.cli import collect as collect_mod
from daimon_loom.cli import eval as eval_mod
from daimon_loom.cli import main as main_mod
from daimon_loom.cli import train as train_mod


# ───────────────────────── _common ─────────────────────────

def test_parse_layers():
    assert C.parse_layers("late") == "late"
    assert C.parse_layers("all") == "all"
    assert C.parse_layers("34,38,42") == [34, 38, 42]
    assert C.parse_layers([1, 2, 3]) == [1, 2, 3]


def test_resolve_prompt_formats():
    qs = ["q"]
    w, flag = C.resolve_prompt(qs, "gemma2", has_chat_template=False)
    assert flag is False and w[0].startswith("<start_of_turn>user")
    w, flag = C.resolve_prompt(qs, "gemma4_direct", has_chat_template=False)
    assert flag is False and w[0].startswith("<|turn>user")
    w, flag = C.resolve_prompt(qs, "auto", has_chat_template=True)
    assert flag is True and w == qs
    w, flag = C.resolve_prompt(qs, "auto", has_chat_template=False)
    assert flag is False and w[0].startswith("<start_of_turn>")
    with pytest.raises(ValueError):
        C.resolve_prompt(qs, "nope", has_chat_template=False)


def test_manifest_round_trip(tmp_path):
    m = {"model_name": "x", "target_layers": [1, 2], "cross_attn_layers": [2]}
    C.write_manifest(str(tmp_path), m)
    assert C.read_manifest(str(tmp_path)) == m
    with pytest.raises(FileNotFoundError):
        C.read_manifest(str(tmp_path / "missing"))


def test_status_merges(tmp_path):
    C.write_status(str(tmp_path), stage="collect", state="running")
    C.write_status(str(tmp_path), state="done", dataset="d.pt")
    s = json.loads((tmp_path / C.STATUS_NAME).read_text(encoding="utf-8"))
    assert s["stage"] == "collect" and s["state"] == "done" and s["dataset"] == "d.pt"


def test_auto_device():
    assert C.auto_device("cpu") == "cpu"
    assert C.auto_device("auto") in ("cpu", "cuda")


# ───────────────────────── dispatcher ─────────────────────────

def test_dispatcher_parses_stages():
    parser = main_mod.build_parser()
    for argv in (
        ["collect", "--run-dir", "r", "--model-name", "m"],
        ["train", "--run-dir", "r"],
        ["eval", "--run-dir", "r"],
    ):
        args = parser.parse_args(argv)
        assert hasattr(args, "_run") and callable(args._run)


def test_dispatcher_requires_stage():
    with pytest.raises(SystemExit):
        main_mod.build_parser().parse_args([])


# ───────────────────────── collector config-key ─────────────────────────

def test_collector_save_load_config(tmp_path, fake_lm, fake_tokenizer):
    cfg = MetaAttentionConfig(model_name="fake", device="cpu", dtype="float32")
    pipe = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    samples = ActivationDatasetCollector(pipe, max_new_tokens=2).collect(
        ["hi"], ["world"], verbose=False)
    p = tmp_path / "ds.pt"
    ActivationDatasetCollector.save(samples, str(p), config={"model_name": "fake"})
    assert ActivationDatasetCollector.load_config(str(p)) == {"model_name": "fake"}
    # without config — None, and load() still works
    ActivationDatasetCollector.save(samples, str(p))
    assert ActivationDatasetCollector.load_config(str(p)) is None
    assert len(ActivationDatasetCollector.load(str(p))) == 1


# ───────────────────────── chain collect→train→eval (FakeLM) ─────────────────────────

def _fake_pipe(fake_lm_factory):
    m = fake_lm_factory(hidden_dim=64, num_layers=4)
    from tests.conftest import FakeTokenizer  # noqa
    cfg = MetaAttentionConfig(
        model_name="fake", device="cpu", dtype="float32",
        target_layers=[0, 1, 2, 3], cross_attn_layers=[0, 1, 2, 3],
    )
    return MetaAttentionPipeline.from_pretrained(cfg, model=m, tokenizer=FakeTokenizer())


def _manifest():
    return {
        "format_version": C.MANIFEST_VERSION, "model_name": "fake", "dtype": "float32",
        "quantization": None, "device": "cpu",
        "target_layers": [0, 1, 2, 3], "cross_attn_layers": [0, 1, 2, 3],
        "encoder_type": "selective", "prompt_format": "gemma2", "dataset": "fake",
        "train_size": 4, "val_size": 1, "test_size": 1,
    }


def test_chain_collect_train_eval(tmp_path, fake_lm_factory):
    run_dir = str(tmp_path / "run")
    qs = [f"q{i}" for i in range(6)]
    ts = [[f"a{i}"] for i in range(6)]

    # collect — manifest + dataset.pt with embedded config
    collect_mod.collect_stage(
        _manifest(), run_dir, pipeline=_fake_pipe(fake_lm_factory),
        questions=qs, ground_truths=ts, collect_chunk=10, max_new_tokens=2, verbose=False,
    )
    man = C.read_manifest(run_dir)
    assert man["target_layers"] == [0, 1, 2, 3]  # resolved from the pipeline
    assert ActivationDatasetCollector.load_config(str(tmp_path / "run" / "dataset.pt"))["dataset"] == "fake"

    # train — reads run-dir (does NOT re-specify the architecture) → checkpoint
    ckpt = train_mod.train_stage(
        run_dir, epochs=1, batch_size=2, grad_accumulation=1,
        pipeline=_fake_pipe(fake_lm_factory), verbose=False,
    )
    assert ckpt.exists()
    assert (tmp_path / "run" / "history.json").exists()

    # eval — reads run-dir + checkpoint → report.json
    report = eval_mod.eval_stage(
        run_dir, pipeline=_fake_pipe(fake_lm_factory),
        check_fn=lambda pred, gts: True, max_tokens=2, verbose=False,
    )
    assert (tmp_path / "run" / "report.json").exists()
    assert "selective_accuracy" in report.modified_metrics


# ───────── thinking toggle + MCQ answer instruction (fix for null on verbose/thinking) ─────────

def test_resolve_prompt_flags():
    from daimon_loom.cli.collect import MCQ_SUFFIX, resolve_prompt_flags
    # default — touch nothing
    assert resolve_prompt_flags(False, False, None) == (None, None)
    # --no-think without a suffix
    assert resolve_prompt_flags(True, False, None) == (None, {"enable_thinking": False, "thinking": False})
    # --mcq-direct = no_think + standard suffix
    suf, kw = resolve_prompt_flags(False, True, None)
    assert suf == MCQ_SUFFIX and kw == {"enable_thinking": False, "thinking": False}
    # explicit --answer-suffix overrides the preset, but --mcq-direct still suppresses thinking
    assert resolve_prompt_flags(False, True, "X") == ("X", {"enable_thinking": False, "thinking": False})
    # suffix without no-think — don't touch thinking
    assert resolve_prompt_flags(False, False, "X") == ("X", None)


def test_collect_applies_suffix_and_no_think(tmp_path, fake_lm_factory):
    """answer_suffix is appended to the question + enable_thinking is forwarded into the template →
    both end up in input_text (eval/train inherit them via s.input_text automatically)."""
    run_dir = str(tmp_path / "run")
    pipe = _fake_pipe(fake_lm_factory)
    pipe.tokenizer.chat_template = "{{x}}"  # has_ct=True → apply_chat_template branch
    man = _manifest()
    man["prompt_format"] = "auto"
    man["answer_suffix"] = "\nONLY_LETTER"
    man["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
    collect_mod.collect_stage(
        man, run_dir, pipeline=pipe, questions=["whatcolor"], ground_truths=[["blue"]],
        collect_chunk=10, max_new_tokens=2, verbose=False,
    )
    s = ActivationDatasetCollector.load(str(tmp_path / "run" / "dataset.pt"))[0]
    assert "ONLY_LETTER" in s.input_text          # suffix inside the user turn
    assert "<think></think>" in s.input_text      # no-think marker of the fake tokenizer


def test_collect_default_no_suffix_no_think(tmp_path, fake_lm_factory):
    """Without flags — input_text has no suffix and no thinking marker (backward compatibility)."""
    run_dir = str(tmp_path / "run")
    pipe = _fake_pipe(fake_lm_factory)
    pipe.tokenizer.chat_template = "{{x}}"
    man = _manifest(); man["prompt_format"] = "auto"
    collect_mod.collect_stage(
        man, run_dir, pipeline=pipe, questions=["whatcolor"], ground_truths=[["blue"]],
        collect_chunk=10, max_new_tokens=2, verbose=False,
    )
    s = ActivationDatasetCollector.load(str(tmp_path / "run" / "dataset.pt"))[0]
    assert "ONLY_LETTER" not in s.input_text
    assert "<think>" not in s.input_text


# ───────────────────────── Part B: device/dtype first-class ─────────────────────────

def test_parse_max_memory():
    assert C.parse_max_memory(None) is None
    assert C.parse_max_memory("") is None
    assert C.parse_max_memory('{"0":"3GiB","cpu":"14GiB"}') == {0: "3GiB", "cpu": "14GiB"}


def test_config_device_auto_resolves(fake_lm, fake_tokenizer):
    cfg = MetaAttentionConfig(model_name="fake", dtype="float32")  # device defaults to "auto"
    assert cfg.device == "auto"
    assert hasattr(cfg, "max_memory") and cfg.max_memory is None
    MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    assert cfg.device in ("cpu", "cuda")  # resolved in from_pretrained
