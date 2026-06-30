"""build-universal (the universal-Doubter factory): pure mix helpers + the orchestration on FakeLM.

The mix *builder* needs HF datasets (network) → not unit-tested here; we test the pure helpers and the
collect→train wiring with an INJECTED tiny mix, so the whole factory path runs on CPU without GPU/net.
"""
import json

from meta_loom.cli import build_universal as bu
from meta_loom.data import agentic_mix as AM
from tests.test_cli import _fake_pipe


# ───────────────────────── pure mix helpers (no tokenizer / no network) ─────────────────────────

def test_tool_call_text_native_format():
    assert (AM.tool_call_text("search", {"query": "x"})
            == '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>')
    # accepts a JSON-string args too
    assert '"a": 1' in AM.tool_call_text("f", '{"a": 1}')


def test_fix_types_python_to_jsonschema():
    assert AM.fix_types({"type": "dict"}) == {"type": "object"}
    assert AM.fix_types({"p": {"type": "float"}}) == {"p": {"type": "number"}}
    assert AM.fix_types([{"type": "tuple"}]) == [{"type": "array"}]


def test_toolcall_regex_parses_when2call_chosen():
    c = 'sure: <TOOLCALL>[{"name": "f", "arguments": {"a": 1}}]</TOOLCALL>'
    m = AM._TOOLCALL_RE.search(c)
    assert m and json.loads(m.group(1))[0]["name"] == "f"


def test_targets_from_samples():
    class S:  # minimal stand-in for a collected sample
        pass
    s = S()
    s.ground_truth = json.dumps({"target": "<tool_call>...", "label": "call"})
    assert AM.targets_from_samples([s]) == [("<tool_call>...", "call")]


def test_suite_exclude(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps([{"question": "  What  IS  X? "}, {"question": "Y"}]), encoding="utf-8")
    ex = bu._suite_exclude(str(p))
    assert "what is x?" in ex and "y" in ex
    assert bu._suite_exclude(None) == set()


# ───────────────────────── factory orchestration on FakeLM (CPU) ─────────────────────────

def test_build_universal_injected_mix(tmp_path, fake_lm_factory):
    run_dir = str(tmp_path / "uni")
    prompts = [f"prompt number {i}" for i in range(8)]
    specs = [json.dumps({"target": f"target {i}", "label": AM.AXES[i % len(AM.AXES)]})
             for i in range(8)]

    ckpt = bu.build_universal_stage(
        run_dir, "fake", pipeline=_fake_pipe(fake_lm_factory),
        mix=(prompts, specs), epochs=1, verbose=False,
    )

    assert ckpt.exists()
    man = json.loads((tmp_path / "uni" / "run.json").read_text(encoding="utf-8"))
    assert man["dataset"] == "diverse-agentic-mix"
    assert man["encoder_type"] == "selective"
    assert man["train_size"] + man["val_size"] == 8          # sizes derived from the mix
    assert man["chat_template_kwargs"]["enable_thinking"] is False
    assert (tmp_path / "uni" / "README.md").exists()
    assert (tmp_path / "uni" / "history.json").exists()
    status = json.loads((tmp_path / "uni" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["mix_size"] == 8


# ───────────────────────── per-axis suite eval (log-prob) ─────────────────────────

def _tiny_suite():
    opts = {"direct": "Paris.", "tool_call": "<tool_call>", "clarify": "Which one?", "refuse": "I don't know."}
    axes = [("1_call", "tool_call"), ("4_memory", "direct"), ("2_abstain", "refuse"), ("7_unknown", "refuse")]
    return [{"axis": ax, "source": "t", "question": f"q for {ax}", "tools": [], "options": opts,
             "correct": correct} for ax, correct in axes]


def test_eval_suite_logprob_base_and_doubter(fake_lm_factory):
    from meta_core import Doubter, DoubterConfig
    from meta_loom.evaluation import agentic_suite as A

    pipe = _fake_pipe(fake_lm_factory)
    suite = _tiny_suite()

    base = A.eval_suite_logprob(pipe, suite, has_doubter=False, verbose=False)
    assert set(base) == {"overall", "by_axis", "floor", "commit_mean"}
    assert set(base["by_axis"]) == {"1_call", "4_memory", "2_abstain", "7_unknown"}
    assert base["commit_mean"] is not None  # 1_call + 4_memory both present

    # doubter path (random init) — exercises pass1 + injection wiring, asserts structure only
    d = Doubter(DoubterConfig(encoder_type="selective", num_cognitive_tokens=4,
                              encoder_bottleneck=256, ca_bottleneck_dim=256, ca_num_heads=8))
    rep = A.compare_base_vs_doubter(pipe, d, suite, verbose=False)
    assert set(rep) == {"base", "doubter"} and "floor" in rep["doubter"]


def test_build_universal_with_eval(tmp_path, fake_lm_factory):
    run_dir = str(tmp_path / "uni2")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(_tiny_suite()), encoding="utf-8")
    prompts = [f"prompt {i}" for i in range(6)]
    specs = [json.dumps({"target": f"t{i}", "label": AM.AXES[i % len(AM.AXES)]}) for i in range(6)]

    bu.build_universal_stage(
        run_dir, "fake", pipeline=_fake_pipe(fake_lm_factory), mix=(prompts, specs),
        epochs=1, suite_path=str(suite_path), eval_suite=True, verbose=False,
    )
    rep = json.loads((tmp_path / "uni2" / "suite_eval.json").read_text(encoding="utf-8"))
    assert "base" in rep and "doubter" in rep and "floor" in rep["base"]
