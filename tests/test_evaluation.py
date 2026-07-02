"""Level 4 smoke tests: EvalHarness + Benchmark + BaselineComparison + OpenRouterJudge.

The LLM judge is mocked — no real API is called. FakeLM is used for inference.
"""

import json
import pytest
import torch

from meta_core import Doubter, DoubterConfig, MetaSpiderConfig, MetaSpiderPipeline
from meta_loom import BaselineComparison, BenchmarkRunner, BenchmarkTask, ComparisonReport, EvalHarness, OpenRouterJudge, QABenchmark, TaskResult
from meta_loom.evaluation.harness import (
    classify_action,
    compute_metrics,
)
from meta_loom.evaluation.comparison import (
    _mcnemar_test,
    _paired_t_test,
)


# ============================================================
# EvalHarness: classify_action + compute_metrics
# ============================================================


def test_classify_action_refuse():
    assert classify_action("I'm not confident enough to answer.") == "refuse"
    assert classify_action("Sorry, I don't know.") == "refuse"


def test_classify_action_opening_rule():
    """Refusal counts only in the OPENING sentence: an answer followed by trailing doubt is a
    commit (pre-EOS-fix checkpoints ramble doubt phrases after answering — v0.3.1 re-measure)."""
    assert classify_action("F. 200 m/s. I'm not confident enough to answer.") == "confirm"
    assert classify_action("Rollo signed the treaty. I'm not sure though.") == "confirm"
    assert classify_action("I don't have enough information to answer that reliably.") == "refuse"
    # no sentence terminator at all — the whole text is the opening
    assert classify_action("not sure") == "refuse"


def test_classify_action_correct():
    # v0.3.1: CORRECTION_PHRASES tightened to the TRAINED template ("Wait, the correct…") —
    # bare "actually"/"wait"/"let me think" fired on ordinary speech/CoT and inflated the
    # correction metrics.
    assert classify_action("Wait, the correct answer is B.") == "correct"
    assert classify_action("Actually, the correct answer is 42.") == "correct"
    assert classify_action("Let me reconsider my approach.") == "correct"
    # ordinary speech no longer counts as a self-correction:
    assert classify_action("Actually, I think it's 42.") == "confirm"
    assert classify_action("Wait, let me think about B.") == "confirm"


def test_classify_action_confirm():
    assert classify_action("The answer is 42.") == "confirm"
    assert classify_action("B) Paris") == "confirm"


def test_compute_metrics_basic():
    preds = ["Paris", "I'm not sure", "Wait, the correct answer is Madrid"]
    truths = ["Paris", "London", "Madrid"]
    m = compute_metrics(preds, truths)
    assert m["n_total"] == 3
    assert m["n_refused"] == 1
    assert m["n_confirmed"] == 1
    assert m["n_corrected"] == 1
    assert m["overall_accuracy"] == round(2 / 3, 4)  # Paris + Madrid correct


def test_compute_metrics_with_pass1():
    preds = ["Paris", "I'm not sure", "Wait, the correct answer is Madrid"]
    truths = ["Paris", "London", "Madrid"]
    pass1 = [True, False, False]
    m = compute_metrics(preds, truths, pass1_correct=pass1)
    assert m["correction"]["n_pass1_wrong"] == 2
    assert m["correction"]["successful_corrections"] == 1
    assert m["correction"]["smart_refusal_rate"] == 0.5


def test_refusal_precision_none_without_oracle():
    """Without pass1_correct, refusal_precision = None (NOT a trivial ≈1.0 — that was a bug)."""
    preds = ["Paris", "I'm not sure"]
    truths = ["Paris", "London"]
    m = compute_metrics(preds, truths)  # without an oracle
    assert m["refusal_precision"] is None
    assert m["over_refusal_rate"] is None


def test_refusal_precision_honest_with_oracle():
    """Honest refusal_precision = the fraction of refusals where the base model would have erred.

    2 refusals: one on a question where the base model would have erred (justified), one where the base
    model knew (over-refusal). Honest precision = 1/2 = 0.5, over_refusal_rate = 0.5.
    The naive (buggy) version would give 1.0 (both refusal texts != the answer).
    """
    preds = ["I'm not sure", "I'm not sure", "Paris"]
    truths = ["London", "Madrid", "Paris"]
    pass1 = [False, True, True]  # q0: base would have erred; q1: base knew; q2: answered
    m = compute_metrics(preds, truths, pass1_correct=pass1)
    assert m["n_refused"] == 2
    assert m["refusal_precision"] == 0.5       # 1 justified out of 2 refusals
    assert m["over_refusal_rate"] == 0.5       # 1 over-refusal out of 2


# ============================================================
# EvalHarness.run
# ============================================================


def test_eval_harness_run(fake_lm, fake_tokenizer):
    """EvalHarness.run runs the pipeline and returns EvalResults."""
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    harness = EvalHarness(pipeline, max_new_tokens=3)
    results = harness.run(
        questions=["What is 2+2?", "Capital of France?"],
        ground_truths=["4", "Paris"],
        verbose=False,
    )
    assert results.n_total == 2
    assert 0.0 <= results.selective_accuracy <= 1.0


# ============================================================
# QABenchmark
# ============================================================


def test_qa_benchmark_scoring_contains():
    tasks = [
        BenchmarkTask(task_id="t1", prompt="?", expected_answer="paris"),
    ]
    bench = QABenchmark(name="test", tasks=tasks, scoring="contains")
    r = TaskResult(task_id="t1", success=False, final_answer="The capital is Paris.")
    assert bench.score_task(tasks[0], r) == 1.0


def test_qa_benchmark_scoring_exact_match():
    tasks = [BenchmarkTask(task_id="t1", prompt="?", expected_answer="Paris")]
    bench = QABenchmark(name="test", tasks=tasks, scoring="exact_match")
    r_match = TaskResult(task_id="t1", success=False, final_answer="paris")
    r_no_match = TaskResult(task_id="t1", success=False, final_answer="Paris, France")
    assert bench.score_task(tasks[0], r_match) == 1.0
    assert bench.score_task(tasks[0], r_no_match) == 0.0


def test_qa_benchmark_scoring_multiple_choice():
    tasks = [BenchmarkTask(task_id="t1", prompt="?", expected_answer="B")]
    bench = QABenchmark(name="test", tasks=tasks, scoring="multiple_choice")
    r1 = TaskResult(task_id="t1", success=False, final_answer="The answer is B")
    r2 = TaskResult(task_id="t1", success=False, final_answer="A")
    assert bench.score_task(tasks[0], r1) == 1.0
    assert bench.score_task(tasks[0], r2) == 0.0


def test_qa_benchmark_from_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    items = [
        {"task_id": "1", "prompt": "Q1?", "expected_answer": "A1"},
        {"task_id": "2", "prompt": "Q2?", "expected_answer": "A2"},
    ]
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    bench = QABenchmark.from_jsonl(str(path), name="loaded", scoring="contains")
    assert len(bench.tasks) == 2
    assert bench.tasks[0].task_id == "1"


# ============================================================
# BenchmarkRunner with QABenchmark
# ============================================================


def test_benchmark_runner_qa(fake_lm, fake_tokenizer):
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    tasks = [
        BenchmarkTask(task_id="t1", prompt="Hi", expected_answer="hi"),
        BenchmarkTask(task_id="t2", prompt="Hello", expected_answer="world"),
    ]
    bench = QABenchmark(name="smoke", tasks=tasks, scoring="contains")
    runner = BenchmarkRunner(pipeline, bench, max_tokens=3, verbose=False)
    results = runner.run()
    assert len(results) == 2
    assert all(isinstance(r, TaskResult) for r in results)
    assert all(r.wall_time_seconds >= 0 for r in results)


# ============================================================
# AgentComparison (agentic eval — delegates to Meta-Agent)
# ============================================================


def test_agent_comparison_native_loop():
    """AgentComparison runs the loop through Meta-Agent + the native tool format (NO GPU):
    a fake pipeline returns scripted output (1 tool_call → final), the tool is actually called,
    the grade is objective. Verifies the Loom→Agent seam end-to-end."""
    from meta_loom.evaluation.agentic import AgentComparison, AgentTask
    from meta_agent import Tool, ToolRegistry

    class FakeTok:
        def apply_chat_template(self, messages, tools=None, tokenize=False,
                                add_generation_prompt=True, **kw):
            return "PROMPT"

    class FakePipe:
        def __init__(self, outs):
            self.tokenizer = FakeTok(); self._o = list(outs); self._i = 0
        def generate(self, prompt, max_new_tokens=64, apply_chat_template=False):
            o = self._o[self._i] if self._i < len(self._o) else ""; self._i += 1; return o
        def attach(self, m): pass
        def detach_all(self): pass

    hits = {"n": 0}
    def _search(query):
        hits["n"] += 1; return f"OBS:{query}"
    reg = ToolRegistry([Tool("search", "find a fact", _search, arg="query")])
    task = AgentTask("t1", "What is the answer?", reg, grade=lambda a: "42" in a)
    pipe = FakePipe([
        "<function=search>\n<parameter=query>\nthe answer\n</parameter>\n</function>",
        "the answer is 42",
    ])
    rep = AgentComparison(pipe, model_name="qwen", max_steps=3).run([task], run_doubter=False)
    assert rep["base"]["pass_rate"] == 1.0, rep
    assert rep["base"]["avg_lookups"] == 1.0, rep      # exactly one real tool call
    assert hits["n"] == 1


# ============================================================
# Statistical tests (without scipy)
# ============================================================


def test_mcnemar_no_disagreement():
    """b=c=0 → p_value=1."""
    r = _mcnemar_test(b=0, c=0)
    assert r["p_value"] == 1.0


def test_mcnemar_strong_effect():
    """b=0, c=20 → significant improvement."""
    r = _mcnemar_test(b=0, c=20)
    assert r["p_value"] < 0.05


def test_paired_t_no_diff():
    """Identical scores → p=1."""
    r = _paired_t_test([1.0] * 10, [1.0] * 10)
    assert r["p_value"] == 1.0


def test_paired_t_clear_diff():
    """Modified strictly better → p<0.05."""
    a = [1.0] * 10
    b = [0.0] * 10
    r = _paired_t_test(a, b)
    assert r["mean_diff"] == 1.0
    assert r["p_value"] < 0.05


# ============================================================
# BaselineComparison full e2e
# ============================================================


def test_baseline_comparison_e2e(fake_lm, fake_tokenizer):
    """BaselineComparison runs with/without the modifier and returns a report."""
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)

    tasks = [
        BenchmarkTask(task_id=f"t{i}", prompt=f"Q{i}", expected_answer=f"a{i}")
        for i in range(3)
    ]
    bench = QABenchmark(name="cmp_smoke", tasks=tasks, scoring="contains")
    cmp = BaselineComparison(pipeline, bench, max_tokens=3)
    report = cmp.run(verbose=False)

    assert isinstance(report, ComparisonReport)
    assert report.n_tasks == 3
    assert len(report.base_results) == 3
    assert len(report.modified_results) == 3
    assert report.modifier_names == ["doubter"]
    assert "overall_accuracy" in report.base_metrics
    assert "overall_accuracy" in report.modified_metrics
    # After the run the Modifier should be re-attached
    assert doubter in pipeline.modifiers


def test_baseline_comparison_summary_str(fake_lm, fake_tokenizer):
    """summary() returns formatted text."""
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    tasks = [BenchmarkTask(task_id="t1", prompt="Q", expected_answer="x")]
    bench = QABenchmark(name="b", tasks=tasks, scoring="contains")
    report = BaselineComparison(pipeline, bench, max_tokens=2).run(verbose=False)

    s = report.summary()
    assert "Benchmark: b" in s
    assert "doubter" in s
    assert "Overall accuracy" in s


def test_baseline_comparison_save_json(tmp_path, fake_lm, fake_tokenizer):
    """save_json creates valid JSON."""
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    tasks = [BenchmarkTask(task_id="t1", prompt="Q", expected_answer="x")]
    bench = QABenchmark(name="b", tasks=tasks, scoring="contains")
    report = BaselineComparison(pipeline, bench, max_tokens=2).run(verbose=False)

    path = tmp_path / "report.json"
    report.save_json(str(path))
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["n_tasks"] == 1
    assert "per_task_deltas" in data


def test_baseline_comparison_refusal_breakdown(fake_lm, fake_tokenizer):
    """refusal_breakdown returns a confusion matrix."""
    cfg = MetaSpiderConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaSpiderPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    tasks = [BenchmarkTask(task_id=f"t{i}", prompt="Q", expected_answer="x") for i in range(2)]
    bench = QABenchmark(name="b", tasks=tasks, scoring="contains")
    report = BaselineComparison(pipeline, bench, max_tokens=2).run(verbose=False)
    breakdown = report.refusal_breakdown()
    assert "modified" in breakdown
    assert "delta_refusal_rate" in breakdown


# ============================================================
# OpenRouterJudge — without a real API call
# ============================================================


def test_openrouter_judge_no_api_key_raises(monkeypatch):
    """If there is neither an env var nor a .env — RuntimeError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Mock _find_env_file so it doesn't look for a real .env
    import meta_loom.evaluation.llm_judge as mod
    monkeypatch.setattr(mod, "_find_env_file", lambda: None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterJudge()


def test_openrouter_judge_init_with_explicit_key():
    j = OpenRouterJudge(api_key="sk-test-key")
    assert j.api_key == "sk-test-key"
    assert j.model == OpenRouterJudge.DEFAULT_MODEL


def test_openrouter_judge_cache_works():
    """If the answer is in cache — _call_api must not be called."""
    j = OpenRouterJudge(api_key="dummy")
    j._cache[("text", "criterion")] = 0.7
    score = j.judge("text", "criterion")
    assert score == 0.7
    assert j._calls_total == 0


def test_openrouter_judge_parse_score():
    j = OpenRouterJudge(api_key="dummy")
    assert j._parse_score("7") == 0.7
    assert j._parse_score("score: 10") == 1.0
    assert j._parse_score("invalid") == 0.5
    # Last valid number
    assert j._parse_score("steps: 5, score: 8") == 0.8


def test_openrouter_judge_stats():
    j = OpenRouterJudge(api_key="dummy")
    stats = j.stats()
    assert stats["calls_total"] == 0
    assert stats["calls_failed"] == 0
    assert stats["model"] == OpenRouterJudge.DEFAULT_MODEL
