"""Evaluation: a harness for computing metrics + LLM-judge + benchmark + comparison.

Two paths for evaluating the value-add of modifiers:
  - `BaselineComparison` (single-shot QA) — base vs base+modifiers on `QABenchmark`
    (MMLU, TriviaQA, GSM8K, HumanEval); the main selective-prediction interface.
  - `AgentComparison` (agentic, multi-step with tool use) — **delegates to Meta-Agent** (the real
    runtime + native tool format + StopBackend), an honest loop base vs Doubter+AGC.

The former `AgenticBenchmark` (a manual ReAct loop in `benchmark.py`) is DROPPED from the public
API — it doesn't work on live instruct models (fake Observation, no stop tokens). Use
`AgentComparison`. The class still exists for backward compatibility but is not exported.
"""

from meta_loom.evaluation.harness import EvalHarness
from meta_loom.evaluation.llm_judge import OpenRouterJudge
from meta_loom.evaluation.benchmark import (
    Benchmark,
    QABenchmark,
    BenchmarkRunner,
    BenchmarkTask,
    TaskResult,
)
from meta_loom.evaluation.comparison import (
    BaselineComparison,
    ComparisonReport,
)
from meta_loom.evaluation.agentic import AgentComparison, AgentTask

__all__ = [
    "EvalHarness",
    "OpenRouterJudge",
    # QA benchmark API (single-shot, selective prediction)
    "Benchmark",
    "QABenchmark",
    "BenchmarkRunner",
    "BenchmarkTask",
    "TaskResult",
    # Comparison API
    "BaselineComparison",
    "ComparisonReport",
    # Agentic eval (delegates to Meta-Agent)
    "AgentComparison",
    "AgentTask",
]
