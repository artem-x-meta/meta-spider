"""Benchmark API — a unified interface for running a model on task sets.

Two benchmark types are built in:

- **QABenchmark** — single-shot question/answer (MMLU / TriviaQA / GSM8K / HumanEval).
  Pipeline: prompt → generate → check_fn → score.
- **AgenticBenchmark** — multi-step with tool use (tau-bench / WebArena / AgentBench).
  Pipeline: loop [prompt → generate → parse tool call → execute tool → feed result] until
  success / failure / max_steps.

Concrete benchmarks register via `register_benchmark("name", BenchmarkClass)`.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from daimon_loom.evaluation.harness import compute_metrics, classify_action

if TYPE_CHECKING:
    from meta_attention.pipeline import MetaAttentionPipeline

__all__ = [
    "BenchmarkTask",
    "TaskResult",
    "Benchmark",
    "QABenchmark",
    "AgenticBenchmark",
    "BenchmarkRunner",
    "Tool",
    "ToolCall",
]


# ============================================================
# Base types
# ============================================================


@dataclass
class BenchmarkTask:
    """One benchmark task."""

    task_id: str
    prompt: str
    expected_answer: Optional[str] = None
    choices: Optional[list[str]] = None
    check_fn: Optional[Callable[[str], bool]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Agentic-only
    available_tools: list["Tool"] = field(default_factory=list)
    max_steps: int = 20
    success_criterion: Optional[Callable[[list["ToolCall"], str], bool]] = None


@dataclass
class TaskResult:
    """Result of running one task."""

    task_id: str
    success: bool
    final_answer: str
    num_steps: int = 1
    num_tokens_generated: int = 0
    wall_time_seconds: float = 0.0
    refused: bool = False

    # Agentic-only
    trajectory: list["ToolCall"] = field(default_factory=list)
    tool_call_count: int = 0

    error: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Tool calling (for agentic)
# ============================================================


@dataclass
class Tool:
    """Tool description for function calling."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]


@dataclass
class ToolCall:
    """One step of an agentic trajectory."""

    step: int
    thought: Optional[str]
    tool_name: Optional[str]
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    raw_output: str = ""


# ============================================================
# Benchmark ABC
# ============================================================


class Benchmark(ABC):
    """Base benchmark interface."""

    name: str = "benchmark"

    @property
    @abstractmethod
    def tasks(self) -> list[BenchmarkTask]:
        ...

    @abstractmethod
    def score_task(self, task: BenchmarkTask, result: TaskResult) -> float:
        ...

    def aggregate(self, results: list[TaskResult],
                  pass1_correct: Optional[list[bool]] = None) -> dict[str, Any]:
        """Default aggregation: overall success rate + basic counters.

        pass1_correct is accepted for signature compatibility (the base doesn't compute
        refusal_precision); it is used in QABenchmark.
        """
        n = len(results)
        if n == 0:
            return {"n_total": 0, "success_rate": 0.0}
        successes = sum(1 for r in results if r.success)
        refused = sum(1 for r in results if r.refused)
        return {
            "n_total": n,
            "n_success": successes,
            "n_refused": refused,
            "success_rate": round(successes / n, 4),
            "refusal_rate": round(refused / n, 4),
            "avg_wall_time": round(sum(r.wall_time_seconds for r in results) / n, 4),
        }


# ============================================================
# QABenchmark — single-shot QA
# ============================================================


class QABenchmark(Benchmark):
    """Single-shot question/answer benchmark.

    Args:
        name: benchmark name.
        tasks: list of tasks.
        scoring: how correctness is scored (custom / contains / exact_match / multiple_choice).
        normalize_fn: opt. text-normalization function applied before comparison.
    """

    SCORING_TYPES = Literal["exact_match", "contains", "multiple_choice", "custom"]

    def __init__(
        self,
        name: str,
        tasks: list[BenchmarkTask],
        scoring: str = "contains",
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        self._name = name
        self._tasks = tasks
        self.scoring = scoring
        self.normalize_fn = normalize_fn or (lambda s: s.lower().strip())

    @property
    def name(self) -> str:
        return self._name

    @property
    def tasks(self) -> list[BenchmarkTask]:
        return self._tasks

    def score_task(self, task: BenchmarkTask, result: TaskResult) -> float:
        if self.scoring == "custom" and task.check_fn is not None:
            return 1.0 if task.check_fn(result.final_answer) else 0.0
        pred = self.normalize_fn(result.final_answer)
        truth = self.normalize_fn(task.expected_answer or "")
        if self.scoring == "exact_match":
            return 1.0 if pred == truth else 0.0
        if self.scoring == "contains":
            return 1.0 if (truth and truth in pred) else 0.0
        if self.scoring == "multiple_choice":
            # Compare by the option letter (A/B/C/D)
            choice_letter = self._extract_choice_letter(pred)
            truth_letter = self._extract_choice_letter(truth)
            if not truth_letter:
                return 1.0 if (truth and truth in pred) else 0.0
            return 1.0 if choice_letter == truth_letter else 0.0
        return 0.0

    @staticmethod
    def _extract_choice_letter(text: str) -> Optional[str]:
        """Find the first A/B/C/D answer in text (case-insensitive)."""
        m = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def aggregate(self, results: list[TaskResult],
                  pass1_correct: Optional[list[bool]] = None) -> dict[str, Any]:
        """Calibration metrics (sel_acc, refusal_prec) + the basic success rate.

        We use the real task.check_fn for each task (rather than recreating a mock_task,
        which loses the check_fn reference under scoring='custom').

        pass1_correct (opt.) — the 'would the base/Pass-1 get it right' oracle: needed for an HONEST
        refusal_precision. Without it, compute_metrics returns refusal_precision=None
        (the naive version is trivially ≈1.0 — a bug, see honest-metrics-correction.md).
        """
        predictions = [r.final_answer for r in results]
        ground_truths = [self._task_by_id(r.task_id).expected_answer or "" for r in results]

        # Per-task scoring via the real tasks (preserves check_fn for scoring='custom')
        correct_flags = []
        for r in results:
            real_task = self._task_by_id(r.task_id)
            score = self.score_task(real_task, r)
            correct_flags.append(score >= 0.5)

        # compute_metrics expects a callable check_correctness — it iterates in predictions
        # order, which matches the order of correct_flags.
        _it = iter(correct_flags)

        def _check(pred: str, truth: str) -> bool:
            return next(_it, False)

        metrics = compute_metrics(predictions, ground_truths,
                                  pass1_correct=pass1_correct, check_correctness=_check)
        metrics["benchmark_success_rate"] = round(
            sum(correct_flags) / len(correct_flags), 4,
        ) if correct_flags else 0.0
        return metrics

    def _task_by_id(self, task_id: str) -> BenchmarkTask:
        for t in self._tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task not found: {task_id}")

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        name: str,
        scoring: str = "contains",
    ) -> "QABenchmark":
        """Load tasks from JSONL: {"task_id", "prompt", "expected_answer", ...}."""
        tasks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tasks.append(BenchmarkTask(
                    task_id=d["task_id"],
                    prompt=d["prompt"],
                    expected_answer=d.get("expected_answer"),
                    choices=d.get("choices"),
                    metadata=d.get("metadata", {}),
                ))
        return cls(name=name, tasks=tasks, scoring=scoring)


# ============================================================
# AgenticBenchmark — multi-step with tools
# ============================================================


_REACT_ACTION_PAT = re.compile(
    r"Action:\s*(\w+)\s*[\n\r]+\s*Args:\s*(\{.*?\})",
    re.DOTALL,
)
_REACT_FINAL_PAT = re.compile(r"Final\s*Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _parse_react_tool_call(output: str, step: int) -> Optional[ToolCall]:
    """ReAct-style: 'Thought: ...\\nAction: name\\nArgs: {...}' or 'Final Answer: ...'."""
    m_final = _REACT_FINAL_PAT.search(output)
    if m_final:
        return None  # the model decided to answer — no tool call
    m = _REACT_ACTION_PAT.search(output)
    if not m:
        return None
    tool_name = m.group(1)
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        args = {}
    thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|$)", output, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None
    return ToolCall(step=step, thought=thought, tool_name=tool_name,
                    arguments=args, raw_output=output)


def _extract_final_answer(output: str) -> str:
    m = _REACT_FINAL_PAT.search(output)
    return m.group(1).strip() if m else output.strip()


class AgenticBenchmark(Benchmark):
    """DEPRECATED (not in the public API) — use `daimon_loom.evaluation.AgentComparison`.

    The manual ReAct loop below (regex parsing + `history += "Observation:"`) does NOT work on
    live instruct models: the model appends a fake Observation, there are no stop tokens, and no
    native tool format. `AgentComparison` delegates to daimon-agent (the real runtime + `NativeToolPrompt`
    + `StopBackend`). The class is kept only for backward compatibility.

    Pipeline per task:

        history = system_prompt + task.prompt
        for step in range(task.max_steps):
            output = pipeline.generate(history)
            tool_call = parse_tool_call(output)
            if tool_call is None:
                final = extract_final_answer(output)
                break
            result = tool.execute(args)
            history += output + observation(result)

        success = task.success_criterion(trajectory, final)
    """


    def __init__(
        self,
        name: str,
        tasks: list[BenchmarkTask],
        tool_call_format: Literal["react", "custom"] = "react",
        parse_tool_call_fn: Optional[Callable[[str, int], Optional[ToolCall]]] = None,
        system_prompt: str = "",
    ):
        self._name = name
        self._tasks = tasks
        self.tool_call_format = tool_call_format
        self.parse_tool_call_fn = parse_tool_call_fn
        if tool_call_format == "react":
            self._parser = _parse_react_tool_call
        elif tool_call_format == "custom":
            if parse_tool_call_fn is None:
                raise ValueError("custom format requires parse_tool_call_fn")
            self._parser = parse_tool_call_fn
        else:
            raise ValueError(f"Unknown tool_call_format: {tool_call_format!r}")
        self.system_prompt = system_prompt

    @property
    def name(self) -> str:
        return self._name

    @property
    def tasks(self) -> list[BenchmarkTask]:
        return self._tasks

    def score_task(self, task: BenchmarkTask, result: TaskResult) -> float:
        if task.success_criterion is None:
            return 1.0 if result.success else 0.0
        try:
            return 1.0 if task.success_criterion(result.trajectory, result.final_answer) else 0.0
        except Exception:
            return 0.0

    def aggregate(self, results: list[TaskResult],
                  pass1_correct: Optional[list[bool]] = None) -> dict[str, Any]:
        base = super().aggregate(results, pass1_correct=pass1_correct)
        if not results:
            return base
        avg_steps = sum(r.num_steps for r in results) / len(results)
        avg_tools = sum(r.tool_call_count for r in results) / len(results)
        timeouts = sum(1 for r in results if r.num_steps >= 1 and r.error == "MAX_STEPS_REACHED")
        base.update({
            "avg_steps_per_task": round(avg_steps, 4),
            "avg_tool_calls_per_task": round(avg_tools, 4),
            "timeout_rate": round(timeouts / len(results), 4),
        })
        return base

    def parse_tool_call(self, output: str, step: int = 0) -> Optional[ToolCall]:
        return self._parser(output, step)


# ============================================================
# Runner — run ONE pipeline on a benchmark
# ============================================================


class BenchmarkRunner:
    """Run a pipeline on a benchmark + collect TaskResults."""

    def __init__(
        self,
        pipeline: "MetaAttentionPipeline",
        benchmark: Benchmark,
        max_tokens: int = 256,
        verbose: bool = True,
        apply_chat_template: bool = True,
        generate_kwargs: Optional[dict] = None,
    ):
        self.pipeline = pipeline
        self.benchmark = benchmark
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.apply_chat_template = apply_chat_template
        # Extra arguments for pipeline.generate (e.g. dynamic_refresh=True)
        self.generate_kwargs = generate_kwargs or {}

    def run(self, num_tasks: Optional[int] = None) -> list[TaskResult]:
        """Run the first `num_tasks` (or all) tasks. Progress every 10 tasks
        with time and rate (flush=True — visible as it goes, even through a pipe)."""
        tasks = self.benchmark.tasks
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        results = []
        t0 = time.time()
        for i, t in enumerate(tasks):
            if self.verbose and i > 0 and i % 10 == 0:
                el = time.time() - t0
                rate = i / el
                eta = (len(tasks) - i) / rate if rate else 0
                print(f"  {i}/{len(tasks)}  {el:.0f}s  {rate*60:.1f} q/min  ETA {eta/60:.1f}m",
                      flush=True)
            results.append(self.run_single_task(t))
        if self.verbose:
            print(f"  {len(tasks)}/{len(tasks)} done in {time.time()-t0:.0f}s", flush=True)
        return results

    def run_single_task(self, task: BenchmarkTask) -> TaskResult:
        """Run one task. Polymorphism by benchmark type."""
        if isinstance(self.benchmark, AgenticBenchmark):
            return self._run_agentic_task(task)
        return self._run_qa_task(task)

    def _run_qa_task(self, task: BenchmarkTask) -> TaskResult:
        t0 = time.time()
        try:
            answer = self.pipeline.generate(
                task.prompt, max_new_tokens=self.max_tokens,
                apply_chat_template=self.apply_chat_template,
                **self.generate_kwargs,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id, success=False, final_answer="",
                wall_time_seconds=time.time() - t0, error=str(e),
            )
        elapsed = time.time() - t0
        # Refusal classification
        action = classify_action(answer)
        refused = action == "refuse"
        # Scoring
        score = self.benchmark.score_task(task, TaskResult(
            task_id=task.task_id, success=False, final_answer=answer,
        ))
        return TaskResult(
            task_id=task.task_id,
            success=(score >= 0.5),
            final_answer=answer,
            num_steps=1,
            wall_time_seconds=elapsed,
            refused=refused,
        )

    def _run_agentic_task(self, task: BenchmarkTask) -> TaskResult:
        """Multi-step loop with tool execution."""
        assert isinstance(self.benchmark, AgenticBenchmark)
        bench = self.benchmark
        tool_map = {t.name: t for t in task.available_tools}

        history = ""
        if bench.system_prompt:
            history += bench.system_prompt + "\n\n"
        history += task.prompt

        trajectory: list[ToolCall] = []
        final_answer = ""
        t0 = time.time()
        error: Optional[str] = None

        for step in range(task.max_steps):
            try:
                output = self.pipeline.generate(
                    history, max_new_tokens=self.max_tokens,
                    apply_chat_template=self.apply_chat_template,
                )
            except Exception as e:
                error = str(e)
                break

            tool_call = bench.parse_tool_call(output, step=step)
            if tool_call is None:
                final_answer = _extract_final_answer(output)
                break

            # Execute
            tool = tool_map.get(tool_call.tool_name or "")
            if tool is None:
                tool_call.result = f"ERROR: tool '{tool_call.tool_name}' not available"
            else:
                try:
                    tool_call.result = tool.execute(tool_call.arguments)
                except Exception as e:
                    tool_call.result = f"ERROR: {e}"

            trajectory.append(tool_call)
            history += f"\n{output}\nObservation: {tool_call.result}\n"

        else:
            # No final answer found within max_steps
            error = "MAX_STEPS_REACHED"

        elapsed = time.time() - t0
        # Refusal — if there were no tool calls and the answer looks like a refusal
        refused = (not trajectory) and (classify_action(final_answer) == "refuse")

        # Scoring
        score = bench.score_task(task, TaskResult(
            task_id=task.task_id, success=False, final_answer=final_answer,
            trajectory=trajectory,
        ))
        return TaskResult(
            task_id=task.task_id,
            success=(score >= 0.5),
            final_answer=final_answer,
            num_steps=max(1, len(trajectory) + (0 if error else 1)),
            wall_time_seconds=elapsed,
            refused=refused,
            trajectory=trajectory,
            tool_call_count=len(trajectory),
            error=error,
        )
