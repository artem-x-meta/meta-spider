"""AgentBench — run a MetaAgent over a set of Tasks, compute success + trace statistics.
A self-contained lightweight daimon-agent bench (the heavy eval machinery lives in daimon-loom).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .runtime import MetaAgent
from .tasks import Task

__all__ = ["TaskOutcome", "BenchReport", "AgentBench"]


@dataclass
class TaskOutcome:
    task_id: str
    success: bool
    steps: int
    tool_calls: int
    answer: str
    abstained: bool = False
    truncated: bool = False


@dataclass
class BenchReport:
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def success_rate(self) -> float:
        return sum(o.success for o in self.outcomes) / self.n if self.n else 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(o.tool_calls for o in self.outcomes)

    @property
    def avg_steps(self) -> float:
        return sum(o.steps for o in self.outcomes) / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return sum(o.abstained for o in self.outcomes) / self.n if self.n else 0.0

    def summary(self) -> str:
        lines = [
            f"AgentBench: {self.n} tasks",
            f"  success_rate : {self.success_rate:.2%}",
            f"  avg_steps    : {self.avg_steps:.1f}",
            f"  tool_calls   : {self.total_tool_calls}",
            f"  abstain_rate : {self.abstain_rate:.2%}",
            "  " + "-" * 52,
        ]
        for o in self.outcomes:
            mark = "OK " if o.success else ("ABST" if o.abstained else ("TRUNC" if o.truncated else "FAIL"))
            lines.append(f"  [{mark:>4}] {o.task_id:<20} steps={o.steps} tools={o.tool_calls} -> {o.answer[:32]!r}")
        return "\n".join(lines)


class AgentBench:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def run(self, agent: MetaAgent, verbose: bool = False) -> BenchReport:
        outcomes: list[TaskOutcome] = []
        for t in self.tasks:
            res = agent.run(t.prompt)                       # fresh session per task
            tool_calls = sum(1 for tr in res.trace if tr.get("action") == "tool")
            success = (not res.abstained) and (not res.truncated) and t.succeeded(res.answer)
            outcomes.append(TaskOutcome(
                task_id=t.task_id, success=success, steps=res.steps,
                tool_calls=tool_calls, answer=res.answer,
                abstained=res.abstained, truncated=res.truncated,
            ))
            if verbose:
                print(f"  {t.task_id}: success={success} steps={res.steps} "
                      f"tools={tool_calls} answer={res.answer[:40]!r}", flush=True)
        return BenchReport(outcomes)
