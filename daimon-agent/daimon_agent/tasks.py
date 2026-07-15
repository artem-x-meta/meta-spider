"""Task — a unit of the agentic bench: a prompt + success check on the final answer.

tau-bench-lite: a small set of multi-step tasks that require tools (Calculator/
KnowledgeBase). Self-contained — no model dependency; solvable by any Policy
(the demo RuleSolverPolicy or a real BackendPolicy on a trained model).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["Task", "tau_lite_suite"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class Task:
    task_id: str
    prompt: str
    answer: str                                  # reference (normalized comparison)
    tools: list[str] = field(default_factory=list)   # expected tools (for the report)
    check: Optional[Callable[[str, str], bool]] = None

    def succeeded(self, final_answer: str) -> bool:
        if self.check:
            return bool(self.check(final_answer, self.answer))
        return _norm(self.answer) in _norm(final_answer)


def tau_lite_suite() -> tuple[list["Task"], dict]:
    """The tau-bench-lite set + facts for the KnowledgeBase.

    Returns (tasks, facts). Covers: pure computation (1 tool step),
    pure lookup (1 step), a lookup→computation chain (2 tool steps).
    """
    facts = {
        "france_capital_population": "2100000",
        "speed_of_light": "299792458",
        "earth_radius_km": "6371",
    }
    tasks = [
        Task("calc_basic", "Compute 23*17+5.", "396", tools=["calculator"]),
        Task("lookup_basic", "Find speed_of_light in the base.", "299792458", tools=["lookup"]),
        Task("chain_lookup_calc",
             "Take france_capital_population from the base and multiply by 3.",
             "6300000", tools=["lookup", "calculator"]),
        Task("chain_radius",
             "Take earth_radius_km from the base and multiply by 2.",
             "12742", tools=["lookup", "calculator"]),
    ]
    return tasks, facts
