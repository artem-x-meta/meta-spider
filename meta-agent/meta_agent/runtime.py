"""MetaAgent — a ReAct-style loop. Deterministic boilerplate.

  user → Policy.act(session) → AgentAction
      final  → return the answer
      tool   → execute from the registry, append the observation to the session, repeat
      abstain→ return with a flag (the model refused and there is no tool)

No connection to the model: everything goes through Policy + ToolRegistry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .action import AgentAction
from .policy import Policy
from .session import Session
from .tools import ToolRegistry

__all__ = ["MetaAgent", "AgentResult"]


@dataclass
class AgentResult:
    answer: str
    steps: int
    trace: list = field(default_factory=list)
    abstained: bool = False
    truncated: bool = False           # hit max_steps without a final


class MetaAgent:
    def __init__(self, policy: Policy, tools: Optional[ToolRegistry] = None,
                 max_steps: int = 6, step_hooks: Optional[list] = None):
        """step_hooks: objects with optional `on_step_start(step, session)` /
        `on_step_end(step, session)` — called around each policy.act (the decision window).
        Used by decision-point modifiers, e.g. GoalAnchor in agent_step mode arms its
        anchor injection only for the decision emission (decide-then-detach)."""
        self.policy = policy
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.step_hooks = list(step_hooks or [])

    def _notify(self, event: str, step: int, session: "Session") -> None:
        for h in self.step_hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(step, session)

    def run(self, user_input: str, session: Optional[Session] = None) -> AgentResult:
        session = session or Session()
        session.user(user_input)
        trace: list = []

        for step in range(1, self.max_steps + 1):
            self._notify("on_step_start", step, session)
            try:
                action = self.policy.act(session)
            finally:
                self._notify("on_step_end", step, session)

            if action.kind == "final":
                session.assistant(action.content)
                trace.append({"step": step, "action": "final"})
                return AgentResult(action.content, step, trace)

            if action.kind == "abstain":
                trace.append({"step": step, "action": "abstain"})
                return AgentResult(action.content, step, trace, abstained=True)

            if action.kind == "tool":
                tool = self.tools.get(action.tool)
                obs = tool(**action.args) if tool else f"[unknown tool: {action.tool}]"
                # record the agent's action + observation (Action → Observation in the transcript)
                arg_val = next(iter(action.args.values()), "") if action.args else ""
                session.add("assistant", f"{action.tool}[{arg_val}]", name="action")
                session.tool(action.tool or "?", obs)
                trace.append({"step": step, "action": "tool",
                              "tool": action.tool, "args": action.args, "obs": obs})
                continue

            # unknown kind → treat as final to avoid looping
            session.assistant(action.content)
            return AgentResult(action.content, step, trace)

        return AgentResult("", self.max_steps, trace, truncated=True)
