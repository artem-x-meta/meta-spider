"""Demo: MetaAgent solves tau-bench-lite tasks with REAL tools, without GPU.

RuleSolverPolicy — a deterministic stub in place of a model: it looks at the session and calls
Calculator/KnowledgeBase. In prod it is replaced by BackendPolicy(MetaSpiderBackend(...)) —
then a trained 12B produces the decisions/tool calls, the loop and bench are the same.

    python examples/tau_lite_demo.py
"""
import re

from meta_agent import (
    AgentAction, AgentBench, MetaAgent, ToolRegistry,
    calculator, knowledge_base, tau_lite_suite,
)


class RuleSolverPolicy:
    """Deterministically solves tau-lite: lookup fact by key → (opt.) multiply → final."""
    def __init__(self, kb_keys):
        self.kb_keys = list(kb_keys)

    def act(self, session):
        text = session.last_user.lower()
        obs = [m for m in session.messages if m.role == "tool"]

        # chain: there is a lookup result and a multiplier is needed → calculator
        mul = re.search(r"multiply\s+by\s+(\d+)", text)
        if mul and obs and obs[-1].name == "lookup":
            return AgentAction.tool_call("calculator",
                                         {"expression": f"{obs[-1].content}*{mul.group(1)}"})

        # a fact from the base is needed (the key appears literally in the prompt)
        if not any(o.name == "lookup" for o in obs):
            for k in self.kb_keys:
                if k in text:
                    return AgentAction.tool_call("lookup", {"key": k})

        # pure arithmetic
        calc = re.search(r"compute\s+([0-9+\-*/().\s]+)", text)
        if calc and not obs:
            return AgentAction.tool_call("calculator",
                                         {"expression": calc.group(1).replace(" ", "")})

        # there is an observation → final
        if obs:
            return AgentAction.final(obs[-1].content)
        return AgentAction.abstain("I don't know how to solve it")


def build_agent():
    tasks, facts = tau_lite_suite()
    tools = ToolRegistry([calculator(), knowledge_base(facts)])
    policy = RuleSolverPolicy(kb_keys=list(facts.keys()))
    return MetaAgent(policy, tools, max_steps=6), tasks


if __name__ == "__main__":
    agent, tasks = build_agent()
    report = AgentBench(tasks).run(agent, verbose=True)
    print("\n" + report.summary())
