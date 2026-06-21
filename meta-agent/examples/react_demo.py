"""Demo of a full ReAct agent: the backend READS the prompt and replies in ReAct format —
the whole real path (prompt → "model" → Action → execution → Observation → ... → Answer)
on the tau-bench-lite set, without GPU. ReActStubBackend = a tiny rule-based "model";
in prod it is replaced by MetaSpiderBackend (a trained 12B), the loop/prompt/parsing are the same.

    python examples/react_demo.py
"""
import re

from meta_agent import (
    AgentBench, InferenceOutput, ToolRegistry,
    build_react_agent, calculator, knowledge_base, tau_lite_suite,
)


class ReActStubBackend:
    """A "model" that reads the transcript from the prompt and emits Action/Answer."""
    def __init__(self, kb_keys):
        self.kb_keys = list(kb_keys)

    def generate(self, prompt: str) -> InferenceOutput:
        # parse ONLY the transcript (after the first 'User:') — otherwise we catch placeholders
        # from the instruction lines ('Observation: <result>').
        tx = prompt[prompt.find("User:"):] if "User:" in prompt else prompt
        user = (re.findall(r"User:\s*(.+)", tx) or [""])[-1].lower()
        obs = re.findall(r"Observation:\s*(.+)", tx)
        did_lookup = "Action: lookup[" in tx

        mul = re.search(r"multiply\s+by\s+(\d+)", user)
        if mul and obs and "Action: calculator[" not in tx:
            return InferenceOutput(f"Action: calculator[{obs[-1].strip()}*{mul.group(1)}]")
        if not did_lookup:
            for k in self.kb_keys:
                if k in user:
                    return InferenceOutput(f"Action: lookup[{k}]")
        calc = re.search(r"compute\s+([0-9+\-*/().\s]+)", user)
        if calc and not obs:
            expr = calc.group(1).replace(" ", "").rstrip(".")
            return InferenceOutput(f"Action: calculator[{expr}]")
        if obs:
            return InferenceOutput(f"Answer: {obs[-1].strip()}")
        return InferenceOutput("Answer: I don't know")


if __name__ == "__main__":
    tasks, facts = tau_lite_suite()
    tools = ToolRegistry([calculator(), knowledge_base(facts)])
    agent = build_react_agent(ReActStubBackend(list(facts.keys())), tools, max_steps=6)

    # show one assembled prompt
    print("=== sample ReAct prompt (after the first step) ===")
    from meta_agent import Session
    s = Session().user(tasks[2].prompt)
    s.add("assistant", "lookup[france_capital_population]", name="action")
    s.tool("lookup", "2100000")
    print(agent.policy.prompt_builder(s))
    print("\n=== tau-bench-lite run ===")
    report = AgentBench(tasks).run(agent, verbose=True)
    print("\n" + report.summary())
