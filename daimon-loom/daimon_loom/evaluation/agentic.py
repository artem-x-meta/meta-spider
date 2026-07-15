"""Agentic eval: an HONEST base vs Doubter comparison in a single loop via daimon-agent.

**Loom uses the Agent.** daimon_loom used to carry its own `AgenticBenchmark` — a manual ReAct loop
(regex parsing + `history += "Observation:"`) that doesn't work on live instruct models (the model
appends a fake Observation, there are no stop tokens). Here — delegation to `daimon_agent`: the real
runtime (`MetaAgent` + the native tool format `NativeToolPrompt`/`Renderer` + `StopBackend`), on which
the working lab harness was built (multihop / rag-coding). This is its reusable form.

BOTH arms (base and Doubter+AGC) run ONE loop with the SAME tools — the only difference is the
Doubter's latent injection. Grading is objective (set by the benchmark adapter). Optional forced
finalization (if an arm produced no usable answer → one generation without tools) is eval-specific,
set via `finalize_fn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from daimon_agent import (
    BackendPolicy, MetaAgent, DaimonBackend, NativeToolPrompt, NativeToolRenderer,
    Session, StopBackend, Tool, ToolRegistry, stops_for,
)

__all__ = ["AgentTask", "AgentComparison"]


@dataclass
class AgentTask:
    """One agentic task: prompt + tools + an objective grade of the final answer."""
    id: str
    prompt: str
    tools: ToolRegistry
    grade: Callable[[str], bool]
    metadata: dict = field(default_factory=dict)


def _counting_registry(tools: ToolRegistry, calls: dict, gathered: list) -> ToolRegistry:
    """Wrap the tools: count calls + collect observations (transparent to the model)."""
    wrapped = []
    for name in tools.names:
        t = tools.get(name)

        def make(tool: Tool):
            def fn(**kw):
                calls["n"] += 1
                obs = tool.fn(**kw)
                gathered.append(str(obs))
                return obs
            return fn
        wrapped.append(Tool(t.name, t.description, make(t), arg=t.arg))
    return ToolRegistry(wrapped)


class AgentComparison:
    """Run a list of `AgentTask` on (base) and (base+Doubter+AGC) → a report with rescued/broke/pass-rate.

    Args:
        pipeline: MetaAttentionPipeline (carries tokenizer + generate).
        doubter:  Doubter (attached on the doubter arm); None → base only.
        system:   the agent's system prompt.
        model_name: for choosing the turn stop-strings (`stops_for`: qwen/granite/llama/gemma).
        agc_floor/agc_tau: AGC on the Doubter's CA modules (keeps doubt in check); None → no AGC.
        finalize_fn: (answer, gathered, task) -> forcing prompt | None. If it returns a prompt —
                     one generation without tools (forced final). None → the answer as-is.
    """
    def __init__(self, pipeline, *, doubter=None, system: Optional[str] = None,
                 model_name: str = "", max_steps: int = 4, max_tokens: int = 320,
                 agc_floor: Optional[float] = None, agc_tau: float = 16.0,
                 finalize_fn: Optional[Callable] = None, verbose: bool = False):
        self.pipeline = pipeline
        self.doubter = doubter
        self.system = system
        self.stops = stops_for(model_name)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.agc_floor = agc_floor
        self.agc_tau = agc_tau
        self.finalize_fn = finalize_fn
        self.verbose = verbose

    def _run_arm(self, task: AgentTask, use_doubter: bool) -> tuple[bool, int, bool]:
        self.pipeline.detach_all()
        if use_doubter and self.doubter is not None:
            self.pipeline.attach(self.doubter)
            if self.agc_floor is not None:
                for m in self.doubter.ca_modules.values():
                    if hasattr(m, "agc_enabled"):
                        m.agc_enabled = True
                        m.agc_mode = "decay"
                        m.agc_floor = self.agc_floor
                        m.agc_tau = self.agc_tau
        calls = {"n": 0}
        gathered: list = []
        reg = _counting_registry(task.tools, calls, gathered)
        prompt_builder = NativeToolPrompt(self.pipeline.tokenizer, reg, system=self.system)
        renderer = NativeToolRenderer(reg)
        be = StopBackend(DaimonBackend(self.pipeline, max_new_tokens=self.max_tokens,
                                           apply_chat_template=False), self.stops)
        policy = BackendPolicy(be, renderer=renderer, prompt_builder=prompt_builder)
        res = MetaAgent(policy, reg, max_steps=self.max_steps).run(task.prompt, session=Session())
        answer = res.answer or ""
        forced = False
        if self.finalize_fn is not None:
            fp = self.finalize_fn(answer, gathered, task)
            if fp:
                forced = True
                answer = be.generate(self._force_prompt(fp)).text
        self.pipeline.detach_all()
        return bool(task.grade(answer)), calls["n"], forced

    def _force_prompt(self, user: str) -> str:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": user})
        return self.pipeline.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, thinking=False)

    def run(self, tasks: list[AgentTask], run_doubter: bool = True) -> dict:
        arms = ["base", "doubter"] if (run_doubter and self.doubter is not None) else ["base"]
        passed = {a: [] for a in arms}
        lookups = {a: [] for a in arms}
        forced = {a: [] for a in arms}
        rows = []
        for i, task in enumerate(tasks):
            row = {"task": task.id}
            for a in arms:
                ok, lk, f = self._run_arm(task, a == "doubter")
                passed[a].append(ok); lookups[a].append(lk); forced[a].append(f)
                row[f"{a}_pass"] = ok; row[f"{a}_lookups"] = lk; row[f"{a}_forced"] = f
            rows.append(row)
            if self.verbose:
                print(f"  T{i} {task.id}: " + " | ".join(
                    f"{a} pass={int(row[a+'_pass'])} lk={row[a+'_lookups']}" for a in arms), flush=True)
        n = max(len(tasks), 1)
        rep = {"n": len(tasks), "rows": rows}
        for a in arms:
            rep[a] = {"pass_rate": round(sum(passed[a]) / n, 4),
                      "avg_lookups": round(sum(lookups[a]) / n, 2),
                      "n_forced": sum(forced[a])}
        if "doubter" in arms:
            b, d = passed["base"], passed["doubter"]
            rep["doubter_minus_base"] = round(rep["doubter"]["pass_rate"] - rep["base"]["pass_rate"], 4)
            rep["rescued"] = sum(1 for x, y in zip(b, d) if (not x) and y)
            rep["broke"] = sum(1 for x, y in zip(b, d) if x and (not y))
        return rep
