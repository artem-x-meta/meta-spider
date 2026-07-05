# Meta-Agent

> Agentic runtime + chat for the **two-pass meta-attention** wrapper ([Meta-Spider](../meta-spider-framework-dev)).
> The Doubter's latent decision → a standard action vocabulary.

A component of the **Meta-Spider** project (umbrella):

| Component | Role |
|---|---|
| **Meta-Core** | core — two-pass meta-attention mechanics + checkpoint contract |
| **Meta-Loom** | training + benchmark-evaluation pipeline (on Meta-Core) |
| **Meta-Agent** (here) | agentic sessions + chat (on Meta-Core) |

Meta-Agent depends on **Meta-Core** (inference), not on Meta-Loom (training).

## Practical usage

Chat or an agentic run with your own wrapper, in one command (console-script `meta-agent`):

```bash
# interactive chat with an attached Doubter
meta-agent chat --model Qwen/Qwen3.5-4B --checkpoint doubter.pt --quantization nf4
# one agentic task (--stdtools = calculator + knowledge_base; without --checkpoint — bare base)
meta-agent run  --model Qwen/Qwen3.5-4B --checkpoint doubter.pt --stdtools "what is 17*23?"
```

Or from Python — one builder assembles the whole stack (backend + native tool format + `StopBackend` + policy):

```python
from meta_core import MetaSpiderConfig
from meta_agent import build_agent, ChatLoop

cfg = MetaSpiderConfig(model_name="Qwen/Qwen3.5-4B", quantization="nf4")
agent = build_agent(cfg, checkpoint="doubter.pt")   # loads base + Doubter, assembles the agent
ChatLoop(agent).repl()                               # interactive; or agent.run("task").answer
```

The native tool format and stop strings are chosen by model name (Qwen / Granite / Llama / Gemma).

## Why a separate framework

Off-the-shelf agent engines (vLLM/TGI server, LangGraph, the OpenAI tool loop, the tau-bench harness)
rest on one contract: **single-pass `generate()`**, a KV cache, and the model's decision = an
**emitted token**. The two-pass Meta-Spider wrapper breaks it:

```
Standard engine:   prompt → generate() → tokens (decision visible in text / tool_call)
Meta-Spider:       prompt → Pass 1 (hidden) → encoder → cognitive tokens
                          → Pass 2 with injection → answer   (decision is LATENT)
```

- two passes → streaming and paged-attention don't fit, doubled TTFT;
- the decision (refuse/answer) is gates / a confidence head, **not a token** → invisible to the orchestrator.

So there is **no off-the-shelf engine for our wrapper by construction** — we need our own. Conceptually
it's the simplest, deterministic part: **95% is standard boilerplate** (loop, tool registry, history
accumulation). The new bit is **one seam**.

Tool format: for tool-tuned instruct models (Qwen / Granite / Llama-3 …) — `native.py`
(`NativeToolPrompt`/`NativeToolRenderer`: prompt via `apply_chat_template(tools=)`, native tool-call +
`StopBackend`). Hand-rolled ReAct text (`prompt.py`) is legacy, for models without tool-tuning.

## One seam: `ActionRenderer`

The latent decision is rendered into an action the ordinary loop reacts to: `final | tool | abstain`.
The injection stays latent (inside Pass 2); what surfaces is a **legitimate model action** (a tool
call into the world is not a text injection into the model's own reasoning context).

- `RefusalToolRenderer` — decision from text ("not sure" → call a tool). The semantics of
  `lab/.../tool_calibration.py`, lifted to a multi-step loop.
- `LatentActionRenderer` — reads the decision from a latent `signal` (`{"refuse": bool}` or
  `{"confidence": float}`), takes priority over text; falls back to text when no signal is present.
  **Half of the seam (the consumer) is ready.** What's left is emitting the signal from the core: the
  pipeline only returns STATIC gates, the per-question decision currently lives only in text — a
  per-input readout is needed (`ConfidenceHead` over the decision-point activations) = the unimplemented
  part of `ActionTokenAdapter` (a Meta-Core change + GPU validation). See `MetaSpiderBackend`.

## Structure

| File | What |
|------|-----|
| `session.py`  | `Session` — growing message history (agent and chat = one loop) |
| `tools.py`    | `Tool`, `ToolRegistry` — executable tool registry |
| `action.py`   | `AgentAction` + `ActionRenderer` — the action **seam** |
| `backends.py` | `InferenceBackend` — the inference **seam** (Meta-Spider / llama.cpp / OpenAI) |
| `policy.py`   | `Policy` protocol; `FakePolicy`, `BackendPolicy` (= backend + renderer) |
| `runtime.py`  | `MetaAgent` — the ReAct loop |
| `prompt.py`   | `ReActPromptBuilder` (tool schema + transcript) + `GemmaFormat`/`PlainFormat` |
| `react.py`    | `build_react_agent` — backend + tools → ready ReAct agent |
| `chat.py`     | `ChatLoop` — same loop, message supplied by a human |
| `stdtools.py` | real tools: `calculator`, `knowledge_base` |
| `tasks.py`    | `Task` + `tau_lite_suite` — multi-step tasks + success check |
| `bench.py`    | `AgentBench` — task run + trace scoring (success/tools/steps) |

**Two seams** keep meta-agent away from everything heavy:

- `Policy` (session → action) — the loop and tools know nothing about the model;
- `InferenceBackend` (prompt → text + latent signal) — we don't know *how* inference is computed:
  `MetaSpiderBackend` (in-process GPU), `LlamaCppBackend` (CPU prod), `OpenAIBackend` (remote). Each
  lazily pulls its own dependency.

So the engine is deterministic and tested on CPU without a GPU (`pytest tests/`), and the **Meta-Core**
core is imported lazily only in `MetaSpiderBackend`. The dependency on Meta-Loom (the training toolkit)
is zero.

## Example

```python
from meta_agent import MetaAgent, Tool, ToolRegistry, BackendPolicy, FakeBackend

tools = ToolRegistry([Tool("lookup", "oracle retrieval", lambda query: "answer: 42")])

# skeleton/test — no model: inference is scripted via FakeBackend
backend = FakeBackend(["I'm not sure", "the answer is 42"])
agent = MetaAgent(BackendPolicy(backend), tools)
print(agent.run("capital of X?").answer)   # → refuse→lookup→final

# real wrapper (needs the meta-spider core + a Doubter checkpoint)
# from meta_agent import MetaSpiderPolicy
# from meta_core import MetaSpiderConfig
# policy = MetaSpiderPolicy(MetaSpiderConfig(...), checkpoint="doubter.pt")
# agent = MetaAgent(policy, tools); agent.run("...")
```

## tau-bench-lite (agentic benchmark)

Multi-step tasks with **real** tools (Calculator/KnowledgeBase) + trace scoring.
`python examples/tau_lite_demo.py` — the agent solves them without a GPU (demo policy
`RuleSolverPolicy` instead of a model; in prod → `BackendPolicy(MetaSpiderBackend(...))`):

```
AgentBench: 4 tasks
  success_rate : 100.00%   avg_steps 2.5   tool_calls 6
  [ OK ] calc_basic         steps=2 tools=1 -> '396'
  [ OK ] lookup_basic       steps=2 tools=1 -> '299792458'
  [ OK ] chain_lookup_calc  steps=3 tools=2 -> '6300000'   (lookup→compute)
  [ OK ] chain_radius       steps=3 tools=2 -> '12742'
```

The loop/tools/bench don't depend on the model → swapping the policy for a real wrapper makes the same
bench measure the trained 12B.

## Full ReAct agent

`build_react_agent` assembles a working multi-tool agent: tool schema in the prompt + parsing of
`Action: <tool>[<arg>]` / `Answer: <x>`. Any model that speaks the ReAct format drives the loop;
`python examples/react_demo.py` — the "model" reads the prompt and solves all of tau-bench-lite (4/4)
without a GPU.

```python
from meta_agent import build_react_agent, ToolRegistry, calculator, knowledge_base, GemmaFormat

tools = ToolRegistry([calculator(), knowledge_base({"speed_of_light": "299792458"})])
agent = build_react_agent(backend, tools, fmt=GemmaFormat())   # GemmaFormat = chat_template gap
agent.run("Take speed_of_light and multiply it by 2.")          # lookup → calculator → answer
```

For a trained 12B: `backend = MetaSpiderBackend.from_meta_spider(cfg, checkpoint="doubter.pt")` —
loop/prompt/parsing are the same.

## Status

`0.0.1` — **full ReAct agent**: loop + real tools + prompt assembly (+`GemmaFormat`) + tool-call
parsing + two seams (action/inference, incl. `LatentActionRenderer`) + chat + tau-bench-lite. Tests
green (24). Next: per-input signal readout from Meta-Core (enables the latent path), a run on a real
12B, an OpenAI-compatible shim.
