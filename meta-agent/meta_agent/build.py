"""One call — a ready agent. Covers the "practical use" of the wrapper.

A random dev shouldn't have to hand-wire backend → native tool format → renderer → stop strings →
policy → MetaAgent. `build_agent(config, checkpoint)` loads the base + Doubter and returns a ready
`MetaAgent` (for `ChatLoop` or `.run(task)`). The CLI `meta-agent chat/run` is built on this.

The seam is split into two levels: `assemble_agent` (from a READY backend+tokenizer — tested on
FakeBackend without GPU) and `build_agent` (loads a real model via Meta-Core; lazy-imports the core).
"""
from __future__ import annotations

from typing import Optional

from .backends import StopBackend
from .native import NativeToolPrompt, NativeToolRenderer, stops_for
from .policy import BackendPolicy
from .runtime import MetaAgent
from .tools import ToolRegistry

__all__ = ["assemble_agent", "build_agent"]


def _registry(tools) -> ToolRegistry:
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(list(tools) if tools else [])


def assemble_agent(backend, tokenizer, *, tools=None, system: Optional[str] = None,
                   model_name: str = "", max_steps: int = 6, stops=None) -> MetaAgent:
    """Assemble a `MetaAgent` from a ready `backend` + `tokenizer`: native tool format
    (`NativeToolPrompt`/`Renderer`) + `StopBackend` (stops by `model_name`) + `BackendPolicy`.

    Low-level seam — independent of Meta-Core, tested on FakeBackend without GPU.
    """
    reg = _registry(tools)
    be = StopBackend(backend, stops if stops is not None else stops_for(model_name))
    policy = BackendPolicy(be, renderer=NativeToolRenderer(reg),
                           prompt_builder=NativeToolPrompt(tokenizer, reg, system=system))
    return MetaAgent(policy, reg, max_steps=max_steps)


def build_agent(config, checkpoint: Optional[str] = None, *, tools=None,
                system: Optional[str] = None, max_new_tokens: int = 256,
                max_steps: int = 6, repetition_penalty: float = 1.0) -> MetaAgent:
    """Full build: loads the base from `config` + Doubter from `checkpoint` (None → bare base) and
    assembles a ready agent. `apply_chat_template=False` — the prompt is built by `NativeToolPrompt`.
    Lazy-import of Meta-Core (GPU path) — `import meta_agent` without a model stays lightweight.
    """
    from .backends import MetaSpiderBackend  # lazy: pulls in meta_core
    backend = MetaSpiderBackend.from_meta_spider(
        config, checkpoint, max_new_tokens=max_new_tokens, apply_chat_template=False,
        repetition_penalty=repetition_penalty)
    return assemble_agent(backend, backend.pipeline.tokenizer, tools=tools, system=system,
                          model_name=getattr(config, "model_name", ""), max_steps=max_steps)
