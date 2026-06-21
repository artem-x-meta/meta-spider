"""build_react_agent — assemble a ready ReAct agent: backend + tools → MetaAgent.

Wires three parts into a working multi-tool agent:
  ReActPromptBuilder (session+tools → prompt) → backend.generate → ReActRenderer
  (text → Action/Answer). Any backend whose model speaks the ReAct format drives the
  agent; for a trained 12B — MetaSpiderBackend + GemmaFormat.
"""
from __future__ import annotations

from typing import Optional

from .action import ActionRenderer, ReActRenderer
from .backends import InferenceBackend
from .policy import BackendPolicy
from .prompt import PromptFormat, ReActPromptBuilder
from .runtime import MetaAgent
from .tools import ToolRegistry

__all__ = ["build_react_agent"]


def build_react_agent(backend: InferenceBackend, tools: ToolRegistry, *,
                      fmt: Optional[PromptFormat] = None,
                      system: Optional[str] = None,
                      renderer: Optional[ActionRenderer] = None,
                      max_steps: int = 6) -> MetaAgent:
    """backend + tools → MetaAgent with a ReAct prompt and ReAct parsing.

    fmt — the per-model wrapper (GemmaFormat for Gemma-4; PlainFormat by default).
    """
    prompt_builder = ReActPromptBuilder(tools, fmt=fmt, system=system)
    policy = BackendPolicy(backend, renderer=renderer or ReActRenderer(tools),
                           prompt_builder=prompt_builder)
    return MetaAgent(policy, tools, max_steps=max_steps)
