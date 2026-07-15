"""Tool registry — standard boilerplate of an agentic engine.

A tool = name + description + a callable function (kwargs → observation string).
No connection to the model: the registry is purely executional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

__all__ = ["Tool", "ToolRegistry"]


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]
    arg: str = "input"            # primary argument name (for the ReAct form tool[value])

    def __call__(self, **kwargs) -> str:
        return str(self.fn(**kwargs))


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
