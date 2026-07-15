"""Dialogue/trajectory state — a growing list of messages.

The agent and the chat use ONE loop; the only difference is the source of the next message
(tool vs human). So a single state — Session — fits both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["Message", "Session"]


@dataclass
class Message:
    role: str                      # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None     # tool name when role == "tool"

    def render(self) -> str:
        tag = self.role if self.name is None else f"{self.role}:{self.name}"
        return f"<{tag}> {self.content}"


@dataclass
class Session:
    """Multi-turn history. The source of truth for building the prompt."""
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str, name: Optional[str] = None) -> "Session":
        self.messages.append(Message(role, content, name))
        return self

    def system(self, content: str) -> "Session":
        return self.add("system", content)

    def user(self, content: str) -> "Session":
        return self.add("user", content)

    def assistant(self, content: str) -> "Session":
        return self.add("assistant", content)

    def tool(self, name: str, content: str) -> "Session":
        return self.add("tool", content, name=name)

    @property
    def last_user(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""

    def render(self) -> str:
        """Simple transcript. Per-model formatting is the Policy's job
        (e.g. Gemma tags or apply_chat_template); here a neutral view."""
        return "\n".join(m.render() for m in self.messages)
