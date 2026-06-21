"""ReAct prompt from a session (LEGACY text protocol) — for models WITHOUT native tool-use.

⚠️ For tool-tuned instruct models (Qwen / Granite / Llama-3 …) use [meta_agent.native]
(`NativeToolPrompt`/`NativeToolRenderer`): prompt via the official `apply_chat_template(tools=)`,
native tool-call. This hand-written ReAct text ("Action: tool[arg]" / "Observation:") is held
poorly by live instruct models (they append a fake Observation) — kept for models with their own
format without tool-tuning / for compatibility.

ReActPromptBuilder turns a Session (+ ToolRegistry) into a prompt string: instructions +
tool list + transcript (User/Action/Observation). PromptFormat wraps the body for a
specific model — GemmaFormat closes the gap of a missing chat_template (manual <start_of_turn>
tags), PlainFormat — neutrally (for models with their own template).
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .session import Session
from .tools import ToolRegistry

__all__ = ["PromptFormat", "PlainFormat", "GemmaFormat", "QwenFormat", "ReActPromptBuilder"]


@runtime_checkable
class PromptFormat(Protocol):
    def wrap(self, body: str) -> str: ...


class PlainFormat:
    """No wrapping — body as-is (the model applies its own chat template)."""
    def wrap(self, body: str) -> str:
        return body


class GemmaFormat:
    """Manual Gemma format (Gemma-4 has NO chat_template set)."""
    def wrap(self, body: str) -> str:
        return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


class QwenFormat:
    """Manual Qwen format (<|im_start|>). think=False inserts a closed empty <think>
    block — equivalent to enable_thinking=False: the model answers right away, without reasoning
    (needed for the no-think agentic loop and compatibility with a no-think-trained Doubter).
    think=True leaves the reasoning channel open (for thinking variants)."""
    def __init__(self, think: bool = False):
        self.think = think

    def wrap(self, body: str) -> str:
        head = f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
        return head if self.think else head + "<think>\n\n</think>\n\n"


_DEFAULT_SYSTEM = (
    "You are an agent that solves a task step by step using tools. "
    "Think briefly and call a tool when you need data or a computation."
)


class ReActPromptBuilder:
    """Session (+ tools) -> prompt. Callable: prompt_builder(session) -> str."""
    def __init__(self, tools: ToolRegistry, fmt: Optional[PromptFormat] = None,
                 system: Optional[str] = None):
        self.tools = tools
        self.fmt = fmt or PlainFormat()
        self.system = system or _DEFAULT_SYSTEM

    def _instructions(self) -> str:
        return "\n".join([
            self.system,
            "",
            "Available tools:",
            self.tools.describe(),
            "",
            "Call format — output EXACTLY one line: Action: <tool>[<argument>]",
            "In return you will get: Observation: <result>",
            "When ready to answer — output: Answer: <final answer>",
        ])

    def _transcript(self, session: Session) -> str:
        out: list[str] = []
        for m in session.messages:
            if m.role == "user":
                out.append(f"User: {m.content}")
            elif m.role == "tool":
                out.append(f"Observation: {m.content}")
            elif m.role == "assistant" and m.name == "action":
                out.append(f"Action: {m.content}")
            elif m.role == "assistant":
                out.append(f"Assistant: {m.content}")
            # system messages are already accounted for in the instructions
        return "\n".join(out)

    def __call__(self, session: Session) -> str:
        body = self._instructions() + "\n\n" + self._transcript(session)
        return self.fmt.wrap(body)
