"""Native tool-calling format — for tool-tuned instruct models (Qwen / Granite / Llama …).

Unlike [ReActPromptBuilder] (hand-written text "Action: tool[arg]" / "Observation: …", which live
instruct models do NOT hold — they append a fake Observation, no stop tokens), here the prompt is
built with the official `tokenizer.apply_chat_template(messages, tools=...)`, and the output is
parsed in the model's NATIVE tool format. This is the working path, validated on Qwen3.5/Gemma-4/Granite
in lab harnesses (multihop / rag-coding) — promoted here as a first-class API.

- `NativeToolPrompt` — prompt_builder, model-agnostic (the tokenizer emits its model's format itself).
- `NativeToolRenderer` — tool-output parser → `AgentAction` (Qwen-XML + JSON fallback; pattern overridable).
- `STOP_SEQUENCES` / `stops_for` — per-turn stop strings (needed by [StopBackend], else the model plays both roles).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .action import AgentAction
from .session import Session
from .tools import ToolRegistry

__all__ = ["NativeToolPrompt", "NativeToolRenderer", "tool_schemas", "STOP_SEQUENCES", "stops_for"]


def tool_schemas(tools: ToolRegistry, arg_desc: str = "the argument value") -> list[dict]:
    """ToolRegistry → OpenAI-style JSON schemas for `apply_chat_template(tools=...)`.

    A Tool carries only name/description/primary argument (`Tool.arg`) → we build a minimal
    single-string schema. For richer signatures pass ready schemas to NativeToolPrompt(schemas=...).
    """
    out = []
    for name in tools.names:
        t = tools.get(name)
        arg = t.arg if (t and t.arg) else "input"
        out.append({"type": "function", "function": {
            "name": name, "description": t.description if t else "",
            "parameters": {"type": "object",
                           "properties": {arg: {"type": "string", "description": arg_desc}},
                           "required": [arg]}}})
    return out


class NativeToolPrompt:
    """Session (+ ToolRegistry) → native prompt via `apply_chat_template(tools=...)`.

    The session.messages → chat-messages mapping (user / assistant-`tool_calls` / tool) is model-
    agnostic: the tokenizer renders its own model's format. `chat_template_kwargs` by default
    suppresses reasoning with BOTH keys (`enable_thinking` for Qwen/Gemma-it + `thinking` for Granite;
    a foreign key is an unused jinja variable) — the single source of truth for the thinking gotcha.
    """
    def __init__(self, tokenizer, tools: ToolRegistry, system: Optional[str] = None,
                 chat_template_kwargs: Optional[dict] = None, schemas: Optional[list[dict]] = None):
        self.tok = tokenizer
        self.tools = tools
        self.system = system
        self.ckw = chat_template_kwargs if chat_template_kwargs is not None \
            else {"enable_thinking": False, "thinking": False}
        self.schemas = schemas if schemas is not None else tool_schemas(tools)

    def __call__(self, session: Session) -> str:
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for m in session.messages:
            if m.role == "user":
                messages.append({"role": "user", "content": m.content})
            elif m.role == "assistant" and m.name == "action":
                # runtime writes the action as "name[arg_value]" → native tool_call
                mm = re.match(r"(\w+)\[(.*)\]", m.content, re.S)
                if mm:
                    name, val = mm.group(1), mm.group(2).strip()
                    tool = self.tools.get(name)
                    arg = tool.arg if (tool and tool.arg) else "input"
                    messages.append({"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": name, "arguments": {arg: val}}}]})
            elif m.role == "tool":
                messages.append({"role": "tool", "content": m.content, "name": m.name or "tool"})
            elif m.role == "assistant":
                messages.append({"role": "assistant", "content": m.content})
            # skip session-system — the system is set via self.system
        return self.tok.apply_chat_template(
            messages, tools=(self.schemas or None),   # empty list → None: clean chat prompt
            tokenize=False, add_generation_prompt=True, **self.ckw)


# Qwen-XML: <function=NAME>…<parameter=ARG>VALUE</parameter>. JSON: {"name":…,"arguments":{…}}.
_XML_CALL = re.compile(r"<function=(\w+)>.*?<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
_JSON_CALL = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.S)
_SPECIAL = re.compile(r"</?tool_call>|<think>|</think>|</?function[^>]*>|</?parameter[^>]*>")


class NativeToolRenderer:
    """Parses native tool output → `AgentAction`. Tries Qwen-XML, then JSON tool_call;
    otherwise — a final answer (with special tokens stripped). `pattern` is overridable for another model."""
    def __init__(self, tools: ToolRegistry, pattern: Optional[re.Pattern] = None,
                 strip: Optional[re.Pattern] = None):
        self.tools = tools
        self.pattern = pattern
        self.strip = strip or _SPECIAL

    def render(self, raw_output: str, signal, session) -> AgentAction:
        raw = raw_output or ""
        if self.pattern is not None:
            m = self.pattern.search(raw)
            if m:
                return self._from_match(m)
        mx = _XML_CALL.search(raw)
        if mx:
            return self._from_match(mx)
        mj = _JSON_CALL.search(raw)
        if mj:
            name = mj.group(1)
            try:
                args = json.loads(mj.group(2))
            except json.JSONDecodeError:
                args = {}
            return AgentAction.tool_call(name, args if isinstance(args, dict) else {})
        return AgentAction.final(self.strip.sub("", raw).strip())

    def _from_match(self, m: re.Match) -> AgentAction:
        name, val = m.group(1), m.group(3)
        tool = self.tools.get(name)
        key = tool.arg if (tool and tool.arg) else "input"
        return AgentAction.tool_call(name, {key: val.strip()})


# Per-turn stop strings (capture a tool_call OR a whole answer). Needed by StopBackend.
STOP_SEQUENCES = {
    "qwen": ("<|im_end|>", "<|im_start|>"),
    "granite": ("<|end_of_text|>", "<|start_of_role|>"),
    "llama": ("<|eot_id|>", "<|start_header_id|>"),
    "gemma": ("<end_of_turn>", "<start_of_turn>"),
}


def stops_for(model_name: str) -> tuple[str, ...]:
    """Stop strings by substring in the model name (qwen/granite/llama/gemma); otherwise empty."""
    n = (model_name or "").lower()
    for key, stops in STOP_SEQUENCES.items():
        if key in n:
            return stops
    return ()
