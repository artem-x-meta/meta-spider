"""OpenAI-compatible server over Daimon — the seam to external agent runtimes.

External harnesses (Hermes Agent, τ²-bench, anything OpenAI-shaped) talk HTTP and cannot host a
voice that lives in the FORWARD PASS. So the integration is inverted: **we serve the modified
model**, they point at us (`provider: custom, base_url: http://localhost:8000/v1`).

Two legs, one server — the same seam the rest of the framework has:

    GPU/research   MetaAttentionPipeline + ChronoAnchor (in-process, HF transformers)
    CPU/deploy     the patched llama.cpp fork (ggml encoder + CA injection), driven by
                   `llama-meta-anchor-session` — the anchor is encoded ONCE from the goal text
                   and held across turns (META_ANCHOR / META_GAIN)

Generation goes through an `InferenceBackend`; the LATENT KNOBS — which have no text channel —
go through a `LatentControl`. One control per leg:

    POST /v1/meta/goal   {"goal": "...", "session": "s1"}   → the session's latent goal
    POST /v1/meta/gain   {"gain": 0.8}                      → the daimon fader
    POST /v1/meta/reset  {"session": "s1"}                  → clear the episode bank

`/v1/chat/completions` takes an optional `"session"` (or the `X-Meta-Session` header). The goal
NEVER appears in the messages — that is the whole point of the channel.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = ["DaimonServer", "LatentControl", "ChronoAnchorControl", "LlamaCppAnchorControl",
           "parse_tool_calls", "from_pipeline", "from_llama_cpp", "build_app", "serve"]


# ══════════════════════════ latent control (one per leg) ══════════════════════════

@runtime_checkable
class LatentControl(Protocol):
    """The knobs that have no text channel. Implemented per serving leg."""

    def set_goal(self, goal: str) -> None: ...
    def set_gain(self, gain: float) -> None: ...
    def reset(self) -> None: ...
    def on_turn(self, transcript: str, turn: int) -> None: ...   # episode recording (may no-op)
    def stats(self) -> dict: ...


class ChronoAnchorControl:
    """GPU/research leg: an in-process ChronoAnchor on a MetaAttentionPipeline."""

    def __init__(self, anchor: Any):
        self.anchor = anchor

    def set_goal(self, goal: str) -> None:
        self.anchor.set_goal(goal)

    def set_gain(self, gain: float) -> None:
        self.anchor.set_gain(gain)

    def reset(self) -> None:
        if hasattr(self.anchor, "clear_bank"):
            self.anchor.clear_bank()

    def on_turn(self, transcript: str, turn: int) -> None:
        if hasattr(self.anchor, "on_step_end"):
            self.anchor.on_step_end(turn, _TextSession(transcript))

    def stats(self) -> dict:
        return {"goal": getattr(self.anchor, "goal_text", None),
                "bank_episodes": len(getattr(self.anchor, "bank", []) or [])}


class LlamaCppAnchorControl:
    """CPU/deploy leg: the patched llama.cpp fork.

    The anchor there is encoded ONCE from the goal text at process start (`META_ANCHOR`) and held
    across turns by `llama-meta-anchor-session`; the gain is `META_GAIN`. So `set_goal` RESPAWNS
    the session process with the new anchor — cheap (the cog is one Pass-1) and honest: the C++
    side has no runtime setter.

    ⚠ The goal text is tokenized RAW there, never chat-wrapped (a measured gotcha: wrapping the
    goal in a chat template silently weakens the anchor).
    """

    def __init__(self, backend_factory: Any, *, gain: float = 1.0):
        """backend_factory(goal: str | None, gain: float) -> InferenceBackend"""
        self.backend_factory = backend_factory
        self.goal: Optional[str] = None
        self.gain = gain
        self.backend: Optional[Any] = None
        self._respawn()

    def _respawn(self) -> None:
        self.backend = self.backend_factory(self.goal, self.gain)

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self._respawn()                     # anchor is set at spawn (META_ANCHOR)

    def set_gain(self, gain: float) -> None:
        self.gain = float(gain)
        self._respawn()                     # META_GAIN

    def reset(self) -> None:
        self._respawn()                     # fresh session process = fresh KV + anchor

    def on_turn(self, transcript: str, turn: int) -> None:
        """No-op: the C++ session driver holds its own KV/anchor across turns."""

    def stats(self) -> dict:
        return {"goal": self.goal, "gain": self.gain, "leg": "llama.cpp"}


# ══════════════════════════ server core ══════════════════════════

@dataclass
class _SessionState:
    goal: Optional[str] = None
    turns: int = 0
    meta: dict = field(default_factory=dict)


class DaimonServer:
    """Sessions + the latent goal + generation. Transport-agnostic (FastAPI wraps it in
    `build_app`; tests drive it directly, no HTTP)."""

    def __init__(self, backend: Any, *, control: Optional[LatentControl] = None,
                 tokenizer: Any = None, model_name: str = "daimon",
                 max_new_tokens: int = 256, record_episodes: bool = True):
        self.backend = backend                     # InferenceBackend: generate(prompt)->out
        self.control = control                     # LatentControl | None
        self.tokenizer = tokenizer                 # for chat templating (None → plain render)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.record_episodes = record_episodes
        self.sessions: dict[str, _SessionState] = {}
        self._active: Optional[str] = None

    # ---------------- sessions ----------------

    def _state(self, session: Optional[str]) -> _SessionState:
        key = session or "default"
        if key not in self.sessions:
            self.sessions[key] = _SessionState()
        return self.sessions[key]

    def _activate(self, session: Optional[str]) -> _SessionState:
        """Switch the latent state to this session (single process = one active bank).

        For truly concurrent sessions run one server per session. Sequential sessions (what an
        agent harness does) are exactly what this handles."""
        key = session or "default"
        st = self._state(key)
        if self.control is None or self._active == key:
            return st
        self.control.reset()
        if st.goal:
            self.control.set_goal(st.goal)
        self._active = key
        return st

    # ---------------- latent knobs ----------------

    def set_goal(self, goal: str, session: Optional[str] = None) -> dict:
        if self.control is None:
            raise RuntimeError("no LatentControl attached — nothing to condition on "
                               "(attach a ChronoAnchor or the llama.cpp anchor leg)")
        st = self._state(session)
        st.goal = goal
        self._active = None
        self._activate(session)
        return {"ok": True, "session": session or "default", "goal": goal}

    def set_gain(self, gain: float) -> dict:
        if self.control is None:
            raise RuntimeError("no LatentControl attached — no gain fader")
        self.control.set_gain(float(gain))
        return {"ok": True, "gain": float(gain)}

    def reset(self, session: Optional[str] = None) -> dict:
        st = self._state(session)
        st.turns = 0
        if self.control is not None:
            self.control.reset()
            if st.goal:
                self.control.set_goal(st.goal)
        return {"ok": True, "session": session or "default"}

    # ---------------- OpenAI shape ----------------

    def _render(self, messages: list[dict], tools: Optional[list] = None) -> str:
        msgs = [{"role": m.get("role", "user"), "content": m.get("content") or ""}
                for m in messages]
        if self.tokenizer is not None:
            kw: dict[str, Any] = dict(add_generation_prompt=True, tokenize=False)
            if tools:
                kw["tools"] = tools
            try:
                return self.tokenizer.apply_chat_template(msgs, **kw)
            except (TypeError, ValueError, AttributeError):
                pass
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"

    def chat(self, messages: list[dict], *, session: Optional[str] = None,
             tools: Optional[list] = None, max_tokens: Optional[int] = None,
             **_: Any) -> dict:
        st = self._activate(session)
        prompt = self._render(messages, tools)
        backend = getattr(self.control, "backend", None) or self.backend   # llama.cpp: respawned
        out = backend.generate(prompt)
        text = getattr(out, "text", out)
        st.turns += 1

        if self.record_episodes and self.control is not None:
            self.control.on_turn(prompt + str(text), st.turns)

        meta = {"session": session or "default", "goal_set": st.goal is not None}
        if self.control is not None:
            meta.update(self.control.stats())

        content, tool_calls = parse_tool_calls(str(text))
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0,
                         "finish_reason": "tool_calls" if tool_calls else "stop",
                         "message": message}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "meta": meta,
        }

    def models(self) -> dict:
        return {"object": "list",
                "data": [{"id": self.model_name, "object": "model", "owned_by": "daimon"}]}


# Форматы тул-вызовов живых моделей: Hermes/Qwen JSON в <tool_call>, Qwen-XML, голый JSON.
_TC_JSON = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
_TC_XML = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_TC_PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
_TC_BARE = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.S)
_SPECIAL = re.compile(r"</?tool_call>|</?think>|</?function[^>]*>|</?parameter[^>]*>")


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Model tool-call text → (clean_content, OpenAI-shaped tool_calls).

    Covers the Hermes/Qwen `<tool_call>{"name":…,"arguments":{…}}</tool_call>` form (what Hermes
    Agent and most local instruct models emit), the Qwen-XML form, and a bare JSON object.
    """
    calls: list[dict] = []

    for m in _TC_JSON.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
        except json.JSONDecodeError:
            continue

    if not calls:
        for m in _TC_XML.finditer(text or ""):
            args = {k: v for k, v in _TC_PARAM.findall(m.group(2))}
            calls.append({"name": m.group(1), "arguments": args})

    if not calls:
        for m in _TC_BARE.finditer(text or ""):
            try:
                calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
            except json.JSONDecodeError:
                continue

    if not calls:
        return (text or "").strip(), []

    openai_calls = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in calls]
    content = _SPECIAL.sub("", _TC_JSON.sub("", _TC_XML.sub("", text or ""))).strip()
    return content, openai_calls


class _TextSession:
    """Minimal Session-shaped object for ChronoAnchor.on_step_end (it calls .render())."""

    def __init__(self, text: str):
        self._text = text

    def render(self) -> str:
        return self._text


# ══════════════════════════ constructors ══════════════════════════

def from_pipeline(pipeline: Any, anchor: Any = None, *, model_name: str = "daimon",
                  max_new_tokens: int = 256) -> DaimonServer:
    """GPU/research leg: MetaAttentionPipeline (+ optional ChronoAnchor)."""
    from daimon_agent.backends import DaimonBackend
    backend = DaimonBackend(pipeline, max_new_tokens=max_new_tokens,
                                apply_chat_template=False)
    control = ChronoAnchorControl(anchor) if anchor is not None else None
    return DaimonServer(backend, control=control, tokenizer=pipeline.tokenizer,
                            model_name=model_name, max_new_tokens=max_new_tokens)


def from_llama_cpp(binary: str, gguf: str, sidecar: str, *, model_name: str = "daimon-cpp",
                   max_new_tokens: int = 256, extra_env: Optional[dict] = None
                   ) -> DaimonServer:
    """CPU/deploy leg: the patched llama.cpp fork (`llama-meta-anchor-session`).

    The goal is passed as META_ANCHOR at spawn (RAW, never chat-wrapped) — `set_goal` respawns.
    """
    import subprocess

    def factory(goal: Optional[str], gain: float):
        env = dict(os.environ, META_SIDECAR=sidecar, META_GAIN=str(gain), **(extra_env or {}))
        if goal:
            env["META_ANCHOR"] = goal              # RAW goal text — the measured gotcha
        cmd = [binary, "-m", gguf, "-n", str(max_new_tokens)]

        class _CppBackend:
            def generate(self, prompt: str):
                res = subprocess.run(cmd + ["-p", prompt], env=env, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace")
                return type("Out", (), {"text": res.stdout.strip(), "signal": None})()

        return _CppBackend()

    control = LlamaCppAnchorControl(factory)
    return DaimonServer(control.backend, control=control, model_name=model_name,
                            max_new_tokens=max_new_tokens, record_episodes=False)


# ══════════════════════════ HTTP (optional dependency) ══════════════════════════

def build_app(server: DaimonServer):
    """FastAPI app around the server core. Needs `pip install fastapi uvicorn`."""
    try:
        from fastapi import Body, FastAPI, Header, HTTPException
    except ImportError as exc:                                        # pragma: no cover
        raise ImportError("serving needs FastAPI: pip install fastapi uvicorn") from exc

    app = FastAPI(title="daimon OpenAI-compatible server")

    @app.get("/v1/models")
    def _models():
        return server.models()

    @app.post("/v1/chat/completions")
    def _chat(body: dict = Body(...), x_meta_session: Optional[str] = Header(None)):
        session = body.get("session") or x_meta_session
        return server.chat(body.get("messages", []), session=session,
                           tools=body.get("tools"), max_tokens=body.get("max_tokens"))

    @app.post("/v1/meta/goal")
    def _goal(body: dict = Body(...)):
        try:
            return server.set_goal(body["goal"], body.get("session"))
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/meta/gain")
    def _gain(body: dict = Body(...)):
        try:
            return server.set_gain(body["gain"])
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/meta/reset")
    def _reset(body: dict = Body(default={})):
        return server.reset(body.get("session"))

    return app


def serve(server: DaimonServer, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(build_app(server), host=host, port=port)
