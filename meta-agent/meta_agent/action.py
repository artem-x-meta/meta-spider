"""THE SEAM — the only non-standard part of the engine.

The Doubter's latent decision (gates / confidence-head) is not a token, so the
orchestrator does not see it. Here it is rendered into the standard action vocabulary
that an ordinary agentic loop reacts to: final | tool | abstain.

The injection stays latent (inside the wrapper's Pass 2); what comes out is a LEGITIMATE
model action — this is not text injection into the model's own reasoning context.

Right now the decision is read from TEXT (RefusalToolRenderer: "not sure" → call a
tool). The proper target is LatentActionRenderer: read the decision from signal
(gates/structural action token), not from text. This is the deferred ActionTokenAdapter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .tools import ToolRegistry

__all__ = [
    "AgentAction", "ActionRenderer", "RefusalToolRenderer", "LatentActionRenderer",
    "ReActRenderer", "looks_like_refusal",
]


@dataclass
class AgentAction:
    kind: str                              # "final" | "tool" | "abstain"
    content: str = ""                      # answer text (kind == "final")
    tool: Optional[str] = None             # tool name (kind == "tool")
    args: dict = field(default_factory=dict)

    @classmethod
    def final(cls, content: str) -> "AgentAction":
        return cls("final", content=content)

    @classmethod
    def tool_call(cls, tool: str, args: Optional[dict] = None) -> "AgentAction":
        return cls("tool", tool=tool, args=dict(args or {}))

    @classmethod
    def abstain(cls, content: str = "") -> "AgentAction":
        return cls("abstain", content=content)


@runtime_checkable
class ActionRenderer(Protocol):
    """Maps the raw model output (+ optional latent signal) into an action."""
    def render(self, raw_output: str, signal: Optional[dict],
               session: "object") -> AgentAction: ...


# The same phrases the Doubter detects (harness.classify_action). Duplicated
# locally so meta-agent has no hard dependency on meta-spider.
REFUSAL_PHRASES = (
    "not confident", "don't know", "do not know", "not sure", "i'm unsure",
    "i am unsure", "cannot answer", "can't answer", "unable to", "i don't know",
)


def looks_like_refusal(text: str) -> bool:
    # Opening-sentence rule (mirrors harness.classify_action): an answer followed by trailing
    # doubt is a commit, not a refusal — phrase-anywhere inflated refusal counts.
    a = (text or "").lower().strip()
    m = re.search(r"[.!?\n]", a)
    opening = a if m is None else a[: m.end()]
    return any(p in opening for p in REFUSAL_PHRASES)


class RefusalToolRenderer:
    """Current seam: the latent decision surfaced as a text refusal → call the
    fallback tool with the last question; otherwise — a final answer.

    Exactly the semantics of lab/.../tool_calibration.py (refusal = oracle call), lifted
    up to a real multi-turn loop.
    """
    def __init__(self, fallback_tool: str = "lookup", arg_name: str = "query"):
        self.fallback_tool = fallback_tool
        self.arg_name = arg_name

    def render(self, raw_output: str, signal: Optional[dict],
               session: "object") -> AgentAction:
        if looks_like_refusal(raw_output):
            query = getattr(session, "last_user", "") or raw_output
            return AgentAction.tool_call(self.fallback_tool, {self.arg_name: query})
        return AgentAction.final(raw_output)


class LatentActionRenderer:
    """TARGET SEAM (= ActionTokenAdapter): decision from the LATENT signal, not from text.

    signal contract (`InferenceOutput.signal`) — what the backend must set:
      {"refuse": bool}            — explicit latent decision (if the backend computed it); OR
      {"confidence": float 0..1}  — confidence to ANSWER; refuse if < threshold.
    If signal is empty/None → fall back to text (RefusalToolRenderer) — until the core emits
    a signal, the renderer degrades to the current behavior, but WITHOUT a rewrite.

    Source status (15.06.2026): the pipeline only exposes STATIC gates
    (get_ca_gate_map — a constant for any input), the per-question decision lives only in
    the text. A per-input readout is needed (ConfidenceHead over the activations at the
    decision point) — that is the unimplemented part of ActionTokenAdapter. See MetaSpiderBackend.
    """
    def __init__(self, fallback_tool: str = "lookup", arg_name: str = "query",
                 threshold: float = 0.5, text_fallback: Optional["ActionRenderer"] = None):
        self.fallback_tool = fallback_tool
        self.arg_name = arg_name
        self.threshold = threshold
        self.text_fallback = text_fallback or RefusalToolRenderer(fallback_tool, arg_name)

    def _latent_refuse(self, signal: Optional[dict]) -> Optional[bool]:
        """True/False = latent decision; None = no signal (fall back to text)."""
        if not signal:
            return None
        if "refuse" in signal:
            return bool(signal["refuse"])
        if "confidence" in signal:
            return float(signal["confidence"]) < self.threshold
        return None

    def render(self, raw_output: str, signal: Optional[dict],
               session: "object") -> AgentAction:
        decision = self._latent_refuse(signal)
        if decision is None:                       # no latent signal → text
            return self.text_fallback.render(raw_output, signal, session)
        if decision:                               # latent refusal → tool
            query = getattr(session, "last_user", "") or raw_output
            return AgentAction.tool_call(self.fallback_tool, {self.arg_name: query})
        return AgentAction.final(raw_output)       # confident → answer


_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\s*\[(.*?)\]", re.S)
_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.S)


class ReActRenderer:
    """Parses the model's ReAct output into an action (multi-tool real run).

      'Action: <tool>[<arg>]' → tool_call (arg placed under the primary name tool.arg);
      'Answer: <x>'           → final;
      text refusal            → abstain;
      otherwise               → the whole text as the final answer (lazily).
    """
    def __init__(self, tools: "ToolRegistry"):
        self.tools = tools

    def render(self, raw_output: str, signal: Optional[dict],
               session: "object") -> AgentAction:
        m = _ACTION_RE.search(raw_output or "")
        if m:
            name, content = m.group(1), m.group(2).strip()
            tool = self.tools.get(name)
            arg = tool.arg if tool is not None else "input"
            return AgentAction.tool_call(name, {arg: content})
        a = _ANSWER_RE.search(raw_output or "")
        if a:
            return AgentAction.final(a.group(1).strip())
        if looks_like_refusal(raw_output):
            return AgentAction.abstain((raw_output or "").strip())
        return AgentAction.final((raw_output or "").strip())
