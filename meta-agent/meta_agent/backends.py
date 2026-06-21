"""InferenceBackend — the "how inference is computed" seam, one level BELOW Policy.

Decouples meta-agent from the engine: the loop knows only Policy, Policy knows only
backend.generate(prompt) -> InferenceOutput. That is why meta-agent is independent of the
training toolkit (Meta-Loom) and holds no copy of the two-pass core (Meta-Spider) — each
backend lazily pulls in ITS OWN heavy dependency.

  MetaSpiderBackend — in-process GPU, lazy import of the Meta-Spider core (research)
  LlamaCppBackend   — CPU prod, calls the llama.cpp fork (already independent)   [stub]
  OpenAIBackend     — remote: meta-agent as a client to an OpenAI-compatible
                      server that does the two passes internally                 [stub]

InferenceOutput.signal — the slot for a latent decision (gates/confidence). None for now;
once it appears, LatentActionRenderer (ActionTokenAdapter) reads it instead of the text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

__all__ = [
    "InferenceOutput", "InferenceBackend", "StopBackend",
    "FakeBackend", "MetaSpiderBackend", "LlamaCppBackend", "OpenAIBackend",
]


@dataclass
class InferenceOutput:
    text: str
    # Latent decision, when the backend provides one. Contract (read by LatentActionRenderer):
    #   {"refuse": bool}            — explicit decision; OR
    #   {"confidence": float 0..1}  — confidence to answer (refuse if < threshold).
    # None → the renderer falls back to text.
    signal: Optional[dict] = None


@runtime_checkable
class InferenceBackend(Protocol):
    def generate(self, prompt: str) -> InferenceOutput: ...


class FakeBackend:
    """Scripted inference for testing the backend seam (no model).

    outputs — a sequence of str | (str, signal-dict).
    """
    def __init__(self, outputs: Sequence[Union[str, Tuple[str, dict]]]):
        self.outputs = list(outputs)
        self._i = 0

    def generate(self, prompt: str) -> InferenceOutput:
        if self._i >= len(self.outputs):
            return InferenceOutput("")
        item = self.outputs[self._i]
        self._i += 1
        if isinstance(item, tuple):
            return InferenceOutput(item[0], item[1])
        return InferenceOutput(item)


class MetaSpiderBackend:
    """In-process two-pass inference through the Meta-Spider core (GPU/research path).

    The meta_core import is lazy (inside from_meta_spider) — the package stays
    self-contained: it depends ONLY on the thin Meta-Core, not on the training toolkit
    (Meta-Loom).
    """
    def __init__(self, pipeline, max_new_tokens: int = 64,
                 apply_chat_template: bool = True, repetition_penalty: float = 1.0):
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.apply_chat_template = apply_chat_template
        self.repetition_penalty = repetition_penalty   # >1.0 damps the repetition loop (Doubter loop)

    def generate(self, prompt: str) -> InferenceOutput:
        gen_kw = dict(max_new_tokens=self.max_new_tokens,
                      apply_chat_template=self.apply_chat_template)
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            gen_kw["repetition_penalty"] = self.repetition_penalty   # opt.: don't break backends without it
        text = self.pipeline.generate(prompt, **gen_kw)
        # signal=None: the pipeline only exposes STATIC gates (Doubter.get_ca_gate_map —
        # a constant for any input), while the per-question decision lives only in the text.
        # To fill signal={"confidence": ...}, the core needs a per-input readout: run the
        # ConfidenceHead over the activations at the decision point and return a scalar from
        # generate(). This is the unimplemented part of ActionTokenAdapter (requires changes
        # to Meta-Core + GPU validation: does the readout correlate with actual refusal?).
        # For now — text.
        return InferenceOutput(text, signal=None)

    @classmethod
    def from_meta_spider(cls, config, checkpoint: Optional[str] = None, **kwargs):
        from meta_core import MetaSpiderPipeline, Doubter  # lazy
        pipeline = MetaSpiderPipeline.from_pretrained(config)
        if checkpoint:
            pipeline.attach(Doubter.from_checkpoint(checkpoint))
        return cls(pipeline, **kwargs)


class StopBackend:
    """Wrapper over any backend: truncates output at stop strings (turn stop sequences).

    CRITICAL for the native tool format ([meta_agent.native]): without truncation the model
    does not stop at the end of its turn and appends a fake tool result itself (plays both
    roles) → never reads the real Observation. Stop strings come from
    `native.stops_for(model_name)` (Qwen `<|im_end|>`, Granite `<|end_of_text|>`, …).
    """
    def __init__(self, inner: "InferenceBackend", stops: Sequence[str]):
        self.inner = inner
        self.stops = tuple(stops)

    def generate(self, prompt: str) -> InferenceOutput:
        out = self.inner.generate(prompt)
        text = out.text
        cut = len(text)
        for s in self.stops:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        out.text = text[:cut].strip()
        return out


class LlamaCppBackend:
    """CPU prod path: calls the llama.cpp fork (ggml encoder+CA, build_cvec hook).

    This path is ALREADY independent of the Python wrapper (a separate C++ fork), so here
    there is only a thin binding. Stub until the binding is wired in.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LlamaCppBackend: the binding to the llama.cpp fork is not wired in yet "
            "(see project-llamacpp-deploy). Use MetaSpiderBackend for now.")

    def generate(self, prompt: str) -> InferenceOutput:  # pragma: no cover
        raise NotImplementedError


class OpenAIBackend:
    """Remote shim: meta-agent as a client to an OpenAI-compatible server that
    does the two passes INTERNALLY and returns a normal chat-completion (+ tool_calls).

    Lets external frameworks see us as an ordinary model. Stub until the server exists.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OpenAIBackend: the OpenAI-compatible server around the pipeline is not up yet "
            "(polish phase). Use MetaSpiderBackend for now.")

    def generate(self, prompt: str) -> InferenceOutput:  # pragma: no cover
        raise NotImplementedError
