"""Policy — the bridge between dialogue state and action.

The loop (runtime.py) depends only on the protocol Policy.act(session) -> AgentAction.
The engine knows nothing about the model: FakePolicy (tests, no GPU) or BackendPolicy
(any InferenceBackend + renderer) is plugged in. Inference itself lives behind the
InferenceBackend seam (backends.py), which lazily pulls in its dependency.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence, Union, runtime_checkable

from .action import AgentAction, ActionRenderer, RefusalToolRenderer
from .backends import InferenceBackend, MetaSpiderBackend
from .session import Session

__all__ = ["Policy", "FakePolicy", "BackendPolicy", "MetaSpiderPolicy"]


@runtime_checkable
class Policy(Protocol):
    def act(self, session: Session) -> AgentAction: ...


class FakePolicy:
    """Deterministic policy for tests/skeleton without a model.

    script — a sequence of reactions: AgentAction (as-is) or str (run through the
    renderer). Exercises the loop and the refusal→tool seam without inference.
    """
    def __init__(self, script: Sequence[Union[AgentAction, str]],
                 renderer: Optional[ActionRenderer] = None):
        self.script = list(script)
        self.renderer = renderer or RefusalToolRenderer()
        self._i = 0

    def act(self, session: Session) -> AgentAction:
        if self._i >= len(self.script):
            return AgentAction.final("")
        item = self.script[self._i]
        self._i += 1
        if isinstance(item, AgentAction):
            return item
        return self.renderer.render(item, signal=None, session=session)


class BackendPolicy:
    """Generic policy: any InferenceBackend + ActionRenderer.

    Builds a prompt from the session → backend.generate → renderer(text, signal) → action.
    The backend decides HOW inference is computed (in-process Meta-Spider / llama.cpp / remote);
    the renderer decides HOW a latent decision becomes an action. meta-agent depends neither
    on the model nor on the training toolkit.
    """
    def __init__(self, backend: InferenceBackend,
                 renderer: Optional[ActionRenderer] = None,
                 prompt_builder: Optional[Callable[[Session], str]] = None):
        self.backend = backend
        self.renderer = renderer or RefusalToolRenderer()
        self.prompt_builder = prompt_builder   # callable(session)->str; None = v0 assembly

    def _prompt(self, session: Session) -> str:
        if self.prompt_builder is not None:
            return self.prompt_builder(session)
        # v0-fallback: last question + accumulated tool observations.
        obs = [m for m in session.messages if m.role == "tool"]
        if not obs:
            return session.last_user
        tail = "\n".join(f"[{m.name}] {m.content}" for m in obs)
        return f"{session.last_user}\n\nTool results:\n{tail}"

    def act(self, session: Session) -> AgentAction:
        out = self.backend.generate(self._prompt(session))
        return self.renderer.render(out.text, out.signal, session)


def MetaSpiderPolicy(config, checkpoint: Optional[str] = None,
                     renderer: Optional[ActionRenderer] = None,
                     max_new_tokens: int = 64,
                     apply_chat_template: bool = True) -> BackendPolicy:
    """Convenience constructor: BackendPolicy over MetaSpiderBackend (research path).

    A thin wrapper — loads the Meta-Spider core and (optionally) attaches a Doubter.
    """
    backend = MetaSpiderBackend.from_meta_spider(
        config, checkpoint=checkpoint,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template)
    return BackendPolicy(backend, renderer=renderer)
