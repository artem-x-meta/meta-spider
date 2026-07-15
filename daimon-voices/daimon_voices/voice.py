"""Voice — the Daimon addition to the mechanism's `Injector` protocol.

`meta_attention.Injector` says only *when* the mechanism will call you. A Voice is that plus the
opinions this framework has formed about **how** to use it:

  * a LIFECYCLE. Three disciplines turned out to matter, and they are not interchangeable:
      reactive     — rebuild the buffer from each prompt's own activations   (Doubter)
      persistent   — encode once, hold across generations                    (GoalAnchor)
      accumulating — a rolling bank of episodes                              (Chronographer)
    The mechanism is indifferent; the voice is not.

  * a CHECKPOINT contract (`kind`, config, encoder + CA states) — via `meta_attention.checkpoint`,
    the same format the ggml implementation reads.

  * a GAIN fader (`set_gain`) — voices sum on the residual stream, each with its own volume.
    That is the mixing-console property: you can turn one down at inference without retraining.

Named after the Socratic daimonion: an inner voice that counsels but does not rule.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from meta_attention import MetaAttentionPipeline

__all__ = ["Voice"]


class Voice(ABC):
    """Base class for a Daimon voice — satisfies `meta_attention.Injector` structurally.

    Subclasses must implement at least `on_post_forward` — the deciding logic of
    "intervene or not". The remaining hooks have default no-op implementations.
    """

    name: str = "voice"

    def __init__(self) -> None:
        self._attached: bool = False
        self._pipeline: Optional["MetaAttentionPipeline"] = None

    @abstractmethod
    def on_post_forward(self, activations: dict[int, Any]) -> Optional[dict[str, Any]]:
        """Called after Pass 1. Receives a snapshot of the activations.

        Returns:
            None — if no intervention is needed.
            dict — a control signal for the pipeline (for example
                   `{"action": "refuse"}` or `{"action": "steer", "direction": tensor}`).
        """
        raise NotImplementedError

    def on_attach(self, pipeline: "MetaAttentionPipeline") -> None:
        """Called on attach to the pipeline. By default — store the reference."""
        self._pipeline = pipeline
        self._attached = True

    def on_detach(self) -> None:
        """Remove all of the voice's registered hooks."""
        self._attached = False
        self._pipeline = None

    def on_pre_forward(self) -> None:
        """Clear local buffers before a new forward."""
        pass

    def on_generate_step(self, step: int, token_id: int) -> Optional[dict[str, Any]]:
        """Called on each generated token of Pass 2.

        Returns:
            None — continue unchanged.
            dict — an adaptive control signal (for voices with trigger logic).
        """
        return None

    # --- Persistence ---

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs: Any) -> "Voice":
        """Load the voice's trained weights."""
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save the voice's current weights."""
        raise NotImplementedError

    # ---------------- the mixing console ----------------

    def set_gain(self, gain: float) -> None:
        """Runtime volume of this voice (1.0 = as trained, 0.0 = silent).

        Voices sum on the residual stream; this is the fader. Not part of the mechanism —
        an injector could hard-wire its strength; a *voice* is expected to be adjustable.
        Default: scale every cross-attention gate the voice owns.
        """
        ca = getattr(self, "ca_modules", None) or getattr(self, "memory_ca_layers", None)
        if ca is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement set_gain (no ca_modules found)")
        for m in ca.values():
            if hasattr(m, "set_gain"):
                m.set_gain(gain)
