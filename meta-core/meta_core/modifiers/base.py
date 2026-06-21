"""Modifier ABC + lifecycle hooks.

A modifier is a component that attaches to `MetaSpiderPipeline` and participates in the
two-pass inference cycle. The base protocol provides hooks at the key moments:

- on_attach:        called when the modifier registers with the pipeline
                    (you may register your own forward hooks on model.layers)
- on_pre_forward:   before each forward pass (to clear buffers)
- on_post_forward:  after Pass 1 (cognitive tokens are ready, you may decide on intervention)
- on_generate_step: on each autoregressive step of Pass 2 (for adaptive steering)
- on_detach:        remove the registered hooks

The API is stabilized on Doubter; custom modifiers inherit Modifier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from meta_core.pipeline import MetaSpiderPipeline

__all__ = ["Modifier"]


class Modifier(ABC):
    """Base interface for a Meta-Spider modifier.

    Subclasses must implement at least `on_post_forward` — the deciding logic of
    "intervene or not". The remaining hooks have default no-op implementations.
    """

    name: str = "modifier"

    def __init__(self) -> None:
        self._attached: bool = False
        self._pipeline: Optional["MetaSpiderPipeline"] = None

    @abstractmethod
    def on_post_forward(self, activations: dict[int, Any]) -> Optional[dict[str, Any]]:
        """Called after Pass 1. Receives a snapshot of the activations.

        Returns:
            None — if no intervention is needed.
            dict — a control signal for the pipeline (for example
                   `{"action": "refuse"}` or `{"action": "steer", "direction": tensor}`).
        """
        raise NotImplementedError

    def on_attach(self, pipeline: "MetaSpiderPipeline") -> None:
        """Called on attach to the pipeline. By default — store the reference."""
        self._pipeline = pipeline
        self._attached = True

    def on_detach(self) -> None:
        """Remove all of the modifier's registered hooks."""
        self._attached = False
        self._pipeline = None

    def on_pre_forward(self) -> None:
        """Clear local buffers before a new forward."""
        pass

    def on_generate_step(self, step: int, token_id: int) -> Optional[dict[str, Any]]:
        """Called on each generated token of Pass 2.

        Returns:
            None — continue unchanged.
            dict — an adaptive control signal (for modifiers with trigger logic).
        """
        return None

    # --- Persistence ---

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs: Any) -> "Modifier":
        """Load the modifier's trained weights."""
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save the modifier's current weights."""
        raise NotImplementedError
