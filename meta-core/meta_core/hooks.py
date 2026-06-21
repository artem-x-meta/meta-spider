"""ActivationCollector — passive observer for the base model's hidden states.

Registers a `register_forward_hook` on each target layer of the base. On every forward
pass it captures the hidden state at the specified position (by default the last token —
the decision point for autoregressive generation).

Supports freeze/unfreeze for compatibility with `model.generate()`: on Pass 1 the hooks
are active and capture the prompt's activations, then before Pass 2 they are frozen so as
not to overwrite the buffer with every generated token.

Supported architectures: Llama / Gemma / Qwen / Mistral (via `model.model.layers`),
GPT-2 / GPT-Neo (via `model.transformer.h`).
"""

from __future__ import annotations

from typing import Any, Literal

import torch

__all__ = ["ActivationCollector"]


PoolMode = Literal["last", "mean", "concat_last_mean"]


class ActivationCollector:
    """Forward-hook collector for multi-layer hidden states.

    Args:
        model: HF causal LM with `.model.layers` (Llama/Gemma/Qwen) or `.transformer.h` (GPT-2).
        target_layers: indices of the layers to register hooks on.
        pool: how to reduce seq_len → a single vector.
            - "last": hidden[:, -1, :] (Phase 1-4 default, the only one supported right now)
            - "mean": hidden.mean(dim=1) (Phase 5 port, Level 2)
            - "concat_last_mean": cat(last, mean) → 2× hidden_dim (Phase 5 record)
    """

    def __init__(
        self,
        model: Any,
        target_layers: list[int],
        pool: PoolMode = "last",
    ):
        if pool != "last":
            raise NotImplementedError(
                f"pool={pool!r} — Phase 5 port, Level 2. "
                f"Only pool='last' is supported right now."
            )
        self.model = model
        self.target_layers = sorted(target_layers)
        self.pool = pool
        self._frozen: bool = False
        self._handles: list[Any] = []
        self._activations: dict[int, torch.Tensor] = {}
        self.attach()

    def _get_layers(self) -> Any:
        """Find the list of transformer layers, family-agnostically (see model_utils)."""
        from meta_core.model_utils import find_decoder_layers
        return find_decoder_layers(self.model)

    def _make_hook(self, layer_idx: int):
        """Build a closure hook for a specific layer."""

        def hook(module, inputs, output):
            if self._frozen:
                return
            hidden_states = output[0] if isinstance(output, tuple) else output
            # [batch, seq_len, hidden_dim] → last token (the decision point)
            last_token = hidden_states[:, -1, :].detach().clone()
            self._activations[layer_idx] = last_token

        return hook

    def attach(self) -> None:
        """Register forward hooks on the target_layers."""
        layers = self._get_layers()
        for layer_idx in self.target_layers:
            if layer_idx >= len(layers):
                continue
            handle = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def freeze(self) -> None:
        """Stop recording new activations (for Pass 2 / generate).
        The hooks stay registered but write nothing to the buffer.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume recording activations."""
        self._frozen = False

    def get_snapshot(self) -> dict[int, torch.Tensor]:
        """Return a copy of the current activation dict {layer_idx: [batch, hidden_dim]}."""
        return dict(self._activations)

    def get_multi_layer_tensor(self) -> torch.Tensor:
        """Gather activations from all target_layers into [batch, n_layers, hidden_dim].

        Layers are ordered by index (ascending).
        """
        if not self._activations:
            raise RuntimeError(
                "No activations collected. Run a base-model forward before calling."
            )
        ordered = [self._activations[i] for i in self.target_layers if i in self._activations]
        # Each element [batch, hidden_dim]; stack on a new axis → [batch, n_layers, hidden_dim]
        return torch.stack(ordered, dim=1)

    def get_activation_list(self) -> list[torch.Tensor]:
        """Gather activations into a list of [batch, hidden_dim] tensors, one per target_layer.

        Convenient for feeding into `Encoder.forward(activation_list)` (SelectiveEncoder /
        TransformerEncoder expect exactly a list, not a stacked tensor).
        """
        if not self._activations:
            raise RuntimeError(
                "No activations collected. Run a base-model forward before calling."
            )
        return [self._activations[i] for i in self.target_layers if i in self._activations]

    def clear(self) -> None:
        """Clear the activation buffer."""
        self._activations.clear()

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
