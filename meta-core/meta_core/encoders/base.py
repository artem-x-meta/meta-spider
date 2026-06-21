"""Encoder ABC — the common interface for cognitive encoders.

Every built-in and user encoder takes a list of activations from all of the base model's
target_layers and returns cognitive tokens — a tensor [batch, n_layers, hidden_dim], where
each "token" is the representation of the corresponding layer in the base's embedding space.

These cognitive tokens are later fed into the `BottleneckCrossAttention` modules
during Pass 2 to inject the meta-signal back into the base model.

Implementations:
  - `SelectiveEncoder` — feedforward (per-layer projectors + gates + output_proj)
  - `TransformerEncoder` — mini-transformer (per-layer projectors + N×TransformerBlock)

A third-party user can register their own architecture via
`register_encoder("name", MyEncoder)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["Encoder"]


class Encoder(ABC):
    """Base interface for a cognitive-token encoder.

    Subclasses are `nn.Module` + Encoder ABC. They must implement forward.
    """

    @abstractmethod
    def forward(self, activation_list: list["torch.Tensor"]) -> "torch.Tensor":
        """Transform activations into cognitive tokens.

        Args:
            activation_list: list of `n_layers` tensors of shape [batch, hidden_dim],
                one per target_layer of the base model.

        Returns:
            cognitive_tokens: [batch, n_layers, hidden_dim] — the meta-signal to
                feed into the BottleneckCrossAttention modules.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """The base model's hidden state dimensionality (= cognitive token dimensionality)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of target layers (= number of output cognitive tokens)."""
        raise NotImplementedError
