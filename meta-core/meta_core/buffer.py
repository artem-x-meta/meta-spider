"""ReflexionBuffer — storage for cognitive tokens between Pass 1 and Pass 2.

A simple container: on Pass 1 the encoder computes cognitive_tokens and writes them
into the buffer via `fill()`. On Pass 2 the BottleneckCrossAttention forward hooks read
the buffer via `expand_batch(batch_size)` for cross-attention K/V.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["ReflexionBuffer"]


class ReflexionBuffer:
    """In-memory storage for cognitive tokens.

    Args:
        num_slots: number of slots (= number of cognitive tokens the encoder outputs).
        hidden_dim: dimensionality of each token (= base model's hidden_dim).
        device: where to place the slots. If None — CPU; on `fill()` the buffer moves to
            the device of the incoming tensor.
    """

    def __init__(self, num_slots: int, hidden_dim: int, device: Optional[str] = None):
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.device: Optional[torch.device] = torch.device(device) if device is not None else None
        slot_device = self.device if self.device is not None else torch.device("cpu")
        self.slots: torch.Tensor = torch.zeros(1, num_slots, hidden_dim, device=slot_device)
        self.is_filled: bool = False

    def fill(self, cognitive_tokens: torch.Tensor) -> None:
        """Write the cognitive_tokens from Pass 1 into the buffer.

        Args:
            cognitive_tokens: [B, num_slots, hidden_dim] from the encoder.
        """
        assert cognitive_tokens.shape[1] == self.num_slots, (
            f"Expected {self.num_slots} tokens, got {cognitive_tokens.shape[1]}"
        )
        self.device = cognitive_tokens.device
        self.slots = cognitive_tokens.to(self.device)
        self.is_filled = True

    def get_tokens(self) -> torch.Tensor:
        """Get the cognitive tokens. Used by the CA hooks on Pass 2."""
        return self.slots

    def expand_batch(self, batch_size: int) -> torch.Tensor:
        """Broadcast slots across the batch for feeding into CA forward.

        Valid cases: slots batch == batch_size (per-prompt tokens, e.g. training), or
        slots batch == 1 broadcast to N (beams / repeats of the SAME prompt). A filled
        multi-prompt buffer hitting a different batch size is a misconfiguration — raise
        loudly instead of letting expand() fail cryptically (or silently feeding prompt-1's
        tokens to other prompts).

        Returns:
            [batch_size, num_slots, hidden_dim] — without allocation, via expand.
        """
        b = self.slots.shape[0]
        if b not in (1, batch_size):
            raise RuntimeError(
                f"ReflexionBuffer batch mismatch: buffer holds {b} prompt(s), the forward "
                f"batch is {batch_size}. Refill the buffer for this batch (Pass 1 per batch).")
        return self.slots.expand(batch_size, -1, -1)

    def clear(self) -> None:
        """Reset the buffer (between different prompts)."""
        slot_device = self.device if self.device is not None else torch.device("cpu")
        self.slots = torch.zeros(1, self.num_slots, self.hidden_dim, device=slot_device)
        self.is_filled = False
