"""Cognitive encoders — trainable networks that turn the base's activations
into cognitive tokens [B, n_layers, hidden_dim] for cross-attention injection.

Two built-in architectures:

- **SelectiveEncoder**   — feedforward (per-layer projectors + encoder gates + output_proj).
  Phase 1-5 canon. Calibration record: selective accuracy 89.1% on Llama-8B MMLU.
- **TransformerEncoder** — mini-transformer (per-layer projectors + N×TransformerBlock).
  Phase 8 winner. Unlocked self-correction (correction_acc 50% on 1B).

Third-party users can add their own via `register_encoder("name", MyEncoder)`.
"""

from meta_core.encoders.base import Encoder
from meta_core.encoders.selective import SelectiveEncoder
from meta_core.encoders.transformer import TransformerEncoder
from meta_core.encoders.multi_token import MultiTokenEncoder

__all__ = [
    "Encoder",
    "SelectiveEncoder",
    "TransformerEncoder",
    "MultiTokenEncoder",
]
