"""Modifiers — trainable components that read the base's activations and, if needed,
inject a correction back.

- **Doubter**: activation encoder → cognitive tokens → cross-attention
  feedback into the base. Reactive — responds to signs of failure without an explicit task.

The architecture allows several parallel modifiers on a single pipeline
(`Modifier` — base class for custom implementations).
"""

from meta_core.modifiers.base import Modifier
from meta_core.modifiers.doubter import Doubter

__all__ = ["Modifier", "Doubter"]
