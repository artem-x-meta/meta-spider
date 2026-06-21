"""Family-agnostic discovery of the base model's structure.

The framework must not be tied to a family (Llama/Gemma/Qwen/GPT/…). The points where it
couples to a concrete architecture — locating the decoder-layer stack and the sizes — are
gathered here.

`find_decoder_layers` looks for the `nn.ModuleList` of decoder blocks: first by known paths
(including nested `*.language_model.*` for multimodals like Gemma 4), then as a fallback —
recursively by matching the length against the number of layers from the config (incl. `text_config`).
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

__all__ = ["find_decoder_layers", "infer_num_layers", "infer_hidden_dim"]


# Multimodal configs hide the text parameters under text_config.
def _configs(model: Any):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return []
    out = [cfg]
    for sub in ("text_config", "llm_config", "language_config"):
        c = getattr(cfg, sub, None)
        if c is not None:
            out.append(c)
    return out


def infer_num_layers(model: Any) -> int:
    """Number of decoder layers from config (or the nested text_config)."""
    for c in _configs(model):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(c, attr):
                return int(getattr(c, attr))
    raise ValueError("Could not find num_hidden_layers / n_layer / num_layers in config (incl. text_config)")


def infer_hidden_dim(model: Any) -> int:
    """Hidden state dimensionality from config (or the nested text_config)."""
    for c in _configs(model):
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(c, attr):
                return int(getattr(c, attr))
    raise ValueError("Could not find hidden_size / n_embd / d_model in config (incl. text_config)")


# Known paths to the decoder-layer stack, from most common to multimodal.
_KNOWN_PATHS = (
    ("model", "layers"),                    # Llama / Gemma 2-3 / Qwen / Mistral
    ("transformer", "h"),                   # GPT-2 / GPT-Neo
    ("model", "language_model", "layers"),  # a number of multimodals (HF >=4.x)
    ("language_model", "model", "layers"),  # alternative nesting
    ("language_model", "layers"),           # encoder-free unified (e.g. Gemma 4)
    ("model", "decoder", "layers"),         # decoder-style stack
    ("decoder", "layers"),
)


def _resolve(model: Any, path: tuple[str, ...]) -> Optional[Any]:
    obj = model
    for attr in path:
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def find_decoder_layers(model: Any) -> nn.ModuleList:
    """Find the `nn.ModuleList` of the base model's decoder blocks, family-agnostically.

    1) known paths (incl. nested language_model in multimodals);
    2) fallback — recursive search for a ModuleList of length == num_hidden_layers from config;
    3) last resort — the longest ModuleList in the model.
    """
    for path in _KNOWN_PATHS:
        obj = _resolve(model, path)
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj

    # Fallback: look for a stack whose length matches the number of layers from config.
    try:
        target = infer_num_layers(model)
    except ValueError:
        target = None

    longest: Optional[nn.ModuleList] = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if target is not None and len(mod) == target:
                return mod
            if longest is None or len(mod) > len(longest):
                longest = mod

    if longest is not None:
        return longest
    raise ValueError(
        f"Could not find a decoder-layer stack in {type(model).__name__}. "
        f"Add the path to meta_core.model_utils._KNOWN_PATHS."
    )
