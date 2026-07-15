"""Family-agnostic layer detection: find_decoder_layers / infer_* across different
structures (standard, GPT-2, nested multimodal, config fallback)."""

import pytest
import torch.nn as nn

from meta_attention.model_utils import find_decoder_layers, infer_hidden_dim, infer_num_layers


def _layers(n=4):
    return nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make(struct: str, n=4, h=8):
    """Build a fake model with the given nesting of the layer stack."""
    m = nn.Module()
    if struct == "standard":            # Llama/Gemma2-3/Qwen: model.model.layers
        m.model = nn.Module(); m.model.layers = _layers(n)
        m.config = _Cfg(num_hidden_layers=n, hidden_size=h)
    elif struct == "gpt2":              # transformer.h
        m.transformer = nn.Module(); m.transformer.h = _layers(n)
        m.config = _Cfg(n_layer=n, n_embd=h)
    elif struct == "mm_nested":         # multimodal: model.language_model.layers
        m.model = nn.Module(); m.model.language_model = nn.Module()
        m.model.language_model.layers = _layers(n)
        m.config = _Cfg(text_config=_Cfg(num_hidden_layers=n, hidden_size=h))
    elif struct == "unified":           # encoder-free: language_model.layers (Gemma 4-style)
        m.language_model = nn.Module(); m.language_model.layers = _layers(n)
        m.config = _Cfg(text_config=_Cfg(num_hidden_layers=n, hidden_size=h))
    elif struct == "weird":             # unknown path → fallback by length from config
        m.backbone = nn.Module(); m.backbone.blocks = _layers(n)
        # noise ModuleList of a different length (e.g. a "tower"), must not win
        m.distractor = _layers(n + 3)
        m.config = _Cfg(num_hidden_layers=n, hidden_size=h)
    return m


@pytest.mark.parametrize("struct", ["standard", "gpt2", "mm_nested", "unified", "weird"])
def test_find_decoder_layers(struct):
    m = _make(struct, n=5)
    layers = find_decoder_layers(m)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 5


@pytest.mark.parametrize("struct", ["standard", "gpt2", "mm_nested", "unified"])
def test_infer_dims(struct):
    m = _make(struct, n=6, h=16)
    assert infer_num_layers(m) == 6
    assert infer_hidden_dim(m) == 16


def test_no_layers_raises():
    m = nn.Module(); m.config = _Cfg(num_hidden_layers=4, hidden_size=8)
    with pytest.raises(ValueError):
        find_decoder_layers(m)
