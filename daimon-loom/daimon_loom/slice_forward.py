"""Slice-forward: a partial pass through `layers[cut+1:]` from the cached `cut_hidden`.

The slice-trainer idea (training the 8-12B wrapper on 4GB). CA injection lives only in the
late layers → the bottom (`layers[0:cut+1]`, `cut = min(cross_attn)-1`) yields the SAME
hidden in pass-1 and pass-2 (there is no injection there). So we cache the output of
`layer[cut]` (`cut_hidden`, the full `[B,seq,H]`) and during training run the forward ONLY
through the slice `layers[cut+1:]` + norm + lm_head. The bottom is not computed (identity
patch) and under offload does not move to the GPU → peak VRAM = the slice.

The key decision (more robust than a manual loop over layers): we reuse HF `model.model.forward`
in full — RoPE/causal-mask/norm are computed as usual. The bottom is replaced with an identity
patch (forward returns the input without matmuls), and a `forward_pre_hook` on `layer[cut+1]`
substitutes the input with `cut_hidden`. This way there is no need to manually reconstruct
position_embeddings/the mask (the pitfall that makes a manual slice easily do a RoPE shift).

Correctness is checked bit-for-bit on a small model: `slice_last_hidden(...)` must
match `model.model(...).last_hidden_state` of a full pass (the same `cut_hidden`,
the same positions). See `lab/experiments/slice-trainer/validate_slice_forward.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from meta_attention.model_utils import find_decoder_layers

__all__ = [
    "base_module", "build_slice_device_map", "slice_device_map_for_model",
    "capture_cut_hidden", "slice_context", "slice_last_hidden",
]


def slice_device_map_for_model(model: Any, cut_layer: int, gpu: int = 0) -> dict:
    """model-aware device_map for slice-train + offload — derives the PATHS from the real structure.

    Why not hardcode `model.layers.N`: on Gemma-4 (encoder-free multimodal) the layers live at
    `model.language_model.layers.N`, on Llama — `model.layers.N`, on others — elsewhere. A hardcode
    would break. Here the prefixes are taken from `find_decoder_layers` + `named_modules`.

    Logic: by DEFAULT everything on `cpu`, on the GPU we raise ONLY the top slice (layers > cut) +
    the decoder's final norm/rotary + lm_head. The bottom layers, embed (not needed in the slice —
    we feed cut_hidden), and the multimodal's vision/audio towers stay on cpu (not computed). For
    bnb-offload the load must go with `llm_int8_enable_fp32_cpu_offload=True`.
    """
    layers = find_decoder_layers(model)
    lq = next(n for n, m in model.named_modules() if m is layers)  # "...layers"
    decoder_qual = lq.rsplit(".", 1)[0]                            # the decoder (parent of the layers)
    decoder = model.get_submodule(decoder_qual)
    embed = model.get_input_embeddings()
    embed_child = next((n for n, m in decoder.named_children() if m is embed), None)

    dm: dict = {"": "cpu"}  # everything on cpu by default (bottom, embed, multimodal towers)
    for i in range(len(layers)):
        if i > cut_layer:
            dm[f"{lq}.{i}"] = gpu  # the top slice is computed → GPU
    for cname, _ in decoder.named_children():
        if cname not in ("layers", embed_child):  # norm/rotary → GPU; embed stays cpu
            dm[f"{decoder_qual}.{cname}"] = gpu
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        lmq = next((n for n, m in model.named_modules() if m is lm), None)
        if lmq is not None:
            dm[lmq] = gpu  # lm_head on the GPU (needed for the loss)
    return dm


def build_slice_device_map(num_layers: int, cut_layer: int, gpu: int = 0) -> dict:
    """device_map for slice-train + offload: bottom (0..cut) → cpu, top + embed/rotary/norm/
    lm_head → GPU.

    Why not `device_map='auto'`: auto places the FIRST layers on the GPU (greedily from 0), but
    the slice computes the TOP → auto would put the computed part in RAM (slow) and waste the GPU
    on the bottom. Here the top (which the slice computes, where CA injects) is on the GPU; the
    bottom on cpu — the identity patch skips it, accelerate does NOT stream it to the GPU → peak
    VRAM = the slice (validated: validate_slice_offload.py). embed/rotary on the GPU →
    position_embeddings on the GPU (avoids a device-mismatch when injecting cut_hidden into the top).

    Module names are Llama/Mistral/Qwen-style (`model.layers.N`, `model.rotary_emb`).
    """
    dm: dict = {
        "model.embed_tokens": gpu,
        "model.rotary_emb": gpu,
        "model.norm": gpu,
        "lm_head": gpu,
    }
    for i in range(num_layers):
        dm[f"model.layers.{i}"] = "cpu" if i <= cut_layer else gpu
    return dm


def base_module(model: Any) -> Any:
    """The base transformer that owns the layers (`model.model` on Llama/Gemma/Qwen).

    The hook collector works family-agnostically via `find_decoder_layers`, but for the
    `forward` that returns `last_hidden_state` (after the final norm) you need the base module itself.
    """
    layers = find_decoder_layers(model)
    # The ModuleList's parent is the base (LlamaModel etc.).
    for module in model.modules():
        for child in module.children():
            if child is layers:
                return module
    # Fallback to common names.
    for attr in ("model", "transformer", "gpt_neox", "base_model"):
        sub = getattr(model, attr, None)
        if sub is not None and find_decoder_layers_safe(sub) is layers:
            return sub
    raise RuntimeError("Could not find the base module that owns the decoder layers")


def find_decoder_layers_safe(module: Any) -> Any:
    try:
        return find_decoder_layers(module)
    except Exception:
        return None


def capture_cut_hidden(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cut_layer: int,
) -> torch.Tensor:
    """The full `[B,seq,H]` output of `layer[cut_layer]` (NOT the last token) for the slice-trainer cache.

    Runs a base-forward under `no_grad`; a forward hook captures the cut layer's output in full.
    Right-padding + standard position_ids (HF from cache_position 0..seq-1) — the same as
    in `slice_last_hidden`, so the slice is consistent with the cache.
    """
    layers = find_decoder_layers(model)
    base = base_module(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs.detach()

    handle = layers[cut_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            base(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"the cut hook on layer[{cut_layer}] did not fire")
    return captured["h"]


@contextmanager
def slice_context(model: Any, cut_hidden: torch.Tensor, cut_layer: int) -> Iterator[None]:
    """A partial-slice context: identity patch on `layers[0:cut+1]` + injection of `cut_hidden`
    into the input of `layer[cut+1]`.

    Inside the context, any `base(input_ids, attention_mask)` computes ONLY the slice
    `layers[cut+1:]` (the bottom is identity, cheap / without weights on the GPU under offload),
    starting from `cut_hidden`. RoPE/mask/norm — standard HF.
    """
    layers = find_decoder_layers(model)
    inject_at = cut_layer + 1
    if inject_at >= len(layers):
        raise ValueError(f"cut_layer={cut_layer} — no layers above for the slice")

    patched: list[int] = []

    def _identity(*args, **kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        return (hs,)

    for i in range(inject_at):
        # HF decoder layers do not hold an instance attribute 'forward' (it's a class method);
        # we set an instance patch and remove it on exit → the class method is restored.
        layers[i].forward = _identity  # type: ignore[assignment]
        patched.append(i)

    def _pre_hook(module, args, kwargs):
        if args:
            args = (cut_hidden,) + tuple(args[1:])
        else:
            kwargs = {**kwargs, "hidden_states": cut_hidden}
        return args, kwargs

    handle = layers[inject_at].register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        for i in patched:
            try:
                del layers[i].forward  # restore the class method
            except AttributeError:
                pass


def slice_last_hidden(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cut_hidden: torch.Tensor,
    cut_layer: int,
) -> torch.Tensor:
    """The final post-norm hidden `[B,seq,H]` via the slice (from `cut_hidden`). Without lm_head.

    Equivalent to `model.model(input_ids, attention_mask).last_hidden_state` of a full pass
    (checked bit-for-bit), but the bottom is not computed.
    """
    base = base_module(model)
    with slice_context(model, cut_hidden, cut_layer):
        out = base(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state
