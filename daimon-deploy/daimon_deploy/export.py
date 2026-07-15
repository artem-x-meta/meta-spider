"""checkpoint.pt → GGUF sidecar for llama.cpp (deploy phase M1).

Dumps ALL encoder_state + ca_state tensors "as is" (encoder-agnostic:
multi_token / selective / transformer) as float32 + config into GGUF metadata. The C++ side
(the meta-attention library, cpp/) loads the sidecar via `gguf_init_from_file` and fetches tensors by name.

Naming convention in the sidecar:
  enc.<orig_key>          — encoder tensors (e.g. enc.queries, enc.layer_projectors.0.1.weight)
  ca.<layer>.<orig_key>   — CA tensors of a specific cross_attn layer (e.g. ca.6.q_proj.weight)
daimon.* metadata: encoder_type, hidden_dim, bottleneck_dim, num_cognitive_tokens,
  ca_num_heads, enc_num_heads, target_layers[], cross_attn_layers[], gate_activation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

ARCH = "daimon"


def export_sidecar(checkpoint_path: str, *, target_layers: Sequence[int],
                   cross_attn_layers: Sequence[int], hidden_dim: int, out: str,
                   verbose: bool = True) -> Path:
    """Sidecar `checkpoint_path` → GGUF sidecar `out`. Returns the .gguf path.

    target_layers — Pass 1 read taps (which layers we read the last token's activations from).
    cross_attn_layers — CA injection points (build_cvec hook in C++). hidden_dim — base size.
    Encoder-agnostic: tensors are written as is, the encoder type goes into metadata.
    """
    import torch
    from gguf import GGUFWriter

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    enc = ck["encoder_state"]
    ca = ck["ca_state"]

    def f32(t):
        a = t.detach().to(torch.float32).cpu().numpy()
        return a.reshape(1) if a.ndim == 0 else a

    # kind: 'doubter' (buffer rebuilt each prompt) vs 'goal_anchor' (static anchor + trigger).
    # GoalAnchor uses a transformer encoder with different config keys (encoder_num_heads,
    # no encoder_type / num_cognitive_tokens) — map them so the C++ side reads uniform metadata.
    kind = str(ck.get("kind", "doubter"))
    is_anchor = kind == "goal_anchor"
    enc_type = "transformer" if is_anchor else str(cfg.get("encoder_type", "unknown"))
    enc_heads = int(cfg.get("encoder_num_heads", cfg.get("transformer_num_heads", 8)))
    num_cog = len(target_layers) if is_anchor else int(cfg.get("num_cognitive_tokens", 8))

    w = GGUFWriter(str(out), ARCH)

    # --- metadata (C++ reads sizes/encoder type from here → doesn't hardcode) ---
    w.add_string("daimon.kind", kind)
    w.add_string("daimon.encoder_type", enc_type)
    w.add_uint32("daimon.hidden_dim", int(hidden_dim))
    w.add_uint32("daimon.bottleneck_dim", int(cfg.get("ca_bottleneck_dim", 256)))
    w.add_uint32("daimon.num_cognitive_tokens", int(num_cog))
    w.add_uint32("daimon.ca_num_heads", int(cfg.get("ca_num_heads", 8)))
    w.add_uint32("daimon.enc_num_heads", enc_heads)
    w.add_array("daimon.target_layers", [int(x) for x in target_layers])
    w.add_array("daimon.cross_attn_layers", [int(x) for x in cross_attn_layers])
    w.add_string("daimon.gate_activation", "tanh")  # C++: cur += tanh(gate)*CA(...)

    # --- trigger metadata (GoalAnchor only): the C++ runtime gates re-injection by these ---
    if is_anchor:
        w.add_string("daimon.trigger", str(cfg.get("trigger", "always")))
        w.add_uint32("daimon.trigger_k", int(cfg.get("trigger_k", 100)))
        w.add_uint32("daimon.trigger_decision_layer", int(cfg.get("trigger_decision_layer", 9)))
        # transformer-encoder shape (C++ builds the encoder graph from these, not hardcoded)
        w.add_uint32("daimon.encoder_dim", int(cfg.get("encoder_dim", 384)))
        w.add_uint32("daimon.encoder_num_blocks", int(cfg.get("encoder_num_blocks", 2)))
        w.add_uint32("daimon.encoder_ffn_expansion", int(cfg.get("encoder_ffn_expansion", 4)))
        w.add_uint32("daimon.use_layer_pos_embeddings",
                     int(bool(cfg.get("use_layer_pos_embeddings", True))))

    # --- encoder tensors ---
    n_enc = 0
    for k, v in enc.items():
        if not torch.is_tensor(v):
            continue
        w.add_tensor(f"enc.{k}", f32(v))
        n_enc += 1

    # --- CA tensors by layer ---
    n_ca = 0
    for layer, sub in ca.items():
        for k, v in sub.items():
            if not torch.is_tensor(v):
                continue
            w.add_tensor(f"ca.{layer}.{k}", f32(v))
            n_ca += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    if verbose:
        size_mb = Path(out).stat().st_size / 1e6
        print(f"→ {out}  ({size_mb:.1f} MB)  encoder_type={cfg.get('encoder_type')}  "
              f"enc-tensors={n_enc}  ca-tensors={n_ca}", flush=True)
        print(f"   target_layers={list(target_layers)}  cross_attn_layers={list(cross_attn_layers)}  "
              f"num_cog={cfg.get('num_cognitive_tokens')}  ca_heads={cfg.get('ca_num_heads')}",
              flush=True)
    return Path(out)


def export_anchor_sidecar(checkpoint_path: str, hidden_dim: int, out: Optional[str] = None,
                          verbose: bool = True) -> Path:
    """GoalAnchor checkpoint → GGUF sidecar. Layers come from the checkpoint itself (v1.1
    stores target_layers/cross_attn_layers); only hidden_dim must be supplied (base size)."""
    import torch

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "goal_anchor":
        raise ValueError(f"not a GoalAnchor checkpoint (kind={ck.get('kind')!r}); "
                         "use export_sidecar / export_from_run_dir for a Doubter.")
    tl = ck.get("target_layers")
    cl = ck.get("cross_attn_layers")
    if not tl or not cl:
        raise ValueError("checkpoint has no target_layers/cross_attn_layers (need format ≥1.1)")
    out = out or str(Path(checkpoint_path).with_name("goal_anchor_sidecar.gguf"))
    return export_sidecar(str(checkpoint_path), target_layers=tl, cross_attn_layers=cl,
                          hidden_dim=int(hidden_dim), out=out, verbose=verbose)


def export_from_run_dir(run_dir: str, out: Optional[str] = None, verbose: bool = True) -> Path:
    """metaloom convention: reads run.json (hidden_dim/target/cross layers, fixed by collect)
    + doubter_checkpoint.pt → GGUF sidecar (default <run-dir>/doubter_sidecar.gguf).
    """
    rd = Path(run_dir)
    mpath = rd / "run.json"
    if not mpath.exists():
        raise FileNotFoundError(f"missing {mpath} — not a metaloom run-dir (needs collect+train)")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    ckpt = rd / "doubter_checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt} — run `metaloom train` first")
    hidden = manifest.get("hidden_dim")
    if hidden is None:
        raise ValueError("run.json has no hidden_dim — rebuild via `metaloom collect` "
                         "(it records hidden_dim/num_layers into the manifest)")
    out = out or str(rd / "doubter_sidecar.gguf")
    return export_sidecar(str(ckpt), target_layers=manifest["target_layers"],
                          cross_attn_layers=manifest["cross_attn_layers"],
                          hidden_dim=int(hidden), out=out, verbose=verbose)
