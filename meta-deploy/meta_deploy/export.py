"""checkpoint.pt → GGUF sidecar for llama.cpp (deploy phase M1).

Dumps ALL encoder_state + ca_state tensors "as is" (encoder-agnostic:
multi_token / selective / transformer) as float32 + config into GGUF metadata. The C++ side
(cpp/) loads the sidecar via `gguf_init_from_file` and fetches tensors by name.

Naming convention in the sidecar:
  enc.<orig_key>          — encoder tensors (e.g. enc.queries, enc.layer_projectors.0.1.weight)
  ca.<layer>.<orig_key>   — CA tensors of a specific cross_attn layer (e.g. ca.6.q_proj.weight)
meta_spider.* metadata: encoder_type, hidden_dim, bottleneck_dim, num_cognitive_tokens,
  ca_num_heads, enc_num_heads, target_layers[], cross_attn_layers[], gate_activation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

ARCH = "meta-spider"


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

    w = GGUFWriter(str(out), ARCH)

    # --- metadata (C++ reads sizes/encoder type from here → doesn't hardcode) ---
    w.add_string("meta_spider.encoder_type", str(cfg.get("encoder_type", "unknown")))
    w.add_uint32("meta_spider.hidden_dim", int(hidden_dim))
    w.add_uint32("meta_spider.bottleneck_dim", int(cfg.get("ca_bottleneck_dim", 256)))
    w.add_uint32("meta_spider.num_cognitive_tokens", int(cfg.get("num_cognitive_tokens", 8)))
    w.add_uint32("meta_spider.ca_num_heads", int(cfg.get("ca_num_heads", 8)))
    w.add_uint32("meta_spider.enc_num_heads", int(cfg.get("transformer_num_heads", 8)))
    w.add_array("meta_spider.target_layers", [int(x) for x in target_layers])
    w.add_array("meta_spider.cross_attn_layers", [int(x) for x in cross_attn_layers])
    w.add_string("meta_spider.gate_activation", "tanh")  # C++: cur += tanh(gate)*CA(...)

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
