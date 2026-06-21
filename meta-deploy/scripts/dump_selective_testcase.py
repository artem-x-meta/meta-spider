"""M2b helper: dump a test case for the ggml selective encoder (cpp/meta_selective.cpp).

Builds a random SelectiveEncoder (small sizes), exports a sidecar, runs PyTorch →
reference cognitive tokens. Dumps raw float32 (little-endian, C-order):
  sel_acts.bin  — NL*HID f32 (layer by layer)
  sel_ref.bin   — NL*HID f32 (token by token; NCOG==NL for selective)
C++ reads sidecar+acts, computes cog → sel_cpp.bin; compare checks them (diff~1e-5 expected).

Usage: python dump_selective_testcase.py --out <dir> [--hidden-dim 64 --num-layers 5 --bottleneck 16]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from meta_core.encoders.selective import SelectiveEncoder
from meta_deploy import export_sidecar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--bottleneck", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    enc = SelectiveEncoder(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                           bottleneck_dim=args.bottleneck)
    with torch.no_grad():
        for i, g in enumerate(enc.layer_gates):
            g.copy_(torch.tensor([0.1 * (i + 1)]))   # different gates per layer
    enc.eval()

    # checkpoint for export (ca_state empty — we only test the encoder)
    ckpt = out / "selective_doubter.pt"
    torch.save({
        "config": {"encoder_type": "selective", "num_cognitive_tokens": args.num_layers,
                   "ca_bottleneck_dim": args.bottleneck, "ca_num_heads": 8,
                   "transformer_num_heads": 8},
        "encoder_state": enc.state_dict(), "ca_state": {},
    }, ckpt)
    side = export_sidecar(str(ckpt), target_layers=list(range(args.num_layers)),
                          cross_attn_layers=[], hidden_dim=args.hidden_dim,
                          out=str(out / "selective_sidecar.gguf"), verbose=False)

    acts = [torch.randn(1, args.hidden_dim) for _ in range(args.num_layers)]
    with torch.no_grad():
        ref = enc(acts)[0].numpy().astype(np.float32)             # [NL, HID]
    np.stack([a[0].numpy() for a in acts], 0).astype(np.float32).tofile(out / "sel_acts.bin")
    ref.tofile(out / "sel_ref.bin")
    print(f"sidecar={side}")
    print(f"acts {args.num_layers}x{args.hidden_dim} → sel_acts.bin   ref {ref.shape} → sel_ref.bin")


if __name__ == "__main__":
    main()
