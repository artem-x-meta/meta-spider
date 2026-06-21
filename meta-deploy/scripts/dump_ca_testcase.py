"""M3a/4a helper: dump a test case for the ggml BottleneckCrossAttention (cpp/meta_ca.cpp).

Self-contained: builds a random BottleneckCrossAttention, exports a sidecar (ca.<layer>.*),
runs the PyTorch CA forward → reference. Dumps raw f32 (little-endian, C-order):
  ca_hidden.bin  — seq*HID   (position by position)
  ca_cog.bin     — NCOG*HID  (token by token)
  ca_ref.bin     — seq*HID
C++ reads sidecar+hidden+cog → ca_cpp.bin; compare checks them (diff~1e-5 expected).

Usage: python dump_ca_testcase.py --out <dir> [--hidden-dim 64 --bottleneck 16 --num-heads 4
       --num-cog 5 --seq 3 --layer 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from meta_core.cross_attention import BottleneckCrossAttention
from meta_deploy import export_sidecar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--bottleneck", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-cog", type=int, default=5)
    ap.add_argument("--seq", type=int, default=3)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    ca = BottleneckCrossAttention(hidden_dim=args.hidden_dim, bottleneck_dim=args.bottleneck,
                                  num_heads=args.num_heads, num_cognitive_tokens=args.num_cog,
                                  dropout=0.0)
    with torch.no_grad():
        ca.gate.copy_(torch.tensor([0.5]))                       # gate in the linear zone
        ca.token_preference.copy_(torch.randn(args.num_cog) * 0.1)  # non-uniform bias
    ca.eval()

    ckpt = out / "ca_doubter.pt"
    torch.save({
        "config": {"encoder_type": "selective", "num_cognitive_tokens": args.num_cog,
                   "ca_bottleneck_dim": args.bottleneck, "ca_num_heads": args.num_heads,
                   "transformer_num_heads": 8},
        "encoder_state": {}, "ca_state": {args.layer: ca.state_dict()},
    }, ckpt)
    side = export_sidecar(str(ckpt), target_layers=[args.layer], cross_attn_layers=[args.layer],
                          hidden_dim=args.hidden_dim, out=str(out / "ca_sidecar.gguf"), verbose=False)

    hidden = torch.randn(1, args.seq, args.hidden_dim)
    cog = torch.randn(1, args.num_cog, args.hidden_dim)
    with torch.no_grad():
        ref = ca(hidden, cog)[0].numpy().astype(np.float32)      # [seq, HID]
    hidden[0].numpy().astype(np.float32).tofile(out / "ca_hidden.bin")
    cog[0].numpy().astype(np.float32).tofile(out / "ca_cog.bin")
    ref.tofile(out / "ca_ref.bin")
    print(f"sidecar={side} layer={args.layer}")
    print(f"hidden {tuple(hidden[0].shape)}, cog {tuple(cog[0].shape)}, ref {ref.shape}")


if __name__ == "__main__":
    main()
