"""M2b helper: dump a test case for the ggml encoder.

Generates seed-0 random activations [5×2304], runs the PyTorch multi_token encoder →
reference cognitive tokens [8×2304]. Dumps both as raw float32 (little-endian, C-order):
  enc_testcase_acts.bin  — 5*2304 f32 (layer by layer)
  enc_testcase_ref.bin   — 8*2304 f32 (token by token)
The C++ program reads acts, computes its own cog → enc_testcase_cpp.bin; compare checks them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework-path", default=None, help="(vestigial — meta_core is installed editable)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden-dim", type=int, default=2304)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from meta_core.encoders.multi_token import MultiTokenEncoder

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    enc = MultiTokenEncoder(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                            num_cognitive_tokens=int(cfg["num_cognitive_tokens"]),
                            bottleneck_dim=int(cfg["encoder_bottleneck"]),
                            num_heads=int(cfg["transformer_num_heads"]))
    enc.load_state_dict(ck["encoder_state"])
    enc.eval()

    torch.manual_seed(0)
    acts = [torch.randn(1, args.hidden_dim) for _ in range(args.num_layers)]
    with torch.no_grad():
        ref = enc(acts)[0].numpy().astype(np.float32)  # [num_cog, hidden]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    acts_np = np.stack([a[0].numpy() for a in acts], 0).astype(np.float32)  # [num_layers, hidden]
    acts_np.tofile(out / "enc_testcase_acts.bin")
    ref.tofile(out / "enc_testcase_ref.bin")
    print(f"acts {acts_np.shape} → enc_testcase_acts.bin")
    print(f"ref  {ref.shape} → enc_testcase_ref.bin")


if __name__ == "__main__":
    main()
