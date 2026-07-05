"""Reference generator for the C++ anchor transformer-encoder validation.

Builds a TINY TransformerEncoder (the GoalAnchor encoder), runs it in PyTorch on fixed
activations, and dumps:
  anchor_sidecar.gguf   — the encoder tensors + shape metadata (via export_anchor_sidecar)
  anchor_acts.bin       — input activations [NL, HID] float32, layer-major
  anchor_ref_cog.bin    — PyTorch reference cognitive tokens [NL, HID] float32, token-major

The C++ (meta_anchor_encoder) reads the gguf + acts → cog and compares to the reference.
Run: python validate_anchor_encoder.py <out_dir>
"""
import json
import sys
from pathlib import Path

import torch

from meta_core.encoders.transformer import TransformerEncoder

HID, NL, ED, NB, NH, FFN = 64, 4, 16, 2, 2, 4


def main(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    enc = TransformerEncoder(hidden_dim=HID, num_layers=NL, encoder_dim=ED, num_blocks=NB,
                             num_heads=NH, ffn_expansion=FFN, dropout=0.0,
                             use_per_layer_projectors=True, use_layer_pos_embeddings=True)
    enc.eval()

    acts = [torch.randn(1, HID) for _ in range(NL)]
    with torch.no_grad():
        ref = enc(acts)[0]                                  # [NL, HID]

    # a goal_anchor checkpoint (format 1.1) with an empty CA — encoder-only validation
    ckpt = {
        "format_version": "1.1", "kind": "goal_anchor",
        "config": {"encoder_dim": ED, "encoder_num_heads": NH, "encoder_num_blocks": NB,
                   "encoder_ffn_expansion": FFN, "ca_bottleneck_dim": 8, "ca_num_heads": 2,
                   "trigger": "always", "use_layer_pos_embeddings": True},
        "target_layers": list(range(NL)), "cross_attn_layers": list(range(NL)),
        "encoder_state": enc.state_dict(), "ca_state": {},
    }
    ckpt_path = out / "anchor_tiny.pt"
    torch.save(ckpt, ckpt_path)

    from meta_deploy.export import export_anchor_sidecar
    export_anchor_sidecar(str(ckpt_path), hidden_dim=HID,
                          out=str(out / "anchor_sidecar.gguf"), verbose=False)

    # acts.bin: layer-major [NL][HID]
    a = torch.stack([acts[i][0] for i in range(NL)], dim=0).contiguous().float()
    a.numpy().tofile(out / "anchor_acts.bin")
    # ref cog: token-major [NL][HID]
    ref.contiguous().float().numpy().tofile(out / "anchor_ref_cog.bin")

    (out / "anchor_shapes.json").write_text(json.dumps(
        {"HID": HID, "NL": NL, "ED": ED, "NB": NB, "NH": NH, "FFN": FFN}), encoding="utf-8")
    print(f"ref cog [{NL},{HID}] range [{ref.min():.4f},{ref.max():.4f}] → {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/anchor_val")
