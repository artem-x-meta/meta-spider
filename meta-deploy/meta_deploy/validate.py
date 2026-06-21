"""Check the numpy encoder spec (spec.py) against the PyTorch implementation — phase M2a.

No base needed: we run random activations [num_layers × hidden] through both paths and compare
cognitive tokens. If they match (diff<1e-3), the port arithmetic is correct and ggml/C++
translates spec.py.

The encoder is imported directly from `meta_core` (editable install); --framework-path is only a
fallback for a tree that isn't installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _import_multi_token(framework_path: Optional[str]):
    try:
        from meta_core.encoders.multi_token import MultiTokenEncoder
        return MultiTokenEncoder
    except ImportError:
        if framework_path:
            sys.path.insert(0, str((Path(framework_path) / "meta-core").resolve()))
            sys.path.insert(0, str(Path(framework_path).resolve()))
        from meta_core.encoders.multi_token import MultiTokenEncoder
        return MultiTokenEncoder


def validate_encoder(framework_path: Optional[str], checkpoint: str, *,
                     hidden_dim: int = 2304, num_layers: int = 5) -> bool:
    """numpy multi_token_forward == PyTorch MultiTokenEncoder on random activations.
    Returns True if diff<1e-3.
    """
    import numpy as np
    import torch

    MultiTokenEncoder = _import_multi_token(framework_path)
    from . import spec

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    es = ck["encoder_state"]
    num_cog = int(cfg["num_cognitive_tokens"])
    bottleneck = int(cfg["encoder_bottleneck"])
    num_heads = int(cfg["transformer_num_heads"])

    enc = MultiTokenEncoder(hidden_dim=hidden_dim, num_layers=num_layers,
                            num_cognitive_tokens=num_cog, bottleneck_dim=bottleneck,
                            num_heads=num_heads)
    enc.load_state_dict(es)
    enc.eval()

    torch.manual_seed(0)
    acts_t = [torch.randn(1, hidden_dim) for _ in range(num_layers)]
    with torch.no_grad():
        ref = enc(acts_t)[0].numpy()

    w = {k: v.detach().float().numpy().astype(np.float64) for k, v in es.items()}
    acts_np = [a[0].numpy().astype(np.float64) for a in acts_t]
    got = spec.multi_token_forward(acts_np, w, num_layers, num_cog,
                                   bottleneck, hidden_dim, num_heads)

    diff = np.abs(ref - got)
    cos = float(ref.flatten() @ got.flatten()
                / (np.linalg.norm(ref) * np.linalg.norm(got)))
    ok = bool(diff.max() < 1e-3)
    print(f"cog tokens: ref {ref.shape}  numpy {got.shape}")
    print(f"max abs diff:  {diff.max():.3e}")
    print(f"cosine sim:    {cos:.8f}")
    print("VERDICT:", "OK — arithmetic matched" if ok else "MISMATCH — check the op order")
    return ok
