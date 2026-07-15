"""Numpy reimplementation of the sidecar forward — the PINNED SPEC for the ggml/C++ port (the meta-attention library, cpp/).

Why: before writing the ggml graph, we pin the exact arithmetic in numpy and check it against
PyTorch (`validate.py`). ggml then translates an already-verified spec → a C++ mismatch is
caught against numpy, not against elusive PyTorch. If numpy↔torch match (diff<1e-3), the port
math is correct.

Critical details: nn.MultiheadAttention in_proj layout [Wq;Wk;Wv], GELU (erf, not tanh),
LayerNorm eps=1e-5 with population variance.
"""
from __future__ import annotations

import numpy as np


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)        # population variance (like torch LayerNorm)
    return (x - mu) / np.sqrt(var + eps) * w + b


def linear(x, w, b=None):
    y = x @ w.T                           # nn.Linear: y = x @ Wᵀ + b, W [out, in]
    return y + b if b is not None else y


def gelu(x):
    from math import sqrt

    from scipy.special import erf         # nn.GELU() default = exact (erf)
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mha_cross(query, kv, in_w, in_b, out_w, out_b, num_heads):
    """nn.MultiheadAttention(batch_first) forward: Q from query, K/V from kv.

    in_w [3D, D] = [Wq;Wk;Wv], in_b [3D]. query [M, D], kv [N, D]. → [M, D].
    """
    M, D = query.shape
    N = kv.shape[0]
    hd = D // num_heads
    Wq, Wk, Wv = in_w[:D], in_w[D:2 * D], in_w[2 * D:]
    bq, bk, bv = in_b[:D], in_b[D:2 * D], in_b[2 * D:]
    Q = query @ Wq.T + bq
    K = kv @ Wk.T + bk
    V = kv @ Wv.T + bv
    Qh = Q.reshape(M, num_heads, hd).transpose(1, 0, 2)
    Kh = K.reshape(N, num_heads, hd).transpose(1, 0, 2)
    Vh = V.reshape(N, num_heads, hd).transpose(1, 0, 2)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(hd)
    attn = softmax(scores, axis=-1)
    out = attn @ Vh
    out = out.transpose(1, 0, 2).reshape(M, D)
    return out @ out_w.T + out_b


def multi_token_forward(acts, w, num_layers, num_cog, bottleneck, hidden, num_heads):
    """MultiTokenEncoder forward. acts: list[num_layers] of [hidden]. w: dict name→np[f64].
    → [num_cog, hidden]. (Pinned in M2a, diff 1e-6 vs PyTorch.)
    """
    # 1. per-layer projectors: LN + Linear + GELU
    proj = []
    for i in range(num_layers):
        x = acts[i]
        x = layernorm(x, w[f"layer_projectors.{i}.0.weight"], w[f"layer_projectors.{i}.0.bias"])
        x = linear(x, w[f"layer_projectors.{i}.1.weight"], w[f"layer_projectors.{i}.1.bias"])
        x = gelu(x)
        proj.append(x)
    kv = np.stack(proj, axis=0)                       # [num_layers, bottleneck]
    # 2. cross-attention: learned queries × kv
    cog = mha_cross(w["queries"], kv,
                    w["cross_attn.in_proj_weight"], w["cross_attn.in_proj_bias"],
                    w["cross_attn.out_proj.weight"], w["cross_attn.out_proj.bias"], num_heads)
    # 3. output_proj: LN + Linear + GELU + Linear; 4. output_norm
    x = layernorm(cog, w["output_proj.0.weight"], w["output_proj.0.bias"])
    x = linear(x, w["output_proj.1.weight"], w["output_proj.1.bias"])
    x = gelu(x)
    x = linear(x, w["output_proj.3.weight"], w["output_proj.3.bias"])
    x = layernorm(x, w["output_norm.weight"], w["output_norm.bias"])
    return x                                          # [num_cog, hidden]


def selective_forward(acts, w, num_layers, bottleneck, hidden):
    """SelectiveEncoder forward (the current Doubter encoder). acts: list[num_layers] of [hidden].
    → [num_layers, hidden] (1:1 layer→cognitive token).

    Difference from multi_token: NO learned-queries MHA. Instead, a per-layer scalar
    gate tanh(layer_gates.i) multiplies the projected layer; inter-layer links live in the CA via
    token_preference (not here). output_proj is applied per-token (over the last axis).
    """
    proj = []
    for i in range(num_layers):
        x = acts[i]
        x = layernorm(x, w[f"layer_projectors.{i}.0.weight"], w[f"layer_projectors.{i}.0.bias"])
        x = linear(x, w[f"layer_projectors.{i}.1.weight"], w[f"layer_projectors.{i}.1.bias"])
        x = gelu(x)
        g = np.tanh(np.asarray(w[f"layer_gates.{i}"]).reshape(()))   # scalar tanh(gate)
        proj.append(g * x)
    stacked = np.stack(proj, axis=0)                     # [num_layers, bottleneck]
    # output_proj per-token: LN + Linear(bn→hidden) + GELU + Linear(hidden→hidden); + output_norm
    x = layernorm(stacked, w["output_proj.0.weight"], w["output_proj.0.bias"])
    x = linear(x, w["output_proj.1.weight"], w["output_proj.1.bias"])
    x = gelu(x)
    x = linear(x, w["output_proj.3.weight"], w["output_proj.3.bias"])
    x = layernorm(x, w["output_norm.weight"], w["output_norm.bias"])
    return x                                             # [num_layers, hidden]

