"""meta-deploy — "train in PyTorch, deploy in llama.cpp".

The trained Doubter sidecar (encoder + BottleneckCA, ~94-188M params) is fully
decoupled from the FROZEN base. llama.cpp can't train, but inference doesn't need
that — we port only the sidecar forward. Result: calibrated refusal ("I answer
confidently or honestly decline") runs on any hardware where llama.cpp runs — CPU,
Metal, phones, without CUDA or PyTorch at all.

Deployment pipeline:
  1. `export_sidecar` / `export_from_run_dir` — sidecar checkpoint.pt → GGUF sidecar.
  2. ggml/C++ forward (cpp/) — encoder + CA, load the sidecar, compute cognitive tokens
     and the injection (validated against PyTorch bit-for-bit ~1e-6).
  3. End-to-end injection into llama.cpp generation (build_cvec hook) — see cpp/llama_hook/.

Public Python API (export side; the C++ forward is a separate build):
  export_sidecar(checkpoint, target_layers, cross_attn_layers, hidden_dim, out)
  export_from_run_dir(run_dir, out=None)   — metaloom convention (reads run.json)
"""
from .export import export_from_run_dir, export_sidecar

__all__ = ["export_sidecar", "export_from_run_dir"]
__version__ = "0.0.1"
