# meta-deploy — "train in PyTorch, deploy in llama.cpp"

The fourth Meta-Spider component. Takes a trained Doubter wrapper (encoder + BottleneckCA,
~94–188M params) and **decouples it from PyTorch/CUDA** for inference on any hardware that runs
llama.cpp — CPU, Metal, phones.

The wrapper is fully separated from the **frozen** base, so only its forward needs porting (no
training inside llama.cpp). Result: calibrated refusal — "answer confidently or refuse honestly" —
runs on zero hardware, on a quantized GGUF base.

```text
meta-core ──provides──> trained doubter_checkpoint.pt
                              │
                   metadeploy export
                              ▼
        doubter_sidecar.gguf  +  ggml/C++ forward (cpp/)
                              ▼
                  llama.cpp (Q4_K_M base)  →  calibrated refusal on CPU
```

## Install

```bash
pip install -e meta-spider-framework/meta-deploy        # numpy + gguf (export side)
# torch is only needed to read the trained .pt (usually already present with meta-core)
```

## Quickstart — export the sidecar

```bash
# from a metaloom run-dir (reads run.json: hidden_dim + target/cross layers):
metadeploy export --run-dir lab/runs/qwen35_mcq
#   → lab/runs/qwen35_mcq/doubter_sidecar.gguf  (encoder_type=selective, enc+ca tensors)

# or manually:
metadeploy export --checkpoint d.pt --target-layers 10 14 18 22 25 \
                  --cross-attn-layers 6 12 18 24 --hidden-dim 2304 --out sidecar.gguf
```

Export is **encoder-agnostic** (multi_token / selective / transformer — tensors dumped as-is +
config into GGUF metadata; the C++ side reads sizes from metadata, no hardcoding).

## End-to-end deploy in llama.cpp (turnkey)

```bash
# 1. sidecar from the trained wrapper
metadeploy export --run-dir lab/runs/dogfood_qwen05b           # → doubter_sidecar.gguf

# 2. base in GGUF (any quant: f16 / Q4_K_M / …)
python llama.cpp/convert_hf_to_gguf.py <hf_snapshot> --outfile base.gguf --outtype f16

# 3. llama.cpp fork with meta-injection (see llama_patch/README.md) + two-pass run
META_SIDECAR=doubter_sidecar.gguf META_LAYERS=16,17,18,19,20,21,22,23 \
META_PROMPT="What is the capital of France?" META_NGEN=64 \
  ./build/bin/llama-meta-generate -m base.gguf -c 2048 -t 4
```

`META_LAYERS` = `cross_attn_layers` from `run.json`. `META_BASE=1` — clean base (for comparison).
The llama.cpp patch + how-to are in [`llama_patch/`](llama_patch/README.md). Helper: `scripts/run_meta_generate.sh`.

## Layout

| | what |
|---|---|
| `meta_deploy/export.py` | checkpoint.pt → GGUF sidecar (library API + `metadeploy export`) |
| `meta_deploy/spec.py` | numpy reference forward of the encoder — **the pinned spec for ggml** |
| `meta_deploy/validate.py` | numpy-spec vs PyTorch check (`metadeploy validate`) |
| `cpp/` | ggml/C++ forward: `meta_encoder.cpp` + `meta_ca.cpp` + CMake |
| `scripts/` | M0/M2 drivers: GGUF ladder, activation extraction, curves |
| `results/` | M0 data: cos-sim / behavioral curves, test cases, sidecar |
| `MILESTONE0.md` | full Milestone 0 results (quantization risk cleared) |

## Status

- **Milestone 0 (quantization risk) — GO ✅.** llama.cpp built, Gemma-2-2b GGUF ladder (f16→Q2_K),
  activation tap matched (cos 0.994–0.997). **Key finding:** refusal precision **flat ~0.97 down to
  Q2_K** — the self-awareness mechanism survives 4-bit compression almost losslessly. Recommended
  deploy quant: **Q4_K_M**. Details — [MILESTONE0.md](MILESTONE0.md).
- **GGUF export — ✅** (encoder-agnostic; verified on selective Qwen3.5-4B).
- **C++ encoder — ✅ validated against PyTorch** (diff ~1e-6): both multi_token (Gemma-2B) and
  **selective** — the current encoder of all Doubters (Qwen/Granite/Gemma-12B), max abs diff **8.9e-7**,
  cos 1.0000. Sizes are **generalized** — inferred from sidecar metadata + tensor shapes (no hardcoding).
  Build: `cpp/build.sh` (direct g++; `cpp/CMakeLists.txt` is the canonical dependency description).
- **C++ CA (BottleneckCrossAttention) — ✅ validated against PyTorch** (max abs diff **1.19e-7**),
  sizes generalized from the sidecar (HID/BN/NCOG/NHEAD).
- **End-to-end injection into llama.cpp generation — ✅ WORKS (turnkey).** The llama.cpp fork
  (`llama_patch/`) loads the sidecar as a meta-adapter, runs the built-in ggml encoder forward
  (selective+multi_token), and injects CA through the control-vector hook; `llama-meta-generate` is
  the two-pass driver. **Verified end-to-end on TWO families:** Qwen2.5-0.5B (base "…Paris…", Doubter →
  `I'm not confident enough to answer this question accurately.` verbatim as PyTorch) and
  **Granite-3.3-8B** (Q8, Llama family — base + injection both coherent). Turnkey is **not Qwen-only** —
  it transfers to 8B and the Llama arch. `train in PyTorch → deploy in llama.cpp` proven.
- **Prompt format:** `llama-meta-generate` wraps in Gemma tags by default; for non-Gemma models
  (Granite/Qwen/Llama) use `META_RAW=1` + your own chat format in `META_PROMPT` (otherwise foreign
  tags → garbage). TODO: apply the model's native chat template automatically.

## Success criterion (turnkey)

Refusal behavior within −5pp of the PyTorch inference of the same wrapper, on a quantized base inside
llama.cpp, at ≥ 0.7× the speed of the clean base.
