# llama.cpp patch — end-to-end Doubter injection (phase 4)

Integration of meta-spider into llama.cpp: loading a GGUF sidecar as a "meta adapter", a built-in
ggml encoder forward (**selective** and multi_token), CA injection into the residual via the
control-vector hook, and a `llama-meta-generate` example (two-pass inference). Implements phase 4 of
the turnkey deploy: `train in PyTorch → deploy in llama.cpp`.

**Base llama.cpp:** commit `d8a24cc` (tag `b9619`). On a different version the hunks may not apply —
go by the content (the integration points are stable: `llama_adapter_cvec::apply_to`,
`llama_context`, `examples/`).

## What's inside

- `meta-spider-llama.patch` — a diff of 6 files (426 lines):
  - `src/llama-adapter.{cpp,h}` — `llama_adapter_meta` (subclass of `llama_adapter_cvec`):
    `load` of the sidecar, built-in `encode` (**selective**: per-layer projector + scalar
    `tanh(layer_gates.i)`, no MHA; **multi_token**: queries × kv MHA — chosen by
    `meta_spider.encoder_type`), `apply_to` = BottleneckCrossAttention injection.
  - `src/llama-context.{cpp,h}` + `include/llama.h` — API: `llama_set_meta_adapter`,
    `llama_set_meta_cog`, `llama_meta_encode` (polymorphically replaces `cvec`).
  - `examples/CMakeLists.txt` — registration of the two examples.
- `examples/meta-generate/` — the two-pass driver (Pass-1 tap `cb_eval` → `llama_meta_encode`
  → Pass-2 CA injection; greedy + optional dynamic refresh). New files (copy them).
- `examples/extract-activations/` — the M0 activation-tap tool (for the quant curve). Copy it.

## Apply

**Scripted (recommended):** `scripts/build_llama.sh --backend {cuda,cpu,metal} [--arch 75] [--out DIR]`
does clone@b9619 + patch + copy examples + cmake configure/build, and includes the CUDA
`CUDA::cuda_driver target not found` fix (empty/read-only toolkit `stubs/` on Kaggle/Colab → points
cmake at the real `libcuda.so`). Manual steps below.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b9619                                   # base commit
git apply /path/to/meta-spider-llama.patch          # or patch -p1 <
cp -r /path/to/llama_patch/examples/meta-generate examples/
cp -r /path/to/llama_patch/examples/extract-activations examples/
cmake -B build -DLLAMA_CURL=OFF && cmake --build build --target llama-meta-generate -j
```

> **Windows/MSVC:** the binary will land in `build/bin/Debug/llama-meta-generate.exe` (not `build/bin/`).
> **Shallow clone:** with `git clone --depth 1` the `b9619` tag is missing — `git fetch origin tag b9619` (or
> a full clone). mingw without MSVC: `-G "Unix Makefiles" -DCMAKE_MAKE_PROGRAM=<...>/make` + g++ (like `cpp/build.sh`).

## Run (two-pass Doubter)

```bash
META_SIDECAR=doubter_sidecar.gguf \
META_LAYERS=16,17,18,19,20,21,22,23 \
META_PROMPT="What is the capital of France?" META_NGEN=64 \
./build/bin/llama-meta-generate -m base.gguf -c 2048 -t 4
#   META_BASE=1 — clean base (oracle); META_SIDECAR and META_LAYERS are still REQUIRED
#   META_GAIN=1.5 — injection-strength knob (default 1.0; parity with PyTorch set_gain; 0 ≈ base)
#   META_RAW=1 — prompt as is (non-Gemma models: pass your own chat format in META_PROMPT)
#   META_PROMPTS=<file> META_OUT=<file> — batch mode (\0-separated), one model load for many prompts
#   META_DYNAMIC=1 META_THRESHOLD=0.5 — periodic cog refresh (Phase 3 dynamic)
```

Verified end-to-end on Qwen2.5-0.5B (selective): the base answers "…Paris…", the Doubter →
`I'm not confident enough to answer this question accurately.` — verbatim like the PyTorch inference
of the same sidecar. The encoder sizes/type are read from the sidecar metadata (`metadeploy export`).
