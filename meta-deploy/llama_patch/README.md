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

## GoalAnchor deploy — status & integration design

The behavior-modifier voice (GoalAnchor) deploys through the SAME machinery with three differences.
Status (2026-07):

- ✅ **GGUF export** — `metadeploy export` autodetects `kind=goal_anchor`; the sidecar carries
  `meta_spider.kind` + trigger metadata + the transformer-encoder shape (encoder_dim / num_blocks /
  enc_num_heads / ffn_expansion / use_layer_pos_embeddings).
- ✅ **Transformer-encoder ggml forward** — the anchor uses a `TransformerEncoder` (self-attention
  over the layer tokens), which the Doubter path did not have (selective / multi_token only). Ported
  to ggml in `cpp/meta_anchor_encoder.cpp`, **validated bit-for-bit vs PyTorch (max |Δ| 7e-7)** by
  `cpp/validate_anchor_encoder.py`. This was the missing primitive.
- ✅ **CA injection** — identical `BottleneckCrossAttention`; the Doubter's `apply_to` is reused as-is.
- ✅ **Decode-loop integration** — done in `llama_adapter_meta`:
  1. `encode` now has an `encoder_type=transformer` branch (the validated graph above, ported line-for-line
     into `src/llama-adapter.cpp`). Re-validated at production scale on the real v4.2 weights (16 layers,
     ED=384, NB=2, NH=8): the standalone twin matches PyTorch to **max |Δ| 3.8e-6**.
  2. **Lifecycle** (`META_ANCHOR` mode of `meta-generate`) — Pass-1 runs on the GOAL TEXT **once, before
     the prompt loop**; `llama_meta_encode` stores the cog in the adapter, and every prompt injects that
     static cog (no per-prompt Pass-1, no dynamic refresh). Simpler than the Doubter's per-prompt rebuild.
  3. **Trigger gating:** `trigger=always` injects every step — the validated regime; `fixed`/`learnable`
     would gate by the trigger metadata (not needed for always).
- ✅ **End-to-end on 14B (plumbing)** — runs on the real `Qwen2.5-14B-Instruct-Q4_K_M.gguf`: the sidecar
  loads, `[meta] anchor cog encoded from goal` fires (transformer `encode` on the 16-layer / ED=384 / NB=2
  config, no crash), injection is **coherent at gain 1.0–4.0** (CA gate + AGC keep it stable, no garbage)
  and **demonstrably changes the output** (base `divide_numbers` + its own docstring → anchor `divide`).
- ✅ **Anchor goal text must be RAW, not chat-wrapped.** `GoalAnchor.set_anchor(goal)` tokenizes the goal
  with `tok(goal_text)` — plain BOS + tokens, **no chat template** — and the encoder was trained on the raw
  structured `spec_text` (`"TASK: …\n\nREQUIREMENTS (all mandatory):\n1. …"`). The `META_ANCHOR` driver
  therefore tokenizes the goal RAW regardless of `META_RAW` (only the *prompt* gets `wrap()`). Feeding a
  chat-wrapped or hand-written goal encodes the cog from the wrong activations → the anchor barely acts.
  (This was a real driver bug, caught while chasing a phantom "Q4 weakens it" — quantization was not the
  cause; the Doubter showed quant barely matters.)
- ◑ **Drift-resistance reproduces (single faithful sample).** On a faithful mid-session slice (spec at t0,
  first `solve` solution, then a lure turn "rename `solve`… add print() to debug"), base drifts on BOTH
  constraints — renames to `divide_numbers` **and** emits active `print()` in the body — while the anchor
  keeps the function **silent** (holds `no_print`; prints only in a comment). `func_name` drifted in both
  (a stylistic family that erodes more). This is one eyeball, directionally consistent with the measured
  +19pp (v4.2), **not** a harness number. A full llama.cpp adherence figure needs the session harness
  ported over the whole spec set — the mechanism and the format are now correct.

### Run (anchor)

```bash
META_SIDECAR=goal_anchor_v42.gguf \
META_LAYERS=32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 \
META_ANCHOR="TASK: …\n\nREQUIREMENTS (all mandatory):\n1. …" \
META_RAW=1 META_PROMPT="<Qwen chat-formatted task/context here>" META_NGEN=256 \
./build/bin/llama-meta-generate -m qwen2.5-14b-instruct-q4_k_m.gguf -c 4096 -t 8
#   META_ANCHOR present → anchor mode: cog encoded ONCE from the goal (tokenized RAW, no chat
#   template — must match how GoalAnchor.set_anchor encodes it), held static every step.
#   META_BASE=1 → clean base (no anchor) for the A/B drift comparison.
```

### Run (multi-turn agentic session) — `llama-meta-anchor-session`

The agentic runtime: the anchor cog is encoded once at session start and **held across every turn**
(the conversation grows; the goal does not wash out). History is rendered with the model's own
**embedded chat template** (Qwen/Gemma/…), so no hand-rolled format. Observations (tool/feedback
turns) are injected after each assistant turn.

```bash
META_SIDECAR=goal_anchor_v42.gguf META_LAYERS=32,33,…,47 \
META_ANCHOR="TASK: …\n\nREQUIREMENTS (all mandatory):\n1. …"   # raw spec_text, encoded once \
META_SYSTEM="You are a precise coding assistant…" \
META_USER="<the initial task/spec turn>" \
META_OBS=observations.bin   # \0-separated user/tool turns injected after each assistant reply \
META_NGEN=160 \
./build/bin/llama-meta-anchor-session -m qwen2.5-14b-instruct-q4_k_m.gguf -c 4096 -t 8
#   META_BASE=1 → same conversation, no anchor (A/B). N observations → up to N+1 assistant turns.
```

Verified on 14B-Q4: encodes once, injects every turn, output stays coherent, base vs anchor differ.
**Caveat — you cannot read the +19pp adherence off a single session.** That effect is statistical
(base adheres ~0.56, anchor ~0.75), so any single eyeballed session drifts often in *both* arms —
seeing the gap needs aggregation over many specs (an adherence harness layered on this driver), not
one transcript. This driver is the *engine* for agentic use; the harness would be the *meter*.
