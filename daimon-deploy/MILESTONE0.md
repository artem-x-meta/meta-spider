# llama.cpp Inference Deploy — "train in PyTorch, deploy in llama.cpp"

**Status: DEFERRED** (queue: Phase 1R canary → nf4 validation → late-CA → this)
**Recorded:** 12.06.2026, following the pipeline-scaling discussion.

## Idea

The trained Doubter sidecar (encoder + BottleneckCA, ~94-188M params) is fully
decoupled from the frozen base. llama.cpp can't train (autograd was stripped from ggml
in 2024), but for **inference** that isn't needed: port only the sidecar forward.

Outcome on success: a calibrated Doubter ("I answer confidently or honestly decline")
runs on any hardware where llama.cpp runs — CPU, Metal, phones, without CUDA
or PyTorch at all.

## Foothold in llama.cpp

**Control vectors** — an existing mechanism for intervening in the residual stream
(adding a vector to the hidden states of selected layers). A precedent for what we need,
only our case is harder: instead of a static vector — cross-attention over
cognitive tokens that depend on the current context.

## What to port (in increasing difficulty)

1. **Hidden state taps** — reading the activations of the target layers at the last token.
   llama.cpp has no hooks; access is needed via graph inspection or a build_graph patch.
2. **Encoder forward** (ggml graph): SelectiveEncoder — LN + Linear + GELU
   per-layer + gates. Trivial math, pure ggml.
3. **CA injection**: BottleneckCrossAttention (down 4096→256, MHA, up 256→4096,
   tanh-gate, residual) at the cross_attn_layers points. The most invasive part —
   intervening between the base graph's layers.
4. **Mode of operation**: two-pass is expensive on CPU → go straight for Phase 3 dynamic
   (refresh every K tokens + introspection cache, threshold 0.5 — validated).
5. **Weight export**: a checkpoint.pt → GGUF sidecar-tensor converter
   (a separate file next to the base GGUF, by analogy with LoRA adapters in llama.cpp).

## GGUF quant ladder — risk as a curve, not a binary

llama.cpp supports the whole compression spectrum: Q8_0 → Q6_K → Q5_K_M → Q4_K_M →
Q3_K → Q2_K (+ IQ variants). The activation shift relative to the bf16 base grows
going down the ladder:

- **Q8/Q6** — activations almost identical to bf16; the sidecar most likely works as-is
- **Q4** — noticeable drift; the sidecar may need adaptation
- **Q3/Q2** — likely breaks; interesting where exactly

Hence a **standalone valuable measurement**: the "refusal calibration vs degree of
base compression" curve — how much quantization does introspection survive? No one has measured this.
The deploy strategy follows from the curve: a recommended minimum quant + optional
quantization-aware sidecar adaptation for the lower rungs.

## Known risks

- **Activation shift** (see the curve above). A two-step pre-check BEFORE forking llama.cpp:
  1. Cheap: activations of one sample via HF bf16 vs bnb int8/nf4 — cos-sim +
     refusal-calibration drop (the bnb ladder as a GGUF proxy; the algorithms differ,
     but the order of magnitude of the drift will show). Compatible with the framework's nf4 validation.
  2. Precise: the same measurements on real GGUF activations — requires porting item 1
     (hidden state taps), done inside the experiment itself.
- **CA performance on CPU**: the bottleneck 256 makes CA cheap
  (~4.3M params/layer), but 32 layers × every K tokens — needs measuring.
- The llama.cpp fork will have to be maintained against upstream (it moves fast).

## Success criterion

Refusal behavior (refusal_precision within −5pp of the PyTorch inference of the same
sidecar) on a quantized base in llama.cpp, speed ≥ 0.7× of the clean base.

## Related

- The framework's compression ladder (nf4/int8) — closes "training on weak hardware",
  this experiment closes "inference on zero hardware".
- Phase 3 dynamic refresh + cache: `docs/.../phase3-dynamic-*.md`

---

# MILESTONE 0 — RESULTS (13.06.2026)

## What was done (without the C++ fork)

- Built llama.cpp on mingw+pip-cmake (patch: guard `THREAD_POWER_THROTTLING_EXECUTION_SPEED`).
- Gemma-2-2b GGUF ladder: f16 + Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q3_K_M/Q2_K.
- Custom tool `examples/extract-activations` (eval-callback fork): dumps
  the residual `l_out-{il}` of the target layers [10,14,18,22,25] for the last token.
  Notes: `add_bos=false` (prompts already contain `<bos>`), resume via
  EXTRACT_SKIP/MAX (works around the cumulative llama.cpp segfault after ~300 prompts).
- 500 test prompts from v9, activations extracted for all 7 base variants.

## Tap confirmed (cross-layer)

`l_out-{il}` corresponds exactly to the HF layer output: cross-layer cos-sim diagonal = max
for each layer (0.994-0.997), off-diagonal drops sharply (0.78→0.08). An off-by-one is
ruled out.

## Activation vs quant curve (cos-sim, 500 questions)

| Base | act vs HF fp16 | pure quantization (vs f16-GGUF) | **cog tokens vs HF** |
|---|---|---|---|
| F16 GGUF | 0.9944 | 1.0000 | 0.9705 |
| Q8_0 | 0.9944 | 0.9999 | 0.9706 |
| Q6_K | 0.9936 | 0.9989 | 0.9701 |
| Q5_K_M | 0.9925 | 0.9977 | 0.9701 |
| **Q4_K_M** | 0.9883 | 0.9931 | **0.9724** |
| Q3_K_M | 0.9771 | 0.9811 | 0.9389 |
| Q2_K | 0.9216 | 0.9238 | 0.9009 |

**Findings:**
1. **Engine floor 0.9944** — the llama.cpp f16 forward differs from HF fp16 even BEFORE any
   quantization (Gemma2 softcapping / sliding-window / precision of norms). Quantization
   is added on top of this floor. A measurement in its own right: the deploy engine has an
   inherent ~0.6% activation drift from the training stack.
2. **Cognitive tokens stay flat up to Q4_K_M** (cog_cos 0.970→0.972, Q8→Q4) —
   the encoder's input-LayerNorm absorbs the small activation drift. The self-awareness
   signal survives 4-bit compression almost losslessly.
3. **Break at Q3/Q2**: cog_cos 0.939 (Q3) → 0.901 (Q2), Q2 activations collapse
   (0.92). The applicability boundary is between Q4 and Q3.

Scripts: `scripts/{export_prompts,run_extract,offline_curve}.py`, `build_gguf.sh`.
Data: `results/gguf_activations_*.bin`, `results/curve_cossim.json`.

## Behavioral curve (nf4 Pass-2 on the 3050, cog tokens from GGUF Pass-1, 500 questions)

| Condition (cog tokens) | refusal precision | selective acc | refusal rate |
|---|---|---|---|
| HF fp16 (baseline, = v15) | 0.949 | 0.652 | 0.592 |
| Q8_0 | 0.976 | 0.667 | 0.586 |
| **Q4_K_M** | **0.976** | 0.693 | 0.590 |
| Q2_K | 0.974 | 0.741 | 0.614 |

**Main finding:** refusal precision is FLAT ~0.97 across all quants, including Q2_K —
even though the cog-token cos-sim on Q2 dropped to 0.90. cog-token degradation does NOT
break the binary refuse/answer decision down to Q2: the decision has margin. Selective accuracy
grows with quantization (0.65→0.74) — a coverage trade-off (Q2 answers less often, but more accurately).

Caveats: the 0.949-0.976 spread is at the edge of noise (CI ±~2pp at ~300 refusals) — correctly
"flat, ~0.95-0.98", not "quantization improves it". The HF baseline matches v15 (94.2%).
This is a PROXY: Pass-2 in nf4, cog from GGUF Pass-1 — isolates the effect of Pass-1 quantization.

## MILESTONE 0 VERDICT: GO ✅

Q4_K_M refusal precision 0.976 ≫ the 0.90 threshold. The self-awareness mechanism survives
4-bit compression almost losslessly (cos) and is behaviorally robust even to Q2 (refusal
precision). The main risk of the C++ branch (activation shift breaks the sidecar) is REMOVED by the data.

The C++ effort is justified: ggml encoder+CA + injection on the control-vector hook (`build_cvec`
in `src/models/gemma2.cpp`) + GGUF sidecar export + dynamic refresh (Phase 3).
Recommended deploy quant: Q4_K_M (1.7GB, balanced) — Q8/Q6 if memory allows,
Q3/Q2 risky by cos but still hold behaviorally.
