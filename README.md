# Daimon

> Pluggable cognitive voices for any HuggingFace causal LM.
> Architectural introspection via meta-attention — no base fine-tuning.
>
> *Formerly published as `meta-spider`. The rename is part of v0.4.0: the framework is the
> **daimon** — the model's inner advisory voices (the Socratic daimonion: it counsels, it does
> not rule); the injection mechanism itself now lives in the separate
> [`meta-attention`](https://github.com/artem-x-meta/meta-attention) library (Apache-2.0).*

📖 **Documentation (with diagrams, EN/RU):** [imperius.codeberg.page/daimon](https://imperius.codeberg.page/daimon/)

```text
Text reflection:    model → "think again" → model re-reads text
Meta-attention:     model → activations → encoder → cognitive tokens
                          → cross-attention with tanh-gates → model
```

The base model is **frozen**. Only a thin wrapper is trained (~2.3% of params on 8B).
On Llama-3.1-8B Phase 2 Selective MMLU: **selective accuracy 89.1%** (base 64.6%) at 37% coverage —
the model answers far more reliably on what it chooses to answer (see the honest-metrics caveat below).

## New in v0.4.0

- **The rename: meta-spider → daimon.** The framework is the family of *voices* — the frozen
  model's inner advisers, summed on the residual stream, each with its own gain fader.
- **The mechanism is now a library.** Reading activations, encoders, gated cross-attention, the
  two-pass pipeline, the checkpoint format **and the whole C++/ggml + llama.cpp leg** moved to
  [`meta-attention`](https://github.com/artem-x-meta/meta-attention) (Apache-2.0). It knows
  nothing about doubt, goals or memory — a boundary test enforces that from both sides. The seam
  is the `Injector` protocol (library: *when the mechanism calls you*) vs the `Voice` contract
  (framework: Injector + lifecycle + checkpoint discipline + gain fader).
- **ChronoAnchor — goal-conditioned episodic memory.** GoalAnchor and Chronographer fused: the
  goal is not a second voice on the residual but a privileged INPUT of the memory organ. See
  "What it gives you" below for numbers.
- **GoalAnchor published** (early-stage): the goal-drift voice, the `metaloom build-anchor`
  factory and GGUF anchor export. Measured on Qwen2.5-14B agentic coding: constraint defense
  **+19pp** in-domain, transfer to unseen constraint families without quality loss.
- **OpenAI-compatible serving** (`daimon-agent`): serve a voice-equipped model behind
  `/v1/chat/completions` with latent side-channels (`/v1/meta/goal`, `/v1/meta/gain`) — two legs,
  HF pipeline and patched llama.cpp. Any OpenAI-client agent framework plugs in unchanged.

### Still true from v0.3.x

- **A latent channel beats a text prompt.** Fair 2×2 on Qwen2.5-14B (both arms allowed to
  abstain): declining the unanswerable — text prompt 0.07 → 0.07 (nothing), wrapper → **0.87**;
  catching the base's errors — wrapper **78%** vs a prompt's 14%.
  (`docs/results/qwen-14b/text-vs-latent.md`)
- **The uncertainty knob (gain).** `doubter.set_gain(x)` / `META_GAIN` in llama.cpp: gain 0 → 1.5
  turns refusals ~2% → 51% smoothly; `<0` inverts toward confidence.
- **The Watchdog (read-only sensor).** A `ConfidenceProbe` on the cognitive token gates an
  external action without injecting (`from daimon_voices.watchdog import ConfidenceProbe`).
- **The wrapper factory.** `metaloom build-universal --model-name N` builds a general uncertainty
  wrapper for any base. (`docs/results/qwen-14b/diverse-train-balanced.md`)

## Structure: one library + four packages

The **mechanism** lives in its own library; the framework packages hold the *opinions* built on it:

| Component | What | Import |
|-----------|------|--------|
| [**meta-attention**](https://github.com/artem-x-meta/meta-attention) | the MECHANISM: two-pass pipeline, activation reader, encoders, gated cross-attention, buffer, checkpoint format, `Injector` protocol, C++/ggml + llama.cpp leg | `from meta_attention import ...` (separate repo, Apache-2.0) |
| **daimon-voices** | the VOICES + the `Voice` contract + `watchdog`: Doubter, GoalAnchor, Chronographer, ChronoAnchor | `from daimon_voices import ...` |
| **daimon-agent** | agentic runtime + chat + OpenAI-compatible serving: MetaAgent, Session, native tool format, backends | `from daimon_agent import ...` |
| **daimon-loom** | training + eval + factories: Trainer, collector, losses; BaselineComparison (QA selective) + AgentComparison (agentic); `metaloom` CLI | `from daimon_loom import ...` |
| **daimon-deploy** | GGUF sidecar export for llama.cpp — `train in PyTorch → deploy in llama.cpp` | `from daimon_deploy import ...` / `metadeploy` |

**Dependency graph:** `meta-attention` depends on nothing. `daimon-voices` → the library.
`daimon-agent` → the library. `daimon-loom` → library + voices + agent (the agentic eval runs the
real runtime). `daimon-deploy` → the library (the checkpoint contract). `from daimon import ...`
is a thin umbrella that re-exports the mechanism under the `Daimon*` names
(`DaimonPipeline is MetaAttentionPipeline`) plus the voices.

## What it gives you

A validated voice:

- **Doubter** — the model learns calibrated refusal ("I'm not sure") instead of hallucinating, plus
  optional self-correction. Phase 2 Selective MMLU on Llama-8B: selective accuracy 89.1%,
  total_recovery 90.7%.

- **Watchdog** — the same signal, read-only: a `ConfidenceProbe` gates an external action without
  touching generation.

- **ChronoAnchor** — goal-conditioned episodic memory for AGENT SESSIONS: the session's goal
  (a policy: *"stay under $60"*, *"read-only, never modify"*) is injected as a LATENT condition
  into the memory organ, so every memory token carries a piece of the goal. The goal never enters
  the prompt — and the agent still obeys it across a long tool-using session. On Llama-3.2-1B:
  **1.000 vs the 0.500 structural ceiling** of the same weights without the goal, and **0.972 under
  unseen adversarial lures** that collapse the bare model to 0.139. On gemma-4-12b it beats even a
  TEXT goal where the model's RLHF prior overrides an instruction (`stdlib-only`: text 0/6 → latent
  6/6). Wiring: `anchor.set_goal(...)` once, `MetaAgent(..., step_hooks=[anchor])` — see
  `examples/chrono_anchor_agent.py`.

And the tooling around it:

- **`gain` knob** — one runtime dial on the injection (`set_gain` / `META_GAIN`); mix N voices on the
  shared residual, each with its own gain. AGC damps the self-amplifying loop over long generation.
- **`metaloom build-universal`** — the factory: one command builds a general uncertainty wrapper for any
  base from a balanced diverse mix (collect → train → per-axis eval → GGUF).
- Trainer on your data (two-pass + the 5-group AdamW from the Phase 2 record)
- BaselineComparison — the main selective-eval tool: (base) vs (base+voices) on a QABenchmark
  (MMLU/GSM8K/HumanEval/custom JSONL) with statistical significance (McNemar + paired t-test, no scipy)
- AgentComparison — agentic eval (multi-step tool use) via the daimon-agent runtime: an honest base vs Doubter+AGC
  loop on the same tools (native tool format, not hand-rolled ReAct)

## Install

```bash
# the mechanism first — everything stands on it:
pip install git+https://github.com/artem-x-meta/meta-attention

git clone https://codeberg.org/imperius/daimon
cd daimon
pip install -e daimon-voices -e daimon-agent -e daimon-loom
# optional llama.cpp deploy (GGUF sidecar export):
pip install -e daimon-deploy
# optional umbrella for `from daimon import ...`:
pip install -e .
```

Core deps: `torch>=2.1`, `numpy>=1.24`.
To load HF models: `pip install transformers accelerate`. For base quantization
(`--quantization nf4`/`int8`) — also `pip install bitsandbytes`.

> **Production inference needs only `meta-attention` + `daimon-voices`** (the library depends on
> nothing; voices depend on the library). That is about the package DEPENDENCY graph, NOT "nothing
> else is needed": running on an HF model still requires `transformers` (+ `bitsandbytes` for nf4).

## Quickstart

### Inference with pre-trained checkpoints

```python
from daimon import DaimonConfig, DaimonPipeline, Doubter
# (umbrella names; equivalently: from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
#  and from daimon_voices import Doubter)

# IMPORTANT: target/cross_attn_layers must match those the checkpoint was trained with
# (they live in run.json next to doubter_checkpoint.pt). Otherwise — "Missing key(s) layer_projectors.*".
cfg = DaimonConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    target_layers="late", cross_attn_layers="late",   # as in training (or an explicit list from run.json)
)
pipeline = DaimonPipeline.from_pretrained(cfg)

pipeline.attach(Doubter.from_checkpoint("doubter.pt"))

text = pipeline.generate("Describe a sunset", max_new_tokens=100)
# → a calibrated answer: answers confidently or refuses honestly
```

> The CLI automates this: `daimon-agent run --run-dir <run> "question"` reads `run.json` itself
> (model + layers + checkpoint) — nothing to specify by hand.

### Training a Doubter on your data

```python
from daimon import DaimonConfig, DaimonPipeline, Doubter, DoubterConfig
from daimon_loom import ActivationDatasetCollector, Trainer, TrainerConfig

cfg = DaimonConfig(model_name="meta-llama/Llama-3.2-1B-Instruct")
pipeline = DaimonPipeline.from_pretrained(cfg)

# 1. Once: collect activations + baseline answers (cache to disk)
collector = ActivationDatasetCollector(pipeline)
samples = collector.collect(my_questions, my_ground_truths)
ActivationDatasetCollector.save(samples, "my_dataset.pt")

# 2. Attach the Doubter and train
doubter = Doubter(DoubterConfig(encoder_type="selective"))
pipeline.attach(doubter)

trainer = Trainer(doubter, pipeline, TrainerConfig(epochs=10))
history = trainer.train(
    train_samples=samples[:1000],
    val_samples=samples[1000:1200],
    checkpoint_dir="./checkpoints",
)
```

### Evaluating the voice's value-add

```python
from daimon_loom import QABenchmark, BaselineComparison

bench = QABenchmark.from_jsonl("mmlu_hard.jsonl", name="mmlu_hard", scoring="multiple_choice")
report = BaselineComparison(pipeline, bench).run(num_tasks=500)

print(report.summary())
# ──────────────────────────────────────────────────────────────────────
# Metric                       Base    Modified         Δ          p
# ──────────────────────────────────────────────────────────────────────
# Overall accuracy           0.4120      0.4870   +0.0750     <0.001
# Selective accuracy         0.4120      0.8910   +0.4790     <0.001
# Refusal rate               0.0000      0.6330   +0.6330         —
# Refusal precision (vs oracle)   —      0.5040        —          —
# ──────────────────────────────────────────────────────────────────────
report.save_json("results.json")
```

### Deploy to llama.cpp (CPU / Metal / edge)

The wrapper is separated from the frozen base → its forward ports to ggml. `train in PyTorch →
deploy in llama.cpp`: calibrated refusal on a quantized GGUF base without CUDA/PyTorch.

```bash
metadeploy export --run-dir runs/my_doubter          # wrapper → doubter_sidecar.gguf
# + the llama.cpp patch (meta-attention library, cpp/llama_patch/) → llama-meta-generate (two-pass injection)
```

Verified end-to-end (Qwen2.5-0.5B selective): base answers, Doubter → calibrated refusal — verbatim as
PyTorch. The encoder (selective+multi_token) and CA match PyTorch (diff ~1e-7). Calibration survives
quantization down to Q4_K_M (refusal_prec ~0.97). Details — [`daimon-deploy/`](daimon-deploy/README.md).

## Architecture

The concept is described in the project write-up (meta-attention-is-all-you-need).
In short — 5 components:

```text
                 Pass 1 (read)                Pass 2 (write)
                 ─────────────                ──────────────
prompt → frozen base.forward         prompt → frozen base.generate
            ↓                                       ↑
   ActivationCollector hooks                  CA hooks inject
   capture hidden states                      cognitive tokens
   from target_layers                         via tanh-gates
            ↓                                       ↑
        Encoder                          ReflexionBuffer
   (Selective / Transformer)            ←─ read ───────┘
            ↓                                       ↑
   cognitive tokens [B, N, h] ───────  fill ────────┘
```

Five components:

1. **`ActivationCollector`** — forward hooks capture hidden states from the base's target layers.
   Pool over the last token (the decision point).
2. **`Encoder`** (Selective / Transformer) — a trainable network turning activations into
   cognitive tokens `[B, num_layers, hidden_dim]`.
3. **`ReflexionBuffer`** — storage for cognitive tokens between Pass 1 and Pass 2.
4. **`BottleneckCrossAttention`** — the meta-attention head, injects cognitive tokens back into the
   LLM at every target layer via `residual + tanh(gate) · CA(h, cog_tokens)`.
   A bottleneck projector (4096→256→4096) filters noise, half the cost of full-rank.
5. **Two-pass forward** — Pass 1 reads + encodes, Pass 2 generates with active injection.
   ActivationCollector is frozen during Pass 2 so it doesn't overwrite the buffer.

## The voice

### Doubter — calibrated answerer

**What it does:** the model learns to say "I'm not sure" when it genuinely is, instead of
hallucinating. On hard tasks, optionally with self-correction ("Wait, actually...").

**When it injects:** on every forward step (always-on). Trained via LM cross-entropy on targets
`confirm / correct / refuse`.

**API:**

```python
from daimon_voices import Doubter, DoubterConfig

doubter = Doubter(DoubterConfig(
    encoder_type="selective",       # or "transformer" (Phase 8, unlocked self-correction)
    encoder_bottleneck=256,
    ca_bottleneck_dim=256,
    ca_num_heads=4,
    num_cognitive_tokens=32,
    correction_ratio=0.5,           # 50/50 split correct vs refuse for a wrong pass1
))
pipeline.attach(doubter)
```

**Validated:** Phase 2 Selective Llama-8B MMLU full — sel_acc 89.1%.

> **Honest-metrics caveat.** The historical headline "refusal precision 99.84%" was a **metric
> artifact** (it checked "refusal text ≠ answer text" ≈ always True, not "would the model have erred").
> Honest refusal precision is ~0.50–0.55 and does NOT grow with scale. The real result is **selective
> accuracy**. Over-refusal is a known cost at every scale, not a failure — the usefulness criterion is
> whether selective accuracy moves.

## Layer selection: "all" / "late" presets

`target_layers` (where to read activations) and `cross_attn_layers` (where to inject) accept an
explicit list of indices or a preset:

```python
cfg = DaimonConfig(
    model_name="...",
    target_layers="all",        # read from all layers (hooks are free)
    cross_attn_layers="late",   # inject only into the top third
)
```

- `"all"` — all base layers (default)
- `"late"` — the top third (for 26 layers → L17-25, for 32 → L21-31)

Why late: (1) on the All-CA injection map, late layers are the most receptive (L27-31 on 8B), early
ones almost closed; (2) autograd builds the graph only from the lowest CA layer upward — late-CA is
proportionally cheaper to train.

**Validated** (Gemma-2-2B Phase 1R, TriviaQA): late vs canonical [6,12,18,24] —
selective accuracy **72.1% vs 68.9%**, refusal precision 97.1% vs 97.5%, training epoch **−15%**.
Trade-off: higher refusal rate (69% vs 56%) — a preset for "answer only when confident" scenarios.

## Dynamic cognitive-token refresh (per-token)

In static two-pass generation, cognitive tokens are computed ONCE from the prompt and frozen for the
whole answer. For long reasoning (math, agentic chains) that's weak: the model's uncertainty evolves
as it goes, and a frozen signal misses it.

`dynamic_refresh=True` enables per-token adaptive refresh via `IntrospectionCache`: generation runs in
chunks, between chunks the current context's activations are collected, and cosine similarity decides —
recompute cognitive tokens (state diverged) or reuse the cache (stable).

```python
text = pipeline.generate(
    "Solve: ...", max_new_tokens=256,
    dynamic_refresh=True,
    refresh_threshold=0.5,      # below → refresh (0.5 optimal for GSM8K)
    refresh_min_interval=3,     # minimum tokens between refreshes
    refresh_max_interval=20,    # force a refresh after N tokens
)
print(pipeline.last_dynamic_stats)  # total_refreshes, cache_hit_rate, avg_similarity
```

**Validated** (research, Llama-8B GSM8K, March 2026): dynamic refresh **+7.2pp** selective accuracy vs
static; threshold 0.5 optimal (above the dataset's mean similarity ~0.4). Adaptive chunk size:
shrinks after a refresh, grows when the state is stable.

## Memory: base compression and checkpointing

The base is frozen — it inherits the whole PEFT arsenal. Backward through quantized weights works
(the QLoRA approach): the wrapper lives outside them.

```python
cfg = DaimonConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    quantization="nf4",            # None | "int8" (~2×) | "nf4" (~4×, recommended) | "fp4"
    double_quant=True,             # another ~0.4 bytes/param for 4-bit
    gradient_checkpointing=True,   # activation memory ↓ a lot, time ↑ ~30%
)
```

Memory ballpark (base weights, no activations):

| Model | bf16 | int8 | nf4 |
| --- | --- | --- | --- |
| Gemma-2-2B | 5.2GB | ~2.8GB | ~1.6GB |
| Llama-8B | 16GB | ~8.5GB | ~4.5GB |
| 70B | 140GB | ~72GB | ~38GB |

Requires `pip install bitsandbytes`. Only Linear layers are quantized; embeddings, layernorms and the
whole wrapper (encoder + CA) stay full-precision.

dtype note: on GPUs without hardware bf16 (T4, P100) use `dtype="float16"` — the Trainer auto-enables a
GradScaler against gradient underflow. bf16 on such cards is emulated via fp32 and runs much slower.

## Encoder architectures

Two built-in:

| Encoder | Architecture | Where it wins |
|---|---|---|
| `SelectiveEncoder` | per-layer projectors (LN+Linear+GELU) + per-layer encoder gates + shared output proj. **No self-attention** between layers | Phase 1-5 canon, the calibration record |
| `TransformerEncoder` | per-layer projectors + N×TransformerBlock (multi-head self-attn + FFN) + output proj. Optional `prenorm_projector=True` (Phase 8 convention) | Phase 8 — unlocked self-correction (correction_acc 50% on 1B) |

You can bring your own — an encoder is any `nn.Module` satisfying the library's `Encoder`
protocol; pass the instance to your own `Injector`/`Voice` (the built-in voices construct
theirs from `encoder_type`):

```python
from meta_attention import Encoder
import torch.nn as nn

class MyEncoder(nn.Module, Encoder):
    def __init__(self, hidden_dim, num_layers, ...):
        ...

    def forward(self, activation_list):
        # activation_list: list of [B, hidden_dim], len=num_layers
        # return cognitive_tokens [B, num_layers, hidden_dim]
        ...
```

## Training

The Trainer carries the exact defaults of the Phase 2 Selective record:

- 5-group AdamW: `encoder weights` / `CA weights` / `encoder gates ×5 LR` /
  `CA gates ×5 LR` / `token preferences ×5 LR`
- Cosine schedule with 5% warmup
- Effective batch = 32 (`batch_size=2 × grad_accumulation=16`)
- Pretrain projectors via `_ConfidenceHead` (BCE on P(correct)) — critical for a ≥16-layer
  SelectiveEncoder; without it a 32-layer net doesn't converge
- Early stopping on val_loss

```python
trainer = Trainer(doubter, pipeline, TrainerConfig(
    epochs=10,
    learning_rate=2e-4,
    gate_lr_multiplier=5.0,
    correction_ratio=0.5,
    pretrain_projectors=True,
))
history = trainer.train(samples, val_samples=val_samples, checkpoint_dir="./ckpt")
```

## Benchmarks

Two built-in types:

### `QABenchmark` — single-shot

```python
from daimon_loom import QABenchmark, BenchmarkTask

tasks = [BenchmarkTask(task_id=f"q{i}", prompt=q, expected_answer=a)
         for i, (q, a) in enumerate(zip(questions, answers))]
bench = QABenchmark(name="my_bench", tasks=tasks, scoring="multiple_choice")
# scoring: "exact_match" / "contains" / "multiple_choice" / "custom"

# Or from JSONL:
bench = QABenchmark.from_jsonl("mmlu_hard.jsonl", name="mmlu_hard", scoring="multiple_choice")
```

### `AgentComparison` / `AgentTask` — agentic eval (multi-step tool use)

An honest base-vs-Doubter loop through the **daimon-agent** runtime (native tool format, not hand-rolled ReAct):

```python
from daimon_agent import Tool, ToolRegistry
from daimon_loom import AgentComparison, AgentTask

# Tool(name, description, fn, arg="input") — fn(**{arg}) -> str
search = Tool("search", "Search the web", lambda input: f"Found: {input}")

task = AgentTask(
    id="agentic_1",
    prompt="Find the capital of France",
    tools=ToolRegistry([search]),
    grade=lambda final: "Paris" in final,     # objective grade of the final answer
)

# pipeline + (optional) doubter; model_name → choice of turn stop-strings
report = AgentComparison(pipeline, doubter=doubter, model_name="Qwen/Qwen3.5-4B").run([task])
print(report)   # pass-rate base vs Doubter, rescued/broke, lookup spend
```

### `BaselineComparison` — the main tool

Runs the benchmark twice (base with voices off + with them on), computes deltas + statistical
significance:

```python
report = BaselineComparison(pipeline, bench).run(num_tasks=500)
print(report.summary())               # formatted table
report.per_task_deltas()              # per-task: base vs modified
report.refusal_breakdown()            # refusal confusion matrix
report.save_json("results.json")
```

## API reference

| Class | Where | What it does |
|---|---|---|
| `MetaAttentionConfig` (`DaimonConfig`) | `meta_attention.config` | Top-level config + `resolve_defaults` auto-detect |
| `MetaAttentionPipeline` (`DaimonPipeline`) | `meta_attention.pipeline` | Wrap an HF LM + two-pass forward |
| `SelectiveEncoder` | `meta_attention.encoders.selective` | Feedforward encoder (Phase 1-5 canon) |
| `TransformerEncoder` | `meta_attention.encoders.transformer` | Mini-transformer over cog tokens (Phase 8) |
| `BottleneckCrossAttention` | `meta_attention.injection` | Meta-attention head |
| `ReflexionBuffer` | `meta_attention.buffer` | Cog-token storage |
| `ActivationCollector` | `meta_attention.reader` | Forward hooks for activations |
| `Injector` | `meta_attention.injector` | The protocol: what the mechanism requires of an attachment |
| `Voice` | `daimon_voices.voice` | Injector + lifecycle + checkpoint contract + gain fader |
| `Doubter` / `DoubterConfig` | `daimon_voices.doubter` | Calibrated answerer |
| `GoalAnchor` / `GoalAnchorConfig` | `daimon_voices.goal_anchor` | Persistent goal anchor, trigger-gated |
| `Chronographer` / `ChronoAnchor` | `daimon_voices.chrono*` | Episodic memory / goal-conditioned memory |
| `ConfidenceProbe` | `daimon_voices.watchdog` | Read-only probe that GATES (not a voice) |
| `Trainer` / `TrainerConfig` | `daimon_loom.training.trainer` | Two-pass + 5-group AdamW |
| `ChronoAnchorTrainer` | `daimon_loom.training.chrono_anchor_trainer` | The four-ingredient tandem recipe |
| `ActivationDatasetCollector` | `daimon_loom.training.collector` | Activation collection + cache |
| `QABenchmark` / `BenchmarkTask` | `daimon_loom.evaluation.benchmark` | Single-shot QA benchmark (selective eval) |
| `AgentComparison` / `AgentTask` | `daimon_loom.evaluation.agentic` | Agentic eval (multi-step tool use) via daimon-agent |
| `BaselineComparison` / `ComparisonReport` | `daimon_loom.evaluation.comparison` | base vs modified with stat tests |
| `EvalHarness` | `daimon_loom.evaluation.harness` | sel_acc / refusal_prec / total_recovery |
| `OpenRouterJudge` | `daimon_loom.evaluation.llm_judge` | LLM-as-a-judge via OpenRouter (Nemotron 120B free) |
| `MetaAgent` / `Session` | `daimon_agent` | Agentic runtime (tools, policy, step hooks) |
| `serve` | `daimon_agent.serve` | OpenAI-compatible server + latent side-channels |

## Project status

Current state (v0.4.0):

- Tests green (framework 239 + agent 27 + library 8+, CPU-only via a FakeLM — no GPU/network)
- Modular public surface: `meta_attention` (the mechanism) / `daimon_voices` / `daimon_loom` /
  `daimon_agent` / `daimon_deploy`; the `daimon` umbrella re-exports the common names
- Voices shipped: **Doubter** (validated, checkpoints on the HF Hub), **GoalAnchor** (early-stage,
  measured, on the Hub), **Chronographer** and **ChronoAnchor** (mechanism validated on Llama-1B,
  scaling in progress)
- **Watchdog** read-side probe (`daimon_voices.watchdog`)
- Factories: `metaloom build-universal` (uncertainty wrapper) and `metaloom build-anchor` (goal anchor)

Absent (by design, deferred):

- Reassembler (the third organ: strategy switch without losing the goal) — text-phase prototype
  validated in research, no latent version yet
- Judge / research probes beyond the Watchdog

## Further reading

- Browsable docs: [`docs-site/index.html`](docs-site/index.html) — open in any browser (EN/RU)
- The mechanism: [`meta-attention`](https://github.com/artem-x-meta/meta-attention) — the library
  README, and `cpp/` for the llama.cpp leg
- Per-package details: the `README.md` inside each of `daimon-voices/` · `daimon-loom/` ·
  `daimon-agent/` · `daimon-deploy/`

## License

MIT — see [`LICENSE`](LICENSE). The `meta-attention` library is licensed separately (Apache-2.0).
