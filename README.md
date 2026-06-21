# Meta-Spider

> Pluggable cognitive modifiers for any HuggingFace causal LM.
> Architectural introspection via meta-attention — no base fine-tuning.

📖 **Documentation (with diagrams, EN/RU):** [imperius.codeberg.page/meta-spider](https://imperius.codeberg.page/meta-spider/)

```text
Text reflection:    model → "think again" → model re-reads text
Meta-attention:     model → activations → encoder → cognitive tokens
                          → cross-attention with tanh-gates → model
```

The base model is **frozen**. Only a thin wrapper is trained (~2.3% of params on 8B).
On Llama-3.1-8B Phase 2 Selective MMLU: **selective accuracy 89.1%** (base 64.6%) at 37% coverage —
the model answers far more reliably on what it chooses to answer (see the honest-metrics caveat below).

## Structure: Meta-Core + Meta-Loom + Meta-Agent + Meta-Deploy

Four pip packages, each in its own folder under `meta-spider-framework/` (its own pyproject):

| Component | What | Import |
|-----------|------|--------|
| **Meta-Core** | inference primitives: pipeline (two-pass), hooks, cross-attention, encoders, modifiers, checkpoint contract | `from meta_core import ...` (pip: `meta-core`) |
| **Meta-Loom** | training + eval: Trainer, collector, losses; BaselineComparison (QA selective) + **AgentComparison** (agentic) | `from meta_loom import ...` (pip: `meta-loom`) |
| **Meta-Agent** | agentic runtime + chat: MetaAgent, Session, native tool format, backends | `from meta_agent import ...` (pip: `meta-agent`) |
| **Meta-Deploy** | llama.cpp deploy: export the wrapper to a GGUF sidecar + ggml/C++ forward (CPU/Metal/edge) — `train in PyTorch → deploy in llama.cpp` | `from meta_deploy import ...` / `metadeploy` (pip: `meta-deploy`) |

**Dependency graph:** `Meta-Core` is the pure core (depends on nothing), production inference installs
only it. `Meta-Agent` → Core. `Meta-Loom` → Core **+ Agent** (the `AgentComparison` agentic eval runs
the Meta-Agent runtime). `Meta-Deploy` → Core (reads the checkpoint contract; the export side is light —
numpy+gguf, the C++ ggml forward builds separately). `from meta_spider import ...` is a thin umbrella
**compat shim** that re-exports the public Core+Loom API (backward-compat top-level names).

## What it gives you

A validated modifier:

- **Doubter** — the model learns calibrated refusal ("I'm not sure") instead of hallucinating, plus
  optional self-correction. Phase 2 Selective MMLU on Llama-8B: selective accuracy 89.1%,
  total_recovery 90.7%.

And the tooling around it:

- Trainer on your data (two-pass + the 5-group AdamW from the Phase 2 record)
- BaselineComparison — the main selective-eval tool: (base) vs (base+modifiers) on a QABenchmark
  (MMLU/GSM8K/HumanEval/custom JSONL) with statistical significance (McNemar + paired t-test, no scipy)
- AgentComparison — agentic eval (multi-step tool use) via Meta-Agent: an honest base vs Doubter+AGC
  loop on the same tools (native tool format, not hand-rolled ReAct)

## Install

```bash
git clone <repo-url>            # <repo-url> — this repository's address
cd reflexion-core/meta-spider-framework
# editable packages (Loom pulls Core+Agent); production inference — only meta-core:
pip install -e meta-core -e meta-agent -e meta-loom
# optional llama.cpp deploy (GGUF sidecar export):
pip install -e meta-deploy
# optional umbrella shim for `from meta_spider import ...`:
pip install -e .
```

Core deps: `torch>=2.1`, `numpy>=1.24`.
To load HF models: `pip install transformers accelerate`. For base quantization
(`--quantization nf4`/`int8`) — also `pip install bitsandbytes`.

> **On "production installs only meta-core".** This is about the package DEPENDENCY graph (meta-core
> depends on nothing, installs alone), NOT "nothing else is needed": running inference on an HF model
> still requires `transformers` (+ `bitsandbytes` for nf4) — installed separately.

## Quickstart

### Inference with pre-trained checkpoints

```python
from meta_core import MetaSpiderConfig, MetaSpiderPipeline, Doubter

# IMPORTANT: target/cross_attn_layers must match those the checkpoint was trained with
# (they live in run.json next to doubter_checkpoint.pt). Otherwise — "Missing key(s) layer_projectors.*".
cfg = MetaSpiderConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    target_layers="late", cross_attn_layers="late",   # as in training (or an explicit list from run.json)
)
pipeline = MetaSpiderPipeline.from_pretrained(cfg)

pipeline.attach(Doubter.from_checkpoint("doubter.pt"))

text = pipeline.generate("Describe a sunset", max_new_tokens=100)
# → a calibrated answer: answers confidently or refuses honestly
```

> The CLI automates this: `meta-agent run --run-dir <run> "question"` reads `run.json` itself
> (model + layers + checkpoint) — nothing to specify by hand.

### Training a Doubter on your data

```python
from meta_core import MetaSpiderConfig, MetaSpiderPipeline, Doubter, DoubterConfig
from meta_loom import ActivationDatasetCollector, Trainer, TrainerConfig

cfg = MetaSpiderConfig(model_name="meta-llama/Llama-3.2-1B-Instruct")
pipeline = MetaSpiderPipeline.from_pretrained(cfg)

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

### Evaluating the modifier's value-add

```python
from meta_loom import QABenchmark, BaselineComparison

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
# + the llama.cpp patch (meta-deploy/llama_patch/) → llama-meta-generate (two-pass injection)
```

Verified end-to-end (Qwen2.5-0.5B selective): base answers, Doubter → calibrated refusal — verbatim as
PyTorch. The encoder (selective+multi_token) and CA match PyTorch (diff ~1e-7). Calibration survives
quantization down to Q4_K_M (refusal_prec ~0.97). Details — [`meta-deploy/`](meta-deploy/README.md).

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

## The modifier

### Doubter — calibrated answerer

**What it does:** the model learns to say "I'm not sure" when it genuinely is, instead of
hallucinating. On hard tasks, optionally with self-correction ("Wait, actually...").

**When it injects:** on every forward step (always-on). Trained via LM cross-entropy on targets
`confirm / correct / refuse`.

**API:**

```python
from meta_core import Doubter, DoubterConfig

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
cfg = MetaSpiderConfig(
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
cfg = MetaSpiderConfig(
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

You can register your own:

```python
from meta_core import register_encoder, Encoder
import torch.nn as nn

class MyEncoder(nn.Module, Encoder):
    def __init__(self, hidden_dim, num_layers, ...):
        ...

    def forward(self, activation_list):
        # activation_list: list of [B, hidden_dim], len=num_layers
        # return cognitive_tokens [B, num_layers, hidden_dim]
        ...

register_encoder("my_encoder", MyEncoder)
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
from meta_loom import QABenchmark, BenchmarkTask

tasks = [BenchmarkTask(task_id=f"q{i}", prompt=q, expected_answer=a)
         for i, (q, a) in enumerate(zip(questions, answers))]
bench = QABenchmark(name="my_bench", tasks=tasks, scoring="multiple_choice")
# scoring: "exact_match" / "contains" / "multiple_choice" / "custom"

# Or from JSONL:
bench = QABenchmark.from_jsonl("mmlu_hard.jsonl", name="mmlu_hard", scoring="multiple_choice")
```

### `AgentComparison` / `AgentTask` — agentic eval (multi-step tool use)

An honest base-vs-Doubter loop through the **Meta-Agent** runtime (native tool format, not hand-rolled ReAct):

```python
from meta_agent import Tool, ToolRegistry
from meta_loom import AgentComparison, AgentTask

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

Runs the benchmark twice (base with modifiers off + with them on), computes deltas + statistical
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
| `MetaSpiderConfig` | `config.py` | Top-level config + `resolve_defaults` auto-detect |
| `MetaSpiderPipeline` | `pipeline.py` | Wrap an HF LM + two-pass forward |
| `Doubter` / `DoubterConfig` | `modifiers/doubter.py` | Calibrated answerer |
| `SelectiveEncoder` | `encoders/selective.py` | Feedforward encoder (Phase 1-5 canon) |
| `TransformerEncoder` | `encoders/transformer.py` | Mini-transformer over cog tokens (Phase 8) |
| `BottleneckCrossAttention` | `cross_attention.py` | Meta-attention head |
| `ReflexionBuffer` | `buffer.py` | Cog-token storage |
| `ActivationCollector` | `hooks.py` | Forward hooks for activations |
| `Trainer` / `TrainerConfig` | `training/trainer.py` | Two-pass + 5-group AdamW |
| `ActivationDatasetCollector` | `training/collector.py` | Activation collection + cache |
| `QABenchmark` / `BenchmarkTask` | `evaluation/benchmark.py` | Single-shot QA benchmark (selective eval) |
| `AgentComparison` / `AgentTask` | `evaluation/agentic.py` | Agentic eval (multi-step tool use) via Meta-Agent |
| `BaselineComparison` / `ComparisonReport` | `evaluation/comparison.py` | base vs modified with stat tests |
| `EvalHarness` | `evaluation/harness.py` | sel_acc / refusal_prec / total_recovery |
| `OpenRouterJudge` | `evaluation/llm_judge.py` | LLM-as-a-judge via OpenRouter (Nemotron 120B free) |

## Project status

The framework is assembled from validated implementations of the original project:

- **Doubter** — a port of the original project's Phase 1-8 implementation.

Current state:

- 163 smoke tests green (`tests/`)
- API stabilized — modular public surface: import from `meta_core` / `meta_loom` / `meta_agent`
  directly (the `meta_spider` umbrella is kept only as an optional back-compat shim)
- Trained Doubter checkpoints (Qwen, Granite) are published on the Hugging Face Hub
  (each with a model card + GGUF sidecar)

Absent (by design, deferred):

- Watchdog (goal-keeping injector) — removed from the framework to focus on the Doubter; will return
  as a separate module after the core stabilizes
- Reassembler / Chronograph / Judge (research modifiers)

## Further reading

- Browsable docs: [`docs-site/index.html`](docs-site/index.html) — open in any browser (EN/RU)
- Per-package details: the `README.md` inside each of `meta-core/` · `meta-loom/` · `meta-agent/` · `meta-deploy/`

## License

MIT — see [`LICENSE`](LICENSE).
