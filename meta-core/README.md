# meta-core

Inference primitives of Meta-Spider: the two-pass mechanism (frozen base + thin trainable wrapper).
**Pure core — depends on nothing**; production inference installs only this package.

## Install

```bash
pip install -e meta-core
pip install transformers accelerate     # to load HF models
pip install bitsandbytes                  # optional — for --quantization nf4/int8
```

"Only meta-core" is about the package dependency graph; running inference on an HF model still
requires `transformers` (+ `bitsandbytes` for nf4), installed separately.

## What's inside

| Class | Role |
|---|---|
| `MetaSpiderConfig` | config: model_name, target_layers/cross_attn_layers, dtype, quantization, offload |
| `MetaSpiderPipeline` | wraps an HF causal LM; two-pass forward + `.generate()` |
| `Doubter` / `DoubterConfig` | calibrated-refusal modifier (CA-injection); `Doubter.from_checkpoint(path)` |
| `Watchdog` / `ConfidenceProbe` | **read-only** uncertainty sensor: encoder → cog token → logistic probe → P(uncertain). NO injection — gates an external action (lookup/refuse). `.fit()` / `.score()` / `.is_uncertain()` |
| `SelectiveEncoder` / `MultiTokenEncoder` | activation encoders → cognitive tokens |
| `BottleneckCrossAttention` | meta-attention head (injects cog tokens into the residual via tanh-gate) |
| `ActivationCollector` | forward hooks on target layers (family-agnostic: Llama/Qwen/Gemma/Phi/Granite) |
| `IntrospectionCache` | dynamic per-token refresh of cog tokens |

## Inference

```python
from meta_core import MetaSpiderConfig, MetaSpiderPipeline, Doubter

# IMPORTANT: target/cross_attn_layers MUST match those the checkpoint was trained with
# (they live in run.json); otherwise "Missing key(s) layer_projectors.*". 'late' = top third.
cfg = MetaSpiderConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct",
                       target_layers="late", cross_attn_layers="late")
pipe = MetaSpiderPipeline.from_pretrained(cfg)
pipe.attach(Doubter.from_checkpoint("doubter_checkpoint.pt"))
print(pipe.generate("What is the capital of France?", max_new_tokens=64))
```

### Watchdog (read-only sensor)

Reuses a trained encoder but **injects nothing** — reads the cognitive token at the decision point and
returns `P(uncertain)` to gate an external action (look up / refuse / escalate). No generation side-effects.

```python
from meta_core import Watchdog, Doubter
doubter = Doubter.from_checkpoint("doubter_checkpoint.pt")
# fit a probe on collected (activations → label) pairs, or load a saved one:
wd = Watchdog.fit(doubter.encoder, activation_lists, labels)   # 1 = uncertain/wrong
if wd.is_uncertain(activations, threshold=0.5):
    ...  # trigger a docs lookup / ask for clarification instead of answering blind
wd.save("watchdog.json"); Watchdog.load("watchdog.json", doubter.encoder)
```

Training/eval — package **meta-loom**; agent + chat — **meta-agent**; llama.cpp deploy — **meta-deploy**.
Details and architecture — in the [root README](../README.md).
