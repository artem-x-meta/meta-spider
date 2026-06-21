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
| `Doubter` / `DoubterConfig` | calibrated-refusal modifier; `Doubter.from_checkpoint(path)` |
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

Training/eval — package **meta-loom**; agent + chat — **meta-agent**; llama.cpp deploy — **meta-deploy**.
Details and architecture — in the [root README](../README.md).
