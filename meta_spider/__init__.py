"""Meta-Spider Framework — pluggable cognitive modifiers for any HuggingFace causal LM.

Concept (from the `meta-attention-is-all-you-need` paper): on every model forward pass, hooks
capture the hidden states, a trainable encoder compresses them into `cognitive tokens`, and those
tokens are injected back into the model through `BottleneckCrossAttention` layers with tanh gates.
The model base is frozen; only the thin wrapper is trained (~2.3% on 8B).

Production modifier:

- **Doubter** — a calibrated-introspection pipeline.
  Phase 2 Selective: sel_acc 89.1% (base 64.6%) on Llama-8B MMLU.

Quick start (target UX, landing in subsequent PRs):

    from meta_spider import MetaSpiderPipeline, MetaSpiderConfig, Doubter

    config = MetaSpiderConfig(model_name="meta-llama/Llama-3.2-1B-Instruct")
    pipeline = MetaSpiderPipeline.from_pretrained(config)
    pipeline.attach(Doubter.from_checkpoint("path/to/doubter.pt"))
    out = pipeline.generate("Hello, how are you?", max_tokens=100)

The current revision is a skeleton: package structure + signatures. Implementations are marked
with `NotImplementedError` and reference the source modules in `publish/github/src/`.
"""

__version__ = "0.0.1"

# Meta-Spider umbrella = Meta-Core (inference primitives) + Meta-Loom (training + benchmarks).
# Component boundaries live in the meta_core / meta_loom subpackages; here they are
# composed into a single public API (top-level names are the same — backward compatibility).
# (Meta-Agent is a separate package meta_agent: agents + chat, depends on Meta-Core, not from here.)
from meta_core import *   # noqa: F401,F403
from meta_loom import *   # noqa: F401,F403
from meta_core import __all__ as _core_all
from meta_loom import __all__ as _loom_all

__all__ = ["__version__", *_core_all, *_loom_all]
