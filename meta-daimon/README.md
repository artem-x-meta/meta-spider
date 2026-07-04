# meta-daimon

The **Meta-Daimon leg** of the Meta-Spider framework: injection modifiers — the frozen
model's inner advisory voices (Socratic daimonion: counsels, doesn't rule).

- **Doubter** — the voice of doubt: calibrated uncertainty (answer / refuse / look up / clarify).
- **GoalAnchor** — the voice of the goal: persistent latent anchor against goal drift
  (dev-only until fully validated).

The mechanism the voices speak through (frozen base, hooks, cognitive-token encoders,
gated cross-attention, the `Modifier` contract) lives in `meta-core`. Voices sum on the
residual stream, each with its own runtime `gain` fader.

```python
from meta_core import MetaSpiderConfig, MetaSpiderPipeline
from meta_daimon import Doubter

pipe = MetaSpiderPipeline.from_pretrained(MetaSpiderConfig(model_name="Qwen/Qwen2.5-14B-Instruct"))
pipe.attach(Doubter.from_checkpoint("doubter.pt"))
pipe.set_gain(1.0)
```

Install: `pip install -e ../meta-core -e .` (editable, from this folder).
