# daimon-voices

The **Daimon leg** of the Daimon framework: injection voices — the frozen
model's inner advisory voices (Socratic daimonion: counsels, doesn't rule).

- **Doubter** — the voice of doubt: calibrated uncertainty (answer / refuse / look up / clarify).
- **GoalAnchor** — the voice of the goal: persistent latent anchor against goal drift
  (dev-only until fully validated).

The mechanism the voices speak through (frozen base, hooks, cognitive-token encoders,
gated cross-attention, the `Voice` contract) lives in the `meta-attention` library. Voices sum on the
residual stream, each with its own runtime `gain` fader.

```python
from daimon import DaimonConfig, DaimonPipeline
from daimon_voices import Doubter

pipe = DaimonPipeline.from_pretrained(DaimonConfig(model_name="Qwen/Qwen2.5-14B-Instruct"))
pipe.attach(Doubter.from_checkpoint("doubter.pt"))
pipe.set_gain(1.0)
```

Install: `pip install meta-attention && pip install -e .` (editable, from this folder).
