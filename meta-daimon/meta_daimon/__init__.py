"""META-DAIMON — the injection modifiers: the frozen model's inner advisory voices.

Named after the Socratic daimonion — the inner voice that does not rule, only counsels
(and mostly warns). Every modifier here is one such voice: it reads the base's own
activations, encodes them (or an external anchor text) into cognitive tokens, and
whispers back through gated cross-attention into the residual stream. Voices SUM on
the residual, each with its own runtime `gain` fader (the mixing-console property).

The MECHANISM the voices speak through (pipeline, hooks, encoders, gated CA, the
`Modifier` contract) lives in `meta-core`; this package holds only the voices themselves.

Current voices:
- **Doubter** — the voice of doubt (calibrated uncertainty): reactive, rebuilt from each
  prompt's own activations; answer / refuse / look up / clarify.
- **GoalAnchor** — the voice of the goal (drift protection): PERSISTENT anchor encoded once
  from the goal/spec text, trigger-gated re-injection across generations. Early-stage but
  published (Qwen2.5-14B anchor on HF): diverse training gives constraint defense +19pp
  (in-domain) and transfer without quality loss on unseen constraint families.

Planned voices (ported/validated separately): Reassembler (structure), Chronographer (time).
The read-side counterpart (probes that only LISTEN: `meta_core.watchdog`) is not a Meta-Daimon —
it gates the voices instead of speaking.
"""

from meta_daimon.config import DoubterConfig, GoalAnchorConfig
from meta_daimon.doubter import Doubter
from meta_daimon.goal_anchor import BinaryTrigger, GoalAnchor, LearnableTrigger

__all__ = [
    "Doubter", "DoubterConfig",
    "GoalAnchor", "GoalAnchorConfig", "BinaryTrigger", "LearnableTrigger",
]
