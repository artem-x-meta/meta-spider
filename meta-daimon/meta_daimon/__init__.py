"""META-DAIMON — the injection modifiers: the frozen model's inner advisory voices.

Named after the Socratic daimonion — the inner voice that does not rule, only counsels
(and mostly warns). Every modifier here is one such voice: it reads the base's own
activations, encodes them into cognitive tokens, and whispers back through gated
cross-attention into the residual stream. Voices SUM on the residual, each with its own
runtime `gain` fader (the mixing-console property).

The MECHANISM the voices speak through (pipeline, hooks, encoders, gated CA, the
`Modifier` contract) lives in `meta-core`; this package holds only the voices themselves.

Current voices:
- **Doubter** — the voice of doubt (calibrated uncertainty): reactive, rebuilt from each
  prompt's own activations; answer / refuse / look up / clarify.

Voices in validation (not yet published): GoalAnchor (goal-drift protection),
Reassembler (structure), Chronographer (time). The read-side counterpart (probes that only
LISTEN: `meta_core.watchdog`) is not a Meta-Daimon — it gates the voices instead of speaking.
"""

from meta_daimon.config import DoubterConfig
from meta_daimon.doubter import Doubter

__all__ = ["Doubter", "DoubterConfig"]
