"""DAIMON VOICES — the injection voices: the frozen model's inner advisory voices.

Named after the Socratic daimonion — the inner voice that does not rule, only counsels
(and mostly warns). Every voice here is one such voice: it reads the base's own
activations, encodes them (or an external anchor text) into cognitive tokens, and
whispers back through gated cross-attention into the residual stream. Voices SUM on
the residual, each with its own runtime `gain` fader (the mixing-console property).

The MECHANISM the voices speak through (pipeline, hooks, encoders, gated CA, the
`Injector` protocol) lives in the `meta-attention` library; this package holds the
voices themselves plus the `Voice` contract (Injector + lifecycle + checkpoint + gain).

Current voices:
- **Doubter** — the voice of doubt (calibrated uncertainty): reactive, rebuilt from each
  prompt's own activations; answer / refuse / look up / clarify.
- **GoalAnchor** — the voice of the goal (drift protection): PERSISTENT anchor encoded once
  from the goal/spec text, trigger-gated re-injection across generations. Published
  early-stage (measured: constraint defense +19pp in-domain on Qwen2.5-14B).
- **Chronographer** — the voice of time (episodic memory): a rolling bank of compressed
  episodes, injected on every forward; against context rot on long sessions.
- **ChronoAnchor** — the two above FUSED into ONE organ: the goal conditions episode
  compression, so every memory token carries a piece of the active goal. Holds the past and
  the future in one tract; the agentic lifecycle (goal per session, episode per step) makes it
  the voice for agent runtimes.

Planned voices (ported/validated separately): Reassembler (structure).
The read-side counterpart (probes that only LISTEN: `daimon_voices.watchdog`) is not a Daimon —
it gates the voices instead of speaking.
"""

from daimon_voices.voice import Voice
from daimon_voices.chrono_anchor import ChronoAnchor
from daimon_voices.chronographer import Chronographer
from daimon_voices.config import (ChronoAnchorConfig, ChronographerConfig, DoubterConfig,
                                GoalAnchorConfig)
from daimon_voices.doubter import Doubter
from daimon_voices.goal_anchor import BinaryTrigger, GoalAnchor, LearnableTrigger

__all__ = [
    "Voice",
    "ChronoAnchor", "ChronoAnchorConfig",
    "Chronographer", "ChronographerConfig",
    "Doubter", "DoubterConfig",
    "GoalAnchor", "GoalAnchorConfig", "BinaryTrigger", "LearnableTrigger",
]
