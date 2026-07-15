"""Daimon — inner voices for a frozen LLM.

Named after the Socratic *daimonion*: the inner voice that counsels but does not rule. A voice
reads the model's own activations, compresses them into cognitive tokens, and whispers them back
into the residual stream. The base's weights never change. Voices SUM on the residual, each with
its own gain fader — a mixing console, not a monolith.

    from daimon import DaimonPipeline, DaimonConfig, Doubter

    pipe = DaimonPipeline.from_pretrained(DaimonConfig(model_name="…"))
    pipe.attach(Doubter.from_checkpoint("doubter.pt"))
    pipe.generate("…")

The MECHANISM the voices speak through — reading activations, encoding them, gated cross-attention
injection, the two-pass pipeline, the checkpoint format, and the C++/ggml implementation — lives in
a separate library: **`meta-attention`** (Apache-2.0). It knows nothing about doubt, goals or
memory. This framework is one set of answers to *what* the model should say to itself, and *when*.

The voices:

    Doubter        doubt        calibrated refusal: answer / refuse / look up / clarify
    GoalAnchor     the goal     a persistent anchor against drift over a long session
    Chronographer  the past     a rolling bank of episodes against context rot
    ChronoAnchor   both, fused  the goal conditions episode compression, so every memory token
                                carries a piece of it — the voice for agentic sessions
    Reassembler    "what if?"   planned: change the approach without losing the goal

Read-side probes (`daimon_voices.watchdog`) are NOT voices: they do not speak, they GATE — they
decide *when* a voice should be heard.

Packages:

    daimon-voices   the voices + the Voice contract + watchdog   → meta-attention
    daimon-loom     training, evaluation, the wrapper factory    → meta-attention, voices, agent
    daimon-agent    agentic runtime, native tool use, serving    → meta-attention
    daimon-deploy   GGUF sidecar export for the llama.cpp leg    → meta-attention
"""

__version__ = "0.4.0"

# The mechanism, from the library …
from meta_attention import BottleneckCrossAttention, Encoder, Injector, MetaAttentionConfig, MetaAttentionPipeline, MultiTokenEncoder, ReflexionBuffer, SelectiveEncoder, TransformerEncoder

# … re-exported under the framework's own names: a user of `daimon` should not have to know where
# the seam is until the day they want to build their own voice.
DaimonPipeline = MetaAttentionPipeline
DaimonConfig = MetaAttentionConfig

# The voices.
from daimon_voices import (BinaryTrigger, ChronoAnchor, ChronoAnchorConfig, Chronographer,
                           ChronographerConfig, Doubter, DoubterConfig, GoalAnchor,
                           GoalAnchorConfig, LearnableTrigger, Voice)

__all__ = [
    "__version__",
    # the mechanism (under our names, and its own)
    "DaimonPipeline", "DaimonConfig",
    "MetaAttentionPipeline", "MetaAttentionConfig",
    "BottleneckCrossAttention", "ReflexionBuffer", "Injector",
    "Encoder", "SelectiveEncoder", "MultiTokenEncoder", "TransformerEncoder",
    # the voices
    "Voice",
    "Doubter", "DoubterConfig",
    "GoalAnchor", "GoalAnchorConfig", "BinaryTrigger", "LearnableTrigger",
    "Chronographer", "ChronographerConfig",
    "ChronoAnchor", "ChronoAnchorConfig",
]
