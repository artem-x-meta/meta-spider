"""Meta-Core — Meta-Spider inference primitives.

Frozen base + trainable wrapper (two-pass pipeline, hooks, cross-attention, encoders,
Doubter/Watchdog modifiers, dynamic) + checkpoint format contract. NO training or
benchmarks.

One of the components under the **Meta-Spider** umbrella (alongside: **Meta-Loom** —
training+benchmarks, **Meta-Agent** — agents+chat). Meta-Core is the foundation: both
Meta-Loom and Meta-Agent depend on it. (Meta-Loom additionally depends on Meta-Agent —
for agentic eval, `AgentComparison`; Meta-Core itself depends on nothing — a pure core
for production inference.)

Core modules live in `meta_core/*` (config, pipeline, hooks, cross_attention, dynamic,
buffer, registry, timing, model_utils, encoders/, modifiers/). This is the core's public API.
"""
from meta_core.config import DoubterConfig, MetaSpiderConfig
from meta_core.pipeline import MetaSpiderPipeline
from meta_core.hooks import ActivationCollector
from meta_core.cross_attention import BottleneckCrossAttention
from meta_core.buffer import ReflexionBuffer
from meta_core.registry import register_encoder, register_modifier
from meta_core.timing import StageTimer
from meta_core.dynamic import IntrospectionCache
from meta_core.modifiers.base import Modifier
from meta_core.modifiers.doubter import Doubter
from meta_core.encoders.base import Encoder
from meta_core.encoders.selective import SelectiveEncoder
from meta_core.encoders.transformer import TransformerEncoder
from meta_core.encoders.multi_token import MultiTokenEncoder
from meta_core.watchdog import Watchdog, ConfidenceProbe

__all__ = [
    "MetaSpiderConfig", "DoubterConfig",
    "MetaSpiderPipeline",
    "ActivationCollector", "BottleneckCrossAttention", "ReflexionBuffer",
    "register_encoder", "register_modifier", "StageTimer", "IntrospectionCache",
    "Modifier", "Doubter",
    "Encoder", "SelectiveEncoder", "TransformerEncoder", "MultiTokenEncoder",
    "Watchdog", "ConfidenceProbe",
]
