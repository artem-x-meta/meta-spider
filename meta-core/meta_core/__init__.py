"""Meta-Core — the Meta-Spider meta-attention MECHANISM.

Frozen base + trainable wrapper primitives: two-pass pipeline, activation hooks,
cognitive-token encoders, gated bottleneck cross-attention, buffer, the `Modifier`
contract, dynamic refresh, the read-side Watchdog probe, checkpoint format contract.
NO training, NO benchmarks — and, since the Meta-Daimon split, NO concrete voices:
the actual injection modifiers (Doubter, GoalAnchor, …) live in the **meta-daimon**
package built on top of this mechanism.

Components under the **Meta-Spider** umbrella: **Meta-Core** (mechanism, depends on
nothing), **Meta-Daimon** (the voices → core), **Meta-Agent** (agents+chat → core),
**Meta-Loom** (training+benchmarks → core, daimon, agent), **Meta-Deploy** (GGUF → core).

Back-compat: `from meta_core import Doubter / DoubterConfig` still works
when `meta-daimon` is installed (lazy PEP-562 forwarding below), but new code should
import voices from `meta_daimon` directly.
"""
from meta_core.config import MetaSpiderConfig
from meta_core.pipeline import MetaSpiderPipeline
from meta_core.hooks import ActivationCollector
from meta_core.cross_attention import BottleneckCrossAttention
from meta_core.buffer import ReflexionBuffer
from meta_core.registry import register_encoder, register_modifier
from meta_core.timing import StageTimer
from meta_core.dynamic import IntrospectionCache
from meta_core.modifiers.base import Modifier
from meta_core.encoders.base import Encoder
from meta_core.encoders.selective import SelectiveEncoder
from meta_core.encoders.transformer import TransformerEncoder
from meta_core.encoders.multi_token import MultiTokenEncoder
from meta_core.watchdog import Watchdog, ConfidenceProbe

# Voices moved to meta-daimon; forwarded lazily for back-compat (no hard/circular dep).
_DAIMON_NAMES = ("Doubter", "DoubterConfig")

__all__ = [
    "MetaSpiderConfig",
    "MetaSpiderPipeline",
    "ActivationCollector", "BottleneckCrossAttention", "ReflexionBuffer",
    "register_encoder", "register_modifier", "StageTimer", "IntrospectionCache",
    "Modifier",
    "Encoder", "SelectiveEncoder", "TransformerEncoder", "MultiTokenEncoder",
    "Watchdog", "ConfidenceProbe",
    *_DAIMON_NAMES,
]


def __getattr__(name: str):
    if name in _DAIMON_NAMES:
        try:
            import meta_daimon
        except ImportError as e:
            raise ImportError(
                f"meta_core.{name} moved to the meta-daimon package "
                f"(`pip install -e meta-daimon`); import it from `meta_daimon`.") from e
        return getattr(meta_daimon, name)
    raise AttributeError(f"module 'meta_core' has no attribute {name!r}")
