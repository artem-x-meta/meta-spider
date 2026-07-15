"""Chronographer submodules — verbatim port from archive/src/phase_chronograph_llama1b.

State-dict names are unchanged, so the legacy `content_pipeline.pt` checkpoint loads as-is.
Research variants (gate_reader, entropy gate, signal capture) stayed in the archive.
"""
from daimon_voices._chrono.emotional_encoder import EmotionalEncoder
from daimon_voices._chrono.episode_compressor import EpisodeCompressor
from daimon_voices._chrono.memory_cross_attention import MemoryBottleneckCrossAttention
from daimon_voices._chrono.persistent_memory_bank import PersistentMemoryBank

__all__ = ["EmotionalEncoder", "EpisodeCompressor", "MemoryBottleneckCrossAttention",
           "PersistentMemoryBank"]
