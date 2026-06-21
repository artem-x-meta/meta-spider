"""Training scaffolding: activation collection, two-pass training loop, LM loss.

The base model stays frozen — only the modifier wrapper is trained
(encoder + per-layer gates + cross-attention layers + token_preferences).
Backward flows through the frozen base as through a passive conductor.
"""

from meta_loom.training.collector import ActivationDatasetCollector, DatasetSample
from meta_loom.training.trainer import Trainer, TrainerConfig
from meta_loom.training.losses import (
    build_correction_target,
    build_target_by_action,
    make_labels_with_prompt_mask,
)

__all__ = [
    "ActivationDatasetCollector",
    "DatasetSample",
    "Trainer",
    "TrainerConfig",
    "build_correction_target",
    "build_target_by_action",
    "make_labels_with_prompt_mask",
]
