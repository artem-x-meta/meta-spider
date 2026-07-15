"""Training scaffolding: activation collection, two-pass training loop, LM loss.

The base model stays frozen — only the voice wrapper is trained
(encoder + per-layer gates + cross-attention layers + token_preferences).
Backward flows through the frozen base as through a passive conductor.
"""

from daimon_loom.training.chrono_anchor_trainer import (ChronoAnchorSample,
                                                      ChronoAnchorTrainer)
from daimon_loom.training.collector import ActivationDatasetCollector, DatasetSample
from daimon_loom.training.trainer import Trainer, TrainerConfig
from daimon_loom.training.losses import (
    build_agentic_target,
    build_correction_target,
    build_target_by_action,
    make_labels_with_prompt_mask,
)

__all__ = [
    "ChronoAnchorTrainer", "ChronoAnchorSample",
    "ActivationDatasetCollector",
    "DatasetSample",
    "Trainer",
    "TrainerConfig",
    "build_agentic_target",
    "build_correction_target",
    "build_target_by_action",
    "make_labels_with_prompt_mask",
]
