"""Daimon Loom — training + benchmark-evaluation pipeline for the voices.

Trainer (two-pass, 5-group AdamW, probe pretrain), ActivationDatasetCollector,
losses; evaluation (BaselineComparison/QABenchmark — selective QA; AgentComparison —
agentic, multi-step; EvalHarness, OpenRouterJudge). Stands on the **meta-attention**
library (the mechanism) and **daimon-voices**; agentic eval **delegates to
daimon-agent** (`AgentComparison` runs the real agent runtime).

Name: the mechanism spins the thread (activations) → the loom weaves the voice out of it.

Modules: `daimon_loom/{training,evaluation,data,cli}/*`. This file is the public API.
"""
from daimon_loom.training.trainer import Trainer, TrainerConfig
from daimon_loom.training.collector import ActivationDatasetCollector, DatasetSample
from daimon_loom.training.losses import (
    build_agentic_target,
    build_correction_target,
    build_target_by_action,
    make_labels_with_prompt_mask,
)
from daimon_loom.evaluation.harness import EvalHarness
from daimon_loom.evaluation.llm_judge import OpenRouterJudge
from daimon_loom.evaluation.benchmark import (
    Benchmark,
    BenchmarkRunner,
    BenchmarkTask,
    QABenchmark,
    TaskResult,
)
from daimon_loom.evaluation.comparison import BaselineComparison, ComparisonReport
from daimon_loom.evaluation.agentic import AgentComparison, AgentTask

__all__ = [
    "Trainer", "TrainerConfig",
    "ActivationDatasetCollector", "DatasetSample",
    "build_agentic_target", "build_correction_target", "build_target_by_action",
    "make_labels_with_prompt_mask",
    "EvalHarness", "OpenRouterJudge",
    "Benchmark", "QABenchmark", "BenchmarkRunner", "BenchmarkTask", "TaskResult",
    "BaselineComparison", "ComparisonReport",
    "AgentComparison", "AgentTask",
]
