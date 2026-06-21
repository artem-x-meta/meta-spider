"""Meta-Loom — training + benchmark-evaluation pipeline for Meta-Spider.

Trainer (two-pass, 5-group AdamW, probe pretrain), ActivationDatasetCollector,
losses; evaluation (BaselineComparison/QABenchmark — selective QA; AgentComparison —
agentic, multi-step; EvalHarness, OpenRouterJudge). Depends on **Meta-Core**; agentic
eval **delegates to Meta-Agent** (`AgentComparison` runs the real Meta-Agent runtime).

Name: the spider (Meta-Core) spins the thread → the loom (Meta-Loom) weaves the
wrapper out of activations.

Modules: `meta_loom/{training,evaluation,data,cli}/*`. This file is the public API.
"""
from meta_loom.training.trainer import Trainer, TrainerConfig
from meta_loom.training.collector import ActivationDatasetCollector, DatasetSample
from meta_loom.training.losses import (
    build_correction_target,
    build_target_by_action,
    make_labels_with_prompt_mask,
)
from meta_loom.evaluation.harness import EvalHarness
from meta_loom.evaluation.llm_judge import OpenRouterJudge
from meta_loom.evaluation.benchmark import (
    Benchmark,
    BenchmarkRunner,
    BenchmarkTask,
    QABenchmark,
    TaskResult,
)
from meta_loom.evaluation.comparison import BaselineComparison, ComparisonReport
from meta_loom.evaluation.agentic import AgentComparison, AgentTask

__all__ = [
    "Trainer", "TrainerConfig",
    "ActivationDatasetCollector", "DatasetSample",
    "build_correction_target", "build_target_by_action", "make_labels_with_prompt_mask",
    "EvalHarness", "OpenRouterJudge",
    "Benchmark", "QABenchmark", "BenchmarkRunner", "BenchmarkTask", "TaskResult",
    "BaselineComparison", "ComparisonReport",
    "AgentComparison", "AgentTask",
]
