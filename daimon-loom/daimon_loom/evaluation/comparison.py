"""BaselineComparison — the main tool for evaluating the value-add of voices.

Runs the benchmark twice:
  1. On the clean base model (all voices temporarily detached)
  2. On the pipeline with active voices

Computes per-task delta + statistical significance + a summary report.

Without it, a framework user can't tell whether their Doubter gives a real
improvement on their data or a regression.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from daimon_loom.evaluation.benchmark import BenchmarkRunner, TaskResult

if TYPE_CHECKING:
    from daimon_loom.evaluation.benchmark import Benchmark
    from meta_attention.pipeline import MetaAttentionPipeline

__all__ = ["BaselineComparison", "ComparisonReport"]


# ============================================================
# Statistical tests (no external deps — pure stdlib)
# ============================================================


def _mcnemar_test(b: int, c: int) -> dict[str, float]:
    """McNemar's test for paired binary results.

    Args:
        b: number of tasks where base=correct, modified=wrong (positives lost)
        c: number of tasks where base=wrong, modified=correct (positives gained)

    Returns:
        {"statistic", "p_value"}. p < 0.05 = a statistically significant difference.
    """
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    # Continuity-corrected McNemar (no scipy):
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    # P-value via the chi-square CDF with df=1
    # Approximated via erfc(sqrt(chi2/2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
    return {"statistic": round(chi2, 4), "p_value": round(p_value, 4)}


def _paired_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired t-test without scipy. Returns t-stat + p-value (two-tailed)."""
    n = len(a)
    if n < 2 or len(b) != n:
        return {"statistic": 0.0, "p_value": 1.0, "n": n, "mean_diff": 0.0}
    diffs = [a[i] - b[i] for i in range(n)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 0.0
    if se == 0:
        # Variance==0: identical pairs or a perfect constant diff
        if mean == 0.0:
            return {"statistic": 0.0, "p_value": 1.0, "n": n, "mean_diff": 0.0}
        # Perfect non-zero difference → strictly significant
        return {
            "statistic": float("inf"),
            "p_value": 0.0,
            "n": n,
            "mean_diff": round(mean, 4),
        }
    t = mean / se
    # Two-tailed normal approximation (valid for large n; conservative for small n)
    p_value = math.erfc(abs(t) / math.sqrt(2))
    return {
        "statistic": round(t, 4),
        "p_value": round(p_value, 4),
        "n": n,
        "mean_diff": round(mean, 4),
    }


# ============================================================
# ComparisonReport
# ============================================================


@dataclass
class ComparisonReport:
    """Summary report comparing base vs modified."""

    benchmark_name: str
    n_tasks: int
    base_results: list[TaskResult]
    modified_results: list[TaskResult]
    base_metrics: dict[str, Any] = field(default_factory=dict)
    modified_metrics: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    significance: dict[str, dict[str, float]] = field(default_factory=dict)
    voice_names: list[str] = field(default_factory=list)

    # --- Reporting helpers ---

    def summary(self) -> str:
        """Text summary: metrics table + significance."""
        lines = []
        lines.append(f"Benchmark: {self.benchmark_name} ({self.n_tasks} tasks)")
        if self.voice_names:
            lines.append(f"Voices: [{', '.join(self.voice_names)}]")
        lines.append("─" * 70)
        lines.append(f"{'Metric':<28}{'Base':>10}{'Modified':>12}{'Δ':>10}{'p':>10}")
        lines.append("─" * 70)

        metrics_to_show = [
            ("overall_accuracy", "Overall accuracy"),
            ("selective_accuracy", "Selective accuracy"),
            ("refusal_rate", "Refusal rate"),
            ("refusal_precision", "Refusal precision*"),
            ("over_refusal_rate", "Over-refusal rate"),
        ]
        for key, label in metrics_to_show:
            b = self.base_metrics.get(key)
            m = self.modified_metrics.get(key)
            if b is None and m is None:
                continue
            delta = self.deltas.get(key)
            p = self.significance.get("mcnemar" if key == "overall_accuracy" else "paired_t", {}).get("p_value")
            b_str = f"{b:.4f}" if b is not None else "—"
            m_str = f"{m:.4f}" if m is not None else "—"
            d_str = f"{delta:+.4f}" if delta is not None else "—"
            p_str = f"{p:.4f}" if p is not None else "—"
            lines.append(f"{label:<28}{b_str:>10}{m_str:>12}{d_str:>10}{p_str:>10}")
        lines.append("─" * 70)
        return "\n".join(lines)

    def per_task_deltas(self) -> list[dict[str, Any]]:
        """Per-task: {task_id, base_success, modified_success, delta}."""
        out = []
        for b, m in zip(self.base_results, self.modified_results):
            out.append({
                "task_id": b.task_id,
                "base_success": int(b.success),
                "modified_success": int(m.success),
                "delta": int(m.success) - int(b.success),
                "base_answer": b.final_answer,
                "modified_answer": m.final_answer,
                "base_refused": b.refused,
                "modified_refused": m.refused,
            })
        return out

    def refusal_breakdown(self) -> dict[str, Any]:
        """Detailed refusal breakdown:
          - confusion: refused_correctly / refused_wrongly / answered_correctly / hallucinated
          - per-voice impact on the refusal rate
        """
        if not self.modified_results:
            return {}

        # The 'base would get it right' oracle by task_id — for an honest refusal breakdown.
        base_success_by_id = {r.task_id: r.success for r in self.base_results}

        def base_ok(r) -> bool:
            return base_success_by_id.get(r.task_id, False)

        # Confusion matrix for modified (a refusal is judged against the BASE, not the refusal text):
        #   refused_correctly  — refusal on a question the base would get wrong (justified)
        #   refused_over       — refusal on a question the base knew (over-refusal, the cost of helpfulness)
        refused_correctly = sum(
            1 for r in self.modified_results if r.refused and not base_ok(r)
        )
        refused_over = sum(
            1 for r in self.modified_results if r.refused and base_ok(r)
        )
        answered_correctly = sum(
            1 for r in self.modified_results if not r.refused and r.success
        )
        hallucinated = sum(
            1 for r in self.modified_results if not r.refused and not r.success
        )
        n_refused = refused_correctly + refused_over

        return {
            "modified": {
                "refused_correctly": refused_correctly,
                "refused_over": refused_over,
                "answered_correctly": answered_correctly,
                "hallucinated": hallucinated,
                "honest_refusal_precision": (round(refused_correctly / n_refused, 4)
                                             if n_refused else None),
                "over_refusal_rate": (round(refused_over / n_refused, 4)
                                      if n_refused else None),
            },
            "delta_refusal_rate": self.deltas.get("refusal_rate", 0.0),
        }

    def save_json(self, path: str) -> None:
        """Save the full report to JSON."""
        payload = {
            "benchmark_name": self.benchmark_name,
            "n_tasks": self.n_tasks,
            "voice_names": self.voice_names,
            "base_metrics": self.base_metrics,
            "modified_metrics": self.modified_metrics,
            "deltas": self.deltas,
            "significance": self.significance,
            "per_task_deltas": self.per_task_deltas(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ============================================================
# BaselineComparison
# ============================================================


class BaselineComparison:
    """Run a benchmark on (base, modified) + compare.

    Args:
        pipeline: MetaAttentionPipeline with voices attached for the modified run.
            For the baseline run the voices are temporarily detached (via detach_all
            + re-attach). The base model is the same — no need to load it twice.
        benchmark: Benchmark.
        max_tokens: token limit per generation.
        significance_tests: which statistical tests to compute.
    """

    def __init__(
        self,
        pipeline: "MetaAttentionPipeline",
        benchmark: "Benchmark",
        max_tokens: int = 256,
        significance_tests: Optional[list[str]] = None,
        apply_chat_template: bool = True,
        generate_kwargs: Optional[dict] = None,
    ):
        self.pipeline = pipeline
        self.benchmark = benchmark
        self.max_tokens = max_tokens
        self.significance_tests = significance_tests or ["mcnemar", "paired_t"]
        self.apply_chat_template = apply_chat_template
        # Extra arguments for pipeline.generate (e.g. dynamic_refresh=True) for modified.
        # IMPORTANT: applied to both base and modified — but base without a voice
        # ignores dynamic (no encoder/buffer), so the comparison stays fair.
        self.generate_kwargs = generate_kwargs or {}

    def run(
        self,
        num_tasks: Optional[int] = None,
        run_baseline_first: bool = True,
        verbose: bool = True,
    ) -> ComparisonReport:
        """Run base + modified, assemble a ComparisonReport."""
        voice_snapshot = list(self.pipeline.injectors)
        voice_names = [m.name for m in voice_snapshot]

        # First note which voices are attached, then do the two runs
        if run_baseline_first:
            if verbose:
                print(f"=== BASELINE (no voices) ===")
            self._detach_voices(voice_snapshot)
            base_results = self._run(num_tasks, verbose)
            if verbose:
                print(f"=== MODIFIED (with {len(voice_snapshot)} voices) ===")
            self._reattach_voices(voice_snapshot)
            modified_results = self._run(num_tasks, verbose)
        else:
            if verbose:
                print(f"=== MODIFIED (with {len(voice_snapshot)} voices) ===")
            modified_results = self._run(num_tasks, verbose)
            if verbose:
                print(f"=== BASELINE (no voices) ===")
            self._detach_voices(voice_snapshot)
            base_results = self._run(num_tasks, verbose)
            self._reattach_voices(voice_snapshot)

        # Aggregate. Base correctness by task_id is the oracle for an HONEST refusal_precision
        # of the modified model (of its refusals, the fraction where the base would be wrong).
        base_success_by_id = {r.task_id: r.success for r in base_results}
        base_oracle = [base_success_by_id.get(r.task_id, False) for r in modified_results]
        base_metrics = self.benchmark.aggregate(base_results)
        modified_metrics = self.benchmark.aggregate(modified_results, pass1_correct=base_oracle)

        # Deltas
        deltas: dict[str, float] = {}
        for k in {*base_metrics.keys(), *modified_metrics.keys()}:
            b = base_metrics.get(k)
            m = modified_metrics.get(k)
            if isinstance(b, (int, float)) and isinstance(m, (int, float)):
                deltas[k] = round(m - b, 4)

        # Significance
        significance = self._compute_significance(base_results, modified_results)

        return ComparisonReport(
            benchmark_name=self.benchmark.name,
            n_tasks=len(base_results),
            base_results=base_results,
            modified_results=modified_results,
            base_metrics=base_metrics,
            modified_metrics=modified_metrics,
            deltas=deltas,
            significance=significance,
            voice_names=voice_names,
        )

    def _detach_voices(self, voices: list) -> None:
        for m in voices:
            if m in self.pipeline.injectors:
                m.on_detach()
                self.pipeline.injectors.remove(m)

    def _reattach_voices(self, voices: list) -> None:
        for m in voices:
            m.on_attach(self.pipeline)
            if m not in self.pipeline.injectors:
                self.pipeline.injectors.append(m)

    def _run(self, num_tasks: Optional[int], verbose: bool) -> list[TaskResult]:
        runner = BenchmarkRunner(
            self.pipeline, self.benchmark,
            max_tokens=self.max_tokens, verbose=verbose,
            apply_chat_template=self.apply_chat_template,
            generate_kwargs=self.generate_kwargs,
        )
        return runner.run(num_tasks=num_tasks)

    def _compute_significance(
        self,
        base_results: list[TaskResult],
        modified_results: list[TaskResult],
    ) -> dict[str, dict[str, float]]:
        """Paired statistical tests for the selected significance tests."""
        sig: dict[str, dict[str, float]] = {}

        if "mcnemar" in self.significance_tests:
            # b = base success → modified failure (positives lost)
            # c = base failure → modified success (positives gained)
            b = sum(
                1 for x, y in zip(base_results, modified_results)
                if x.success and not y.success
            )
            c = sum(
                1 for x, y in zip(base_results, modified_results)
                if not x.success and y.success
            )
            sig["mcnemar"] = _mcnemar_test(b, c)
            sig["mcnemar"]["positives_lost"] = b
            sig["mcnemar"]["positives_gained"] = c

        if "paired_t" in self.significance_tests:
            base_scores = [float(r.success) for r in base_results]
            mod_scores = [float(r.success) for r in modified_results]
            sig["paired_t"] = _paired_t_test(mod_scores, base_scores)

        return sig
