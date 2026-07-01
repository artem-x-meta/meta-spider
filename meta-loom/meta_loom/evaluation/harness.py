"""EvalHarness — computes calibration metrics + classifies the model's action.

Meta-Spider-specific metrics (per the article):

  - **selective_accuracy**  = correct_among_non_refused / total_non_refused
    "when the model answers — how often it is right"
  - **refusal_rate**        = refused / total
  - **refusal_precision**   = refused_AND_would_be_wrong / refused
    "of the refusals, what fraction is justified" — the main calibration metric
  - **correction_accuracy** = successful_corrections / correction_attempts
  - **total_recovery**      = (successful_corrections + smart_refusals) / pass1_wrong
    "of the originally wrong, what fraction is 'rescued'"

Model action classification (by text):
  - "refuse"  — "not sure", "don't know", "cannot answer"
  - "correct" — "wait", "actually", "reconsider", "correction"
  - "confirm" — otherwise (just answered)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from meta_core.pipeline import MetaSpiderPipeline

__all__ = ["EvalHarness", "EvalResults", "classify_action", "compute_metrics"]


ActionType = Literal["confirm", "correct", "refuse"]


REFUSAL_PHRASES = (
    "not confident",
    "don't know",
    "not sure",
    "cannot answer",
    "i'm unsure",
    "unable to",
    "i don't know",
)

# Tightened (v0.3.1): the old list ("actually", "wait", "let me think") fired on ordinary
# speech/CoT and inflated the correction metrics. Match the TRAINED correction template
# (" Wait, the correct answer is …") and unambiguous phrasings only.
CORRECTION_PHRASES = (
    "reconsider",
    "correction:",
    "wait, the correct",
    "actually, the correct",
)


def classify_action(answer: str) -> ActionType:
    """Determine which action the model chose from its text."""
    a = answer.lower()
    if any(p in a for p in REFUSAL_PHRASES):
        return "refuse"
    if any(p in a for p in CORRECTION_PHRASES):
        return "correct"
    return "confirm"


def compute_metrics(
    predictions: list[str],
    ground_truths: list[str],
    pass1_correct: Optional[list[bool]] = None,
    check_correctness: Optional[Callable[[str, str], bool]] = None,
) -> dict[str, Any]:
    """Summary metrics over a list of predictions and ground truths.

    Args:
        predictions: the model's answer texts.
        ground_truths: the correct answers (for the correctness check).
        pass1_correct: opt. flags "did the model get it right in the Pass 1 baseline" —
            needed for the correction / total_recovery metrics.
        check_correctness: callable (pred, truth) → bool. Default: case-insensitive substring.

    Returns:
        a dict of metrics. If pass1_correct=None — the correction section is empty, and
        refusal_precision = None (it can't be computed honestly without the oracle).

    IMPORTANT about refusal_precision: honestly it means "of the refusals, the fraction where the
    model really would have been wrong had it answered". That requires the `pass1_correct` oracle
    (did the base/Pass-1 get it right). WITHOUT it, no number can be returned: the naive check
    "refusal text ≠ answer" is trivially ≈1.0 (a refusal never equals the answer) and does NOT
    measure calibration — this was a project-wide bug (see docs/results/honest-metrics-correction.md).
    """
    if check_correctness is None:
        # Same checker as the dataset loader (word-boundary MCQ letters, normalization) —
        # the old naive substring here ("b" matched inside any word) disagreed with it.
        from meta_loom.data.dataset import check_answer_correctness as _check
        check_correctness = lambda p, t: _check(p, [t])

    n_total = len(predictions)
    correct_flags = [check_correctness(p, t) for p, t in zip(predictions, ground_truths)]
    actions = [classify_action(a) for a in predictions]

    n_refused = sum(1 for a in actions if a == "refuse")
    n_answered = n_total - n_refused
    n_corrected = sum(1 for a in actions if a == "correct")
    n_confirmed = sum(1 for a in actions if a == "confirm")

    overall_acc = sum(correct_flags) / n_total if n_total else 0.0

    selective_correct = sum(c for c, a in zip(correct_flags, actions) if a != "refuse")
    selective_acc = selective_correct / n_answered if n_answered else 0.0

    refusal_rate = n_refused / n_total if n_total else 0.0

    # Honest refusal_precision — ONLY when the pass1_correct oracle is present:
    # of the refusals, the fraction where the base/Pass-1 really would have been wrong.
    # over_refusal_rate — the cost: refusals on questions the model knew.
    has_oracle = pass1_correct is not None and len(pass1_correct) == n_total
    if has_oracle and n_refused:
        refused_justified = sum(
            1 for i, a in enumerate(actions) if a == "refuse" and not pass1_correct[i]
        )
        refused_over = n_refused - refused_justified
        refusal_precision: Optional[float] = round(refused_justified / n_refused, 4)
        over_refusal_rate: Optional[float] = round(refused_over / n_refused, 4)
    else:
        # Can't be computed honestly without the oracle — None (NOT the trivial ≈1.0).
        refusal_precision = None
        over_refusal_rate = None

    correction_metrics: dict[str, Any] = {}
    if pass1_correct is not None and len(pass1_correct) == n_total:
        pass1_wrong = [i for i, c in enumerate(pass1_correct) if not c]
        n_p1w = len(pass1_wrong)
        if n_p1w > 0:
            attempted = sum(1 for i in pass1_wrong if actions[i] == "correct")
            successful = sum(
                1 for i in pass1_wrong
                if actions[i] == "correct" and correct_flags[i]
            )
            smart_refusals = sum(1 for i in pass1_wrong if actions[i] == "refuse")
            confirmed_wrong = sum(1 for i in pass1_wrong if actions[i] == "confirm")
            correction_metrics = {
                "n_pass1_wrong": n_p1w,
                "attempted_corrections": attempted,
                "successful_corrections": successful,
                "correction_rate": round(attempted / n_p1w, 4),
                "correction_accuracy": round(successful / attempted, 4) if attempted else 0.0,
                "smart_refusal_rate": round(smart_refusals / n_p1w, 4),
                "confirmed_wrong_rate": round(confirmed_wrong / n_p1w, 4),
                "total_recovery_rate": round(
                    (successful + smart_refusals) / n_p1w, 4
                ),
            }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "selective_accuracy": round(selective_acc, 4),
        "refusal_rate": round(refusal_rate, 4),
        "refusal_precision": refusal_precision,        # None without the pass1_correct oracle
        "over_refusal_rate": over_refusal_rate,        # the cost: refusals on known questions
        "n_total": n_total,
        "n_answered": n_answered,
        "n_refused": n_refused,
        "n_corrected": n_corrected,
        "n_confirmed": n_confirmed,
        "action_distribution": {
            "confirm": n_confirmed,
            "correct": n_corrected,
            "refuse": n_refused,
        },
        "correction": correction_metrics,
    }


@dataclass
class EvalResults:
    """Metrics summary for one run."""

    selective_accuracy: float
    refusal_rate: float
    refusal_precision: float
    overall_accuracy: float
    n_total: int
    n_refused: int
    correction: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any]) -> "EvalResults":
        return cls(
            selective_accuracy=metrics["selective_accuracy"],
            refusal_rate=metrics["refusal_rate"],
            refusal_precision=metrics["refusal_precision"],
            overall_accuracy=metrics["overall_accuracy"],
            n_total=metrics["n_total"],
            n_refused=metrics["n_refused"],
            correction=metrics.get("correction", {}),
            extras={k: v for k, v in metrics.items()
                    if k not in {"selective_accuracy", "refusal_rate", "refusal_precision",
                                 "overall_accuracy", "n_total", "n_refused", "correction"}},
        )


class EvalHarness:
    """Run an evaluation dataset through the attached pipeline + compute metrics.

    Args:
        pipeline: MetaSpiderPipeline with modifiers already attached.
        max_new_tokens: token limit per generation.
        check_correctness: opt callable (pred, truth) → bool. Default: substring.
    """

    def __init__(
        self,
        pipeline: "MetaSpiderPipeline",
        max_new_tokens: int = 80,
        check_correctness: Optional[Callable[[str, str], bool]] = None,
    ):
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.check_correctness = check_correctness

    def run(
        self,
        questions: list[str],
        ground_truths: list[str],
        pass1_correct: Optional[list[bool]] = None,
        verbose: bool = True,
    ) -> EvalResults:
        """Run pipeline.generate on every question + compute metrics."""
        predictions = []
        for i, q in enumerate(questions):
            if verbose and i % 20 == 0:
                print(f"  Evaluating {i}/{len(questions)}...")
            try:
                pred = self.pipeline.generate(q, max_new_tokens=self.max_new_tokens)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] generate failed at {i}: {e}")
                pred = ""
            predictions.append(pred)

        metrics = compute_metrics(
            predictions, ground_truths, pass1_correct=pass1_correct,
            check_correctness=self.check_correctness,
        )
        return EvalResults.from_metrics(metrics)
