"""Helpers for building modifier training targets and labels.

The main loss is language-modeling cross-entropy via the HF model's `outputs.loss`.
Prompt masking is done via `labels[i, :prompt_len] = -100`
(the standard PyTorch CE skip).

The Phase 2 target is built by action type:

    if pass1_correct:
        target = " B) 4 Hz"                                  # confirm
    elif random() < correction_ratio:
        target = " Wait, the correct answer is B) 4 Hz."     # correct
    else:
        target = " I'm not confident enough to answer..."    # refuse

Self-correction and calibrated refusal are trained on the same LM CE — the model
learns to produce, from the cognitive tokens, a chain that leads to the correct
target. No separate classification heads.
"""

from __future__ import annotations

import random as _random
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "ActionType",
    "AgenticActionType",
    "build_correction_target",
    "build_target_by_action",
    "build_agentic_target",
    "make_labels_with_prompt_mask",
    "chunked_lm_loss",
]


ActionType = Literal["confirm", "correct", "refuse"]
AgenticActionType = Literal["code", "lookup"]


REFUSAL_TEMPLATES = [
    " I'm not confident enough to answer this question accurately.",
]

CORRECTION_TEMPLATE = " Wait, the correct answer is {answer}."

# Agentic targets — same LM-CE mechanism as QA, but the "uncertain" branch routes the
# latent doubt into a TOOL CALL instead of a text refusal (project-agentic-session-training).
# Confident (pass1 solved it directly) → start writing code; uncertain → emit docs_lookup.
# The tool-call string matches the native Qwen format parsed by the agentic harness
# (`<tool_call>{json}</tool_call>`), so a trained nudge becomes a real tool invocation.
AGENTIC_CODE_TEMPLATE = "\n```python\n"
AGENTIC_LOOKUP_TEMPLATE = (
    '\n<tool_call>\n{{"name": "docs_lookup", "arguments": {{"query": "{query}"}}}}\n</tool_call>'
)


def build_agentic_target(
    query: str,
    pass1_correct: bool,
) -> tuple[str, AgenticActionType]:
    """Build an agentic (target_text, action_type) from the Pass-1 result.

    Parallel to `build_correction_target`, but routes doubt into action:
      - pass1_correct (base solved it directly) → start a code block ("code").
      - pass1 wrong (base couldn't / was uncertain) → emit a `docs_lookup` tool call ("lookup").

    Args:
        query: the lookup query for the uncertain branch (e.g. the task intent). Stored per
            sample in `DatasetSample.ground_truth` by the agentic loader.
        pass1_correct: whether the base got it right on its own (execution-graded for ODEX).

    Returns:
        (target_text, action_type) — for Pass-2 supervision via the same LM CE.
    """
    if pass1_correct:
        return AGENTIC_CODE_TEMPLATE, "code"
    safe = (query or "").replace("\\", " ").replace('"', "'").replace("\n", " ").strip()
    return AGENTIC_LOOKUP_TEMPLATE.format(query=safe), "lookup"


def build_correction_target(
    ground_truth: str,
    pass1_correct: bool,
    correction_ratio: float = 0.5,
    rng: Optional[_random.Random] = None,
) -> tuple[str, ActionType]:
    """Build target_text + action_type from the Pass 1 result.

    Args:
        ground_truth: the correct answer ("B) 4 Hz" for MMLU MC).
        pass1_correct: whether the model got it right on its own.
        correction_ratio: fraction of correct vs refuse for incorrect pass1 (0.5 = 50/50).
        rng: optional random.Random for reproducibility.

    Returns:
        (target_text, action_type) — for Pass 2 supervision.

    Port from publish/github/src/phase2_selective_llama8b/
                01_prepare_correction_dataset.py:create_correction_target
    """
    rng = rng or _random
    if pass1_correct:
        return f" {ground_truth}", "confirm"
    if rng.random() < correction_ratio:
        return CORRECTION_TEMPLATE.format(answer=ground_truth), "correct"
    return REFUSAL_TEMPLATES[0], "refuse"


def build_target_by_action(
    ground_truth: str,
    action: ActionType,
    pass1_text: Optional[str] = None,
) -> str:
    """Build target_text from an explicit action type (no randomness).

    Used when we want to reproduce the target from a saved action_type label
    (for evaluation or fine-tuning).
    """
    if action == "confirm":
        return f" {ground_truth}"
    if action == "correct":
        return CORRECTION_TEMPLATE.format(answer=ground_truth)
    if action == "refuse":
        return REFUSAL_TEMPLATES[0]
    raise ValueError(f"Unknown action: {action!r}")


def make_labels_with_prompt_mask(
    input_ids,
    prompt_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
):
    """Make labels from input_ids with prompt + padding positions masked.

    Used in DataLoader collate: tokenize(input + target) → full input_ids,
    then labels = input_ids.clone(), but the first prompt_len positions are masked,
    and pad tokens are masked too.

    Args:
        input_ids: [B, seq_len] from the tokenizer.
        prompt_len: length of the prompt part (without target).
        pad_token_id: the tokenizer's pad-token id.
        ignore_index: value to use for masked positions (default -100).

    Returns:
        labels of the same shape, with -100 on the prompt and pad positions.

    Port from publish/github/src/phase2_selective_llama8b/03_train.py:collate_batch
    """
    labels = input_ids.clone()
    seq_len = input_ids.shape[-1]
    cut = min(prompt_len, seq_len)
    labels[..., :cut] = ignore_index
    labels[labels == pad_token_id] = ignore_index
    return labels


def chunked_lm_loss(hidden, lm_head_weight, labels, n_chunks: int = 4,
                    ignore_index: int = -100, softcap: Optional[float] = None):
    """Shift-LM CE in chunks along (B·seq), WITHOUT materializing the full [.., vocab]
    logit tensor. For large vocabularies (262K on Gemma-4) `[seq, vocab]` fp32 is the main
    VRAM hog (a documented OOM on T4/4GB); chunking + per-chunk checkpoint (logits
    recomputed in backward) keep the peak ≈ one chunk, not the whole vocab × all tokens.

    Numerically identical to `F.cross_entropy(reduction='mean', ignore_index=...)` on full
    logits (sum over target tokens / number of target tokens).

    Args:
        hidden: [B, seq, H] final hidden states (after norm).
        lm_head_weight: [vocab, H] lm_head weights (frozen base) OR a callable module
            (`h → logits`). For nf4-lm_head (Gemma on 4GB) the MODULE `Linear4bit` is passed
            (it dequants the packed weight internally; `F.linear` on the packed weight would fail).
            The gradient through the frozen Linear4bit flows into the wrapper (grad w.r.t. input;
            grad w.r.t. weight = None, by QLoRA design).
        labels: [B, seq], target tokens; ignore_index — mask (prompt/pad).
        softcap: CRITICAL for Gemma (final_logit_softcapping, 30.0 on Gemma-4-12B):
            `logits = softcap·tanh(logits/softcap)` BEFORE CE. Without it, gradients on Gemma
            blow up (logits up to ±120). The value lives in `config.text_config.
            final_logit_softcapping` (NOT the top level). Llama/Mistral have no softcap
            → None (as before).
    """
    sh_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    sh_labels = labels[:, 1:].reshape(-1).to(sh_hidden.device)
    keep = sh_labels != ignore_index
    sh_hidden, sh_labels = sh_hidden[keep], sh_labels[keep]
    n = sh_hidden.size(0)
    if n == 0:
        return hidden.sum() * 0.0  # keep the graph, zero contribution
    chunk = max(1, (n + n_chunks - 1) // n_chunks)

    # lm_head_weight may be a weight tensor OR a callable module (Linear4bit for nf4).
    _is_module = callable(lm_head_weight) and not isinstance(lm_head_weight, torch.Tensor)

    def _chunk_loss(h, lbl):
        logits = (lm_head_weight(h) if _is_module else F.linear(h, lm_head_weight)).float()
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits, lbl, reduction="sum")

    loss_sum = sh_hidden.new_zeros(())
    for i in range(0, n, chunk):
        h, lbl = sh_hidden[i:i + chunk], sh_labels[i:i + chunk]
        if h.requires_grad:
            loss_sum = loss_sum + checkpoint(_chunk_loss, h, lbl, use_reentrant=False)
        else:
            loss_sum = loss_sum + _chunk_loss(h, lbl)
    return loss_sum / n
