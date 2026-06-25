"""Train/val/test splitting with a built-in leakage guard.

Classic DL holdout: the test split is never seen during training. The sample list is already shuffled at
collect time (`load_qa_dataset` shuffles with a fixed seed), so a sequential slice gives disjoint splits.
`split_samples(..., verify=True)` additionally ASSERTS that no question (`input_text`) appears in more than
one split — catching leakage from misconfigured sizes or from samples drawn from overlapping pools.
"""
from __future__ import annotations
from typing import Sequence, Tuple, List


def _question_key(sample) -> str:
    """Stable identity of a sample = its question text (works for DatasetSample or a dict)."""
    t = getattr(sample, "input_text", None)
    if t is None and isinstance(sample, dict):
        t = sample.get("input_text")
    return str(t)


def split_samples(
    samples: Sequence,
    train_size: int,
    val_size: int,
    test_size: int,
    *,
    verify: bool = True,
) -> Tuple[List, List, List]:
    """Slice a (collect-shuffled) sample list into disjoint train / val / test.

    Args:
        samples: the full collected list (shuffled at collect time).
        train_size / val_size / test_size: split sizes (sequential, non-overlapping).
        verify: if True, assert the three splits share no question (`input_text`) — raises
            ValueError with the overlap counts on leakage.

    Returns:
        (train, val, test) lists.
    """
    n = len(samples)
    end = train_size + val_size + test_size
    if end > n:
        raise ValueError(f"train+val+test={end} exceeds available samples={n}")

    train = list(samples[:train_size])
    val = list(samples[train_size:train_size + val_size])
    test = list(samples[train_size + val_size:end])

    if verify:
        tr = {_question_key(s) for s in train}
        vl = {_question_key(s) for s in val}
        ts = {_question_key(s) for s in test}
        leak_tt, leak_tv, leak_vt = tr & ts, tr & vl, vl & ts
        if leak_tt or leak_tv or leak_vt:
            raise ValueError(
                "DATA LEAKAGE between splits — questions must not be shared: "
                f"train∩test={len(leak_tt)}, train∩val={len(leak_tv)}, val∩test={len(leak_vt)}. "
                "Check train/val/test sizes and that samples come from one disjoint pool."
            )
    return train, val, test


def assert_disjoint_from(samples: Sequence, holdout: Sequence) -> None:
    """Guard for the CROSS-run / cross-dataset case: assert none of `samples` (e.g. a fresh train set)
    shares a question with `holdout` (e.g. a previously-used test set). Use when train and eval are
    sourced from DIFFERENT collects / index spaces (e.g. full-mmlu train vs mmlu_hard eval)."""
    held = {_question_key(s) for s in holdout}
    leak = {_question_key(s) for s in samples} & held
    if leak:
        raise ValueError(f"DATA LEAKAGE: {len(leak)} questions of this set also appear in the holdout set.")
