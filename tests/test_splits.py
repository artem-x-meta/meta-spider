"""Tests for the train/val/test split + leakage guard (daimon_loom.data.splits)."""
import pytest

from daimon_loom.data import split_samples, assert_disjoint_from


def _samples(n, prefix="q"):
    return [{"input_text": f"{prefix}{i}", "ground_truth": "A"} for i in range(n)]


def test_split_sizes_and_disjoint():
    s = _samples(100)
    tr, va, te = split_samples(s, 60, 20, 20)
    assert (len(tr), len(va), len(te)) == (60, 20, 20)
    keys = lambda xs: {x["input_text"] for x in xs}
    assert keys(tr) & keys(te) == set()
    assert keys(tr) & keys(va) == set()
    assert keys(va) & keys(te) == set()


def test_split_too_large_raises():
    with pytest.raises(ValueError, match="exceeds available"):
        split_samples(_samples(10), 8, 2, 2)


def test_leakage_detected():
    # a duplicate question that lands in both train and test → must raise
    s = _samples(8)
    s.append({"input_text": "q0", "ground_truth": "A"})  # dup of train[0], placed into test region
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        split_samples(s, 5, 2, 2)  # train=[q0..q4], test=[q7, q0(dup)]


def test_verify_off_skips_check():
    s = _samples(8) + [{"input_text": "q0"}]
    tr, va, te = split_samples(s, 5, 2, 2, verify=False)  # no raise
    assert len(te) == 2


def test_assert_disjoint_from():
    train = _samples(5, "t")
    holdout = _samples(5, "t")  # same keys t0..t4
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_disjoint_from(train, holdout)
    assert_disjoint_from(_samples(5, "a"), _samples(5, "b"))  # disjoint → no raise


def test_works_with_dataclass_samples():
    from daimon_loom.training.collector import DatasetSample
    s = [DatasetSample(input_text=f"q{i}", ground_truth="A", activations={}) for i in range(10)]
    tr, va, te = split_samples(s, 6, 2, 2)
    assert len(tr) == 6 and len(te) == 2
