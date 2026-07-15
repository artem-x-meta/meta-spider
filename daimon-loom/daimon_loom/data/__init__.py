"""daimon_loom.data — родные загрузчики датасетов (в пакете, не во внешнем репо)."""
from daimon_loom.data.dataset import (
    check_answer_correctness,
    check_gsm8k_answer,
    extract_gsm8k_number,
    load_qa_dataset,
    normalize_answer,
)
from daimon_loom.data.splits import assert_disjoint_from, split_samples

__all__ = [
    "load_qa_dataset", "check_answer_correctness", "check_gsm8k_answer",
    "extract_gsm8k_number", "normalize_answer",
    "split_samples", "assert_disjoint_from",
]
