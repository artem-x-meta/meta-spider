"""Родной загрузчик QA-датасетов + проверка корректности ответов (meta-loom).

Раньше жил в `publish/github/src/utils/dataset.py` (исследовательский репо) → рандом-дев с
одним фреймворком не мог запустить `collect`/`eval` (F4-блокер, поймал Codex). Теперь это
часть пакета: `resolve_dataset_loader()` берёт его по умолчанию, `--loader-path` нужен лишь
для своего внешнего загрузчика.

Датасеты: trivia_qa / simple_qa / mmlu / mmlu_hard / mmlu_pro / gsm8k (качаются через `datasets`).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = [
    "load_qa_dataset", "check_answer_correctness", "check_gsm8k_answer",
    "extract_gsm8k_number", "normalize_answer",
]


def load_qa_dataset(
    dataset_name: str, num_questions: int, offset: int = 0
) -> Tuple[List[str], List[List[str]]]:
    """Загрузить вопросы + эталонные ответы. → (questions, ground_truths[список алиасов]).

    offset — пропустить первые N (разделение train/test). MCQ-датасеты возвращают текст с
    A)/B)/C)/D) в вопросе и список допустимых форм ответа («B) 4 Hz» / «B)» / «4 Hz»).
    """
    import os
    # Кастомный JSONL из коробки: --dataset path/to/data.jsonl, строки
    # {"question": str, "answer": str|[str]} (синонимы: prompt/input, answers/ground_truth).
    if dataset_name.endswith(".jsonl") or os.path.isfile(dataset_name):
        import json
        qs, gts = [], []
        with open(dataset_name, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                q = r.get("question") or r.get("prompt") or r.get("input")
                a = r["answer"] if "answer" in r else (r.get("answers") or r.get("ground_truth"))
                if q is None or a is None:
                    raise ValueError(
                        "JSONL: каждая строка должна быть {'question': str, 'answer': str|[str]} "
                        "(синонимы ключей: prompt/input, answers/ground_truth)")
                qs.append(q)
                gts.append([str(x) for x in a] if isinstance(a, (list, tuple)) else [str(a)])
        return qs[offset:offset + num_questions], gts[offset:offset + num_questions]

    from datasets import load_dataset  # lazy — `import meta_loom` остаётся лёгким

    if dataset_name == "trivia_qa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation").shuffle(seed=42)
        ds = ds.select(range(offset, min(offset + num_questions, len(ds))))
        questions = [item["question"] for item in ds]
        ground_truths = [item["answer"]["aliases"] for item in ds]

    elif dataset_name == "simple_qa":
        ds = load_dataset("openai/simple-qa", split="test").shuffle(seed=42)
        ds = ds.select(range(offset, min(offset + num_questions, len(ds))))
        questions = [item["problem"] for item in ds]
        ground_truths = [[item["answer"]] for item in ds]

    elif dataset_name in ("mmlu", "mmlu_hard"):
        ds = load_dataset("cais/mmlu", "all", split="test")
        if dataset_name == "mmlu_hard":
            hard_subjects = {
                "abstract_algebra", "college_mathematics", "college_physics",
                "formal_logic", "machine_learning", "high_school_mathematics",
                "college_chemistry", "electrical_engineering", "astronomy",
                "conceptual_physics", "college_computer_science",
                "computer_security", "medical_genetics", "virology",
            }
            ds = ds.filter(lambda x: x["subject"] in hard_subjects)
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(offset, min(offset + num_questions, len(ds))))
        letters = ["A", "B", "C", "D"]
        questions, ground_truths = [], []
        for item in ds:
            mc = item["question"] + "\n"
            for j, choice in enumerate(item["choices"]):
                mc += f"{letters[j]}) {choice}\n"
            questions.append(mc.strip())
            idx = item["answer"]
            text = item["choices"][idx]
            ground_truths.append([f"{letters[idx]}) {text}", f"{letters[idx]})", text])

    elif dataset_name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=42)
        ds = ds.select(range(offset, min(offset + num_questions, len(ds))))
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        questions, ground_truths = [], []
        for item in ds:
            mc = item["question"] + "\n"
            for j, opt in enumerate(item["options"]):
                mc += f"{letters[j]}) {opt}\n"
            questions.append(mc.strip())
            idx = item["answer_index"]
            text = item["options"][idx]
            ground_truths.append([f"{letters[idx]}) {text}", f"{letters[idx]})", text])

    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
        ds = ds.select(range(offset, min(offset + num_questions, len(ds))))
        questions = [item["question"] for item in ds]
        ground_truths = [[extract_gsm8k_number(item["answer"])] for item in ds]

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Доступны: trivia_qa, simple_qa, mmlu, "
            f"mmlu_hard, mmlu_pro, gsm8k. Свой датасет → передай --loader-path к модулю "
            f"с функцией load_qa_dataset(name, n, offset).")

    return questions, ground_truths


def extract_gsm8k_number(answer_text: str) -> str:
    """Финальное число из ответа GSM8K (после '#### '), без запятых-разделителей."""
    match = re.search(r"####\s*(.+)", answer_text)
    return match.group(1).strip().replace(",", "") if match else ""


def check_gsm8k_answer(generated: str, ground_truth) -> bool:
    """Содержит ли вывод правильное число: '#### N' или последнее число текста == эталон."""
    if not generated or not ground_truth:
        return False
    if isinstance(ground_truth, str):
        gt_list = [ground_truth]
    elif isinstance(ground_truth, (list, tuple)):
        gt_list = [str(x) for x in ground_truth]
    else:
        gt_list = [str(ground_truth)]

    match = re.search(r"####\s*(.+)", generated)
    if match:
        return match.group(1).strip().replace(",", "").rstrip(".") in gt_list
    # десятичная часть только с цифрами и старт с цифры: `\.?\d*` захватывал точку КОНЦА
    # ПРЕДЛОЖЕНИЯ («The answer is 5.» → «5.» ≠ «5»), а `[\d,]+` матчил одинокую запятую —
    # оба дефекта молча занижали оракул на фразовых ответах
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", generated)
    if numbers:
        return numbers[-1].replace(",", "") in gt_list
    return False


_MCQ_ANSWER_PATTERNS = (
    # explicit statements win: "answer is B", "answer: (B)", "correct option is B", "**B**)…"
    r"(?:answer|option|choice)\s*(?:is|:)?\s*\*{0,2}\(?([A-Ja-j])\)?(?:\b|[).:*])",
    # a leading bare letter: "B", "B).", "(B) …", "B: …"
    r"^\s*\*{0,2}\(?([A-Ja-j])\)?[).:\s*]",
)


def extract_mcq_letter(generated: str) -> Optional[str]:
    """Extract the chosen MCQ letter from a (possibly verbose) generation; None if ambiguous.

    Order: explicit "answer is X" patterns → a leading bare letter → the ONLY distinct
    standalone letter in the text. If several distinct letters are mentioned with no explicit
    pick ("Between B and C…"), returns None — the caller should treat it as not-an-answer.
    """
    text = generated.strip()
    if not text:
        return None
    for pat in _MCQ_ANSWER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    standalone = {m.group(1).upper() for m in re.finditer(r"\b([A-Ja-j])\b", text)}
    if len(standalone) == 1:
        return standalone.pop()
    return None


def check_answer_correctness(generated: str, ground_truths: List[str]) -> bool:
    """Содержит ли вывод хотя бы один эталонный ответ (нормализованное substring-совпадение).

    Для ОДНОБУКВЕННЫХ истин (MCQ) — извлечение выбранной буквы через `extract_mcq_letter`:
    старое `\\bB\\b`-совпадение засчитывало «Between B and C, I'd pick C» за B (буква
    упомянута ≠ буква выбрана) и перекашивало оракул pass1_correct на многословных выводах.
    Неоднозначный вывод (несколько букв, нет явного «answer is X») теперь считается НЕверным —
    консервативный оракул для калибровки безопаснее оптимистичного."""
    if not generated or not ground_truths:
        return False
    gen = normalize_answer(generated)
    letter_cache: Optional[str] = None
    letter_extracted = False
    for truth in ground_truths:
        if not truth:
            continue
        tl = normalize_answer(truth)
        if not tl or not gen:
            continue
        if len(tl) == 1 and tl.isalpha():
            # MCQ letter: compare against the EXTRACTED choice, not "mentioned anywhere".
            if not letter_extracted:
                letter_cache = extract_mcq_letter(generated)
                letter_extracted = True
            if letter_cache is not None and letter_cache.lower() == tl:
                return True
        elif min(len(tl), len(gen)) <= 2:
            short = tl if len(tl) <= 2 else gen
            long = gen if len(tl) <= 2 else tl
            if short == long or re.search(r"\b" + re.escape(short) + r"\b", long):
                return True
        elif tl in gen or gen in tl:
            return True
    return False


def normalize_answer(text: str) -> str:
    """Нижний регистр, удаление артиклей (кроме одиночного 'A' как ответа MC) и пунктуации."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b(?=\s+\S)", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
