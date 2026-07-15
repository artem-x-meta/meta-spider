# daimon-loom

Training + evaluation for the Daimon wrapper. Depends on the **meta-attention** library and **daimon-voices** (+ **daimon-agent** for
agentic eval).

## CLI: `metaloom` (stages linked through `run.json`)

```bash
# 1. collect base activations → dataset.pt + run.json (the manifest = source of truth)
metaloom collect --run-dir runs/my --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset mmlu --train-size 80 --val-size 20 --test-size 40 --mcq-direct --device auto

# 2. train the Doubter from the run-dir (architecture/layers taken from run.json — no duplicate flags)
metaloom train --run-dir runs/my --epochs 10

# 3. selective eval: base vs Doubter → report.json (McNemar + paired t, no scipy)
metaloom eval --run-dir runs/my
```

### run-dir artifacts

| file | what |
|---|---|
| `run.json` | manifest: model_name, target/cross_attn_layers, encoder_type, hidden_dim, dataset, sizes |
| `dataset.pt` | collected activations + pass1 labels (train/val/test) |
| `doubter_checkpoint.pt` | trained wrapper (loaded via `Doubter.from_checkpoint`) |
| `report.json` | eval metrics: selective_accuracy / refusal_rate / refusal_precision / McNemar |

`run.json` is self-contained: `metaloom train/eval --run-dir` and `daimon-agent run --run-dir` read
everything from it.

### Datasets (`--dataset`)

- **Built-in:** `mmlu / mmlu_hard / mmlu_pro / gsm8k / trivia_qa / simple_qa` (fetched via `datasets`).
- **Your own JSONL — out of the box:** `--dataset path/to/data.jsonl`, lines `{"question": str, "answer": str|[str]}`
  (key synonyms: `prompt`/`input`, `answers`/`ground_truth`).
- **Your own loader:** `--loader-path <dir>` to a `src/utils/dataset.py` module exposing
  `load_qa_dataset(name, n, offset)`, `check_answer_correctness`, `check_gsm8k_answer`.

`--mcq-direct` for MCQ/thinking models (disables reasoning + asks for "the letter only"; otherwise
pass1 never reaches the answer and the dataset degenerates).

## Python API

| API | what |
|---|---|
| `BaselineComparison(pipeline, QABenchmark(...)).run()` | selective QA eval: base vs voice, with significance |
| `AgentComparison(pipeline, doubter=...).run([AgentTask(...)])` | agentic eval (multi-step tool use) via daimon-agent |
| `Trainer(doubter, pipeline, TrainerConfig(...))` | two-pass trainer (5-group AdamW, probe pretrain) |
| `ActivationDatasetCollector(pipeline)` | activation collection + cache |

Details and overall architecture — in the [root README](../README.md).
