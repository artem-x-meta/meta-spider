"""`metaloom collect` — stage that collects base activations → dataset.pt + run.json manifest.

A heavy forward-only pass (can go to the cloud/offload). Artifacts are self-contained:
dataset.pt carries its own config, run.json is the single source of truth for train/eval.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from meta_loom.cli import _common as C


def collect_stage(
    manifest: dict,
    run_dir: str,
    *,
    collect_batch: int = 4,
    collect_chunk: int = 250,
    offset: int = 0,
    max_new_tokens: int = 50,
    loader_path: Optional[str] = None,
    device: Optional[str] = None,
    gradient_checkpointing: bool = False,
    max_memory: Optional[dict] = None,
    pipeline=None,
    questions: Optional[list[str]] = None,
    ground_truths: Optional[list] = None,
    verbose: bool = True,
) -> Path:
    """Collect activations per the manifest → <run-dir>/dataset.pt + run.json.

    pipeline/questions/ground_truths are injectable (for tests without GPU/dataset).
    Returns the path to dataset.pt.
    """
    from meta_loom import ActivationDatasetCollector

    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    C.write_status(run_dir, stage="collect", state="running")

    total = manifest["train_size"] + manifest["val_size"] + manifest["test_size"]

    # 1) pipeline (resolves layer presets via model num_layers)
    if pipeline is None:
        cfg = C.build_meta_config(manifest, device=device,
                                  gradient_checkpointing=gradient_checkpointing,
                                  max_memory=max_memory)
        from meta_core import MetaSpiderPipeline
        pipeline = MetaSpiderPipeline.from_pretrained(cfg)

    # 2) freeze the RESOLVED layers + dimensions into the manifest (reproducibility)
    slice_mode = manifest.get("target_layers") == "late_slice"
    manifest = dict(manifest)
    manifest["target_layers"] = list(pipeline.config.target_layers)
    manifest["cross_attn_layers"] = list(pipeline.config.cross_attn_layers)
    manifest["hidden_dim"] = getattr(pipeline.config, "hidden_dim", None)
    manifest["num_layers"] = getattr(pipeline.config, "num_layers", None)
    # SLICE TRAINER: cut layer into the manifest → train reads it into TrainerConfig.slice_cut_layer
    slice_cut_layer = pipeline.config.slice_cut_layer() if slice_mode else None
    if slice_mode:
        manifest["slice_cut_layer"] = slice_cut_layer
        if verbose:
            print(f"  SLICE TRAINER: cut={slice_cut_layer}, cut_hidden on prompt+target", flush=True)

    # 3) dataset + prompt format
    if questions is None:
        load_qa_dataset, check_answer_correctness, _ = C.resolve_dataset_loader(loader_path)
        questions, ground_truths = load_qa_dataset(
            manifest["dataset"], total, offset=offset)
    else:
        check_answer_correctness = None  # injection: use the collector's default checker

    # answer_suffix: answer-format instruction placed inside the user turn (verbose instruct
    # models on MCQ explain instead of giving the letter → pass1 fails to parse). Carried via
    # input_text → eval/train inherit the same prompt automatically.
    answer_suffix = manifest.get("answer_suffix")
    if answer_suffix:
        questions = [q + answer_suffix for q in questions]

    has_ct = getattr(pipeline.tokenizer, "chat_template", None) is not None
    wrapped, apply_ct = C.resolve_prompt(questions, manifest["prompt_format"], has_ct)
    # chat_template_kwargs (e.g. enable_thinking=False) — only when actually templating
    ct_kwargs = manifest.get("chat_template_kwargs") or {}
    if verbose:
        print(f"  collect: {len(wrapped)} questions, prompt_format={manifest['prompt_format']}, "
              f"apply_chat_template={apply_ct}, suffix={'yes' if answer_suffix else 'no'}, "
              f"ct_kwargs={ct_kwargs}", flush=True)

    collector = ActivationDatasetCollector(
        pipeline, max_new_tokens=max_new_tokens, apply_chat_template=apply_ct,
        check_correctness=check_answer_correctness, batch_size=collect_batch,
        chat_template_kwargs=ct_kwargs,
    )

    # 4) chunked collection with partial-save/resume (as in run_gemma12b)
    dataset_path = rd / "dataset.pt"
    partial_path = rd / "dataset_partial.pt"
    samples: list = []
    if partial_path.exists():
        samples = ActivationDatasetCollector.load(str(partial_path))
        if verbose:
            print(f"  RESUME: {len(samples)} from {partial_path}", flush=True)

    t0 = time.time()
    while len(samples) < len(wrapped):
        start, end = len(samples), min(len(samples) + collect_chunk, len(wrapped))
        chunk = collector.collect(wrapped[start:end], ground_truths[start:end], verbose=False,
                                  slice_cut_layer=slice_cut_layer, correction_ratio=0.0)
        samples.extend(chunk)
        ActivationDatasetCollector.save(samples, str(partial_path), config=manifest)
        if verbose:
            print(f"  [{len(samples)}/{len(wrapped)}] chunk", flush=True)

    ActivationDatasetCollector.save(samples, str(dataset_path), config=manifest)
    partial_path.unlink(missing_ok=True)

    C.write_manifest(run_dir, manifest)
    C.write_status(run_dir, stage="collect", state="done",
                   dataset=str(dataset_path), n_samples=len(samples),
                   collect_sec=round(time.time() - t0, 1))
    C.mark_artifact(dataset_path)
    C.mark_peak_vram()
    C.mark("STAGE_DONE", "collect")
    n_ok = sum(1 for s in samples if s.pass1_correct)
    frac = n_ok / max(len(samples), 1)
    if verbose:
        print(f"  collect done: {len(samples)} samples, pass1_correct "
              f"{n_ok} ({100*frac:.1f}%) → {dataset_path}", flush=True)
    # F2: degenerate dataset — training will give NO signal (no correct/incorrect contrast).
    # Catches a silent failure (pass1=0; Gemma fp16→NaN→empty) BEFORE train→eval comes up zero.
    if n_ok == 0 or n_ok == len(samples):
        print(f"  [WARNING] DEGENERATE dataset: pass1_correct={n_ok}/{len(samples)} "
              f"({100*frac:.0f}%) — no correct/incorrect contrast, nothing to train the Doubter on.\n"
              f"            Causes: the model can't hold the answer format (try --mcq-direct), "
              f"the dataset is too easy/hard, or NaN logits (Gemma → --dtype bfloat16).",
              flush=True)
    elif frac < 0.1 or frac > 0.9:
        print(f"  [warn] skewed dataset: pass1_correct {100*frac:.0f}% — little contrast, "
              f"calibration may be weak.", flush=True)
    return dataset_path


def add_args(p) -> None:
    p.add_argument("--run-dir", required=True, help="artifact directory (dataset.pt + run.json)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset", default="mmlu",
                   help="built-in: mmlu / mmlu_hard / mmlu_pro / gsm8k / trivia_qa / simple_qa; "
                        'OR a path to your own .jsonl (lines {"question": str, "answer": str|[str]})')
    p.add_argument("--train-size", type=int, default=5000)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--test-size", type=int, default=300)
    p.add_argument("--target-layers", default="late",
                   help="'late' / 'all' / COMMA-SEPARATED indices (e.g. 16,17,18 — no spaces)")
    p.add_argument("--cross-attn-layers", default="late", help="same as --target-layers")
    p.add_argument("--slice", action="store_true",
                   help="SLICE TRAINER: read layers below the lowest CA + cut_hidden cache "
                        "(target_layers='late_slice'); for training an 8-12B wrapper on 4GB")
    p.add_argument("--encoder-type", default="selective",
                   choices=["selective", "multi_token", "transformer"])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization", default=None, choices=["int8", "nf4", "fp4"])
    p.add_argument("--prompt-format", default="auto",
                   choices=["auto", "gemma2", "gemma4_direct"])
    p.add_argument("--attn-implementation", default=None,
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="Gemma-4 (local/global hybrid, FA2 unsupported on global head_dim=512) → eager")
    p.add_argument("--quantize-lm-head", action="store_true",
                   help="force nf4 on lm_head (models with a huge vocabulary: Gemma 262K / Qwen3.5 248K) "
                        "→ fits in 4GB WITHOUT slice/offload")
    p.add_argument("--no-think", action="store_true",
                   help="enable_thinking=False in chat_template (thinking models Qwen3.5/Gemma-it): "
                        "otherwise the template opens <think> and a short pass1 never reaches the answer")
    p.add_argument("--answer-suffix", default=None,
                   help="answer-format instruction string, appended to the END of each question "
                        "(inside the user turn); for verbose instruct models on MCQ")
    p.add_argument("--mcq-direct", action="store_true",
                   help="preset: --no-think + the standard answer-suffix 'answer with ONLY the letter' "
                        "(MMLU and other A/B/C/D on thinking/verbose instruct models)")
    p.add_argument("--abstain-affordance", action="store_true",
                   help="append an explicit 'you may say you are not confident' instruction to every "
                        "question — the fair baseline: BOTH eval arms (base and Doubter) get the same "
                        "refusal affordance, so the refusal delta is not a prompt artifact")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--collect-batch", type=int, default=4)
    p.add_argument("--collect-chunk", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-memory", default=None,
                   help='offload budget JSON, e.g. {"0":"3GiB","cpu":"14GiB"} (large base on a small GPU)')
    p.add_argument("--loader-path", default=None,
                   help="your own external loader (directory with src/utils/dataset.py); "
                        "by default — the native meta_loom.data.dataset (MMLU/GSM8K/TriviaQA/…)")


# Standard MCQ answer-format instruction (A/B/C/D) for thinking/verbose instruct models.
MCQ_SUFFIX = ("\n\nAnswer with ONLY the letter of the correct option "
              "(A, B, C, or D). Do not explain.")

# Abstain affordance: gives BOTH eval arms the explicit option to refuse. Without it the
# base is structurally unable to refuse (an anti-refusal MCQ prompt) → base refusal ≈ 0 is a
# prompt artifact, and the Doubter's added-refusal delta is overstated. Rides on answer_suffix
# → lands inside the user turn at collect and flows through train AND eval via input_text —
# so the affordance reaches both arms automatically (the fair-baseline lesson).
# The wording matches REFUSAL_PHRASES ("not confident") so refusal detection fires.
ABSTAIN_AFFORDANCE_SUFFIX = (
    ' If you are not confident in the answer, reply "I\'m not confident enough to answer" '
    "instead of guessing.")


def resolve_prompt_flags(no_think: bool, mcq_direct: bool,
                         answer_suffix: Optional[str],
                         abstain_affordance: bool = False) -> tuple[Optional[str], Optional[dict]]:
    """(--no-think/--mcq-direct/--answer-suffix/--abstain-affordance) → (answer_suffix, chat_template_kwargs).

    Fixes the null result on thinking models: without disabling reasoning the template opens
    <think> and a short pass1 never reaches the answer (Qwen3.5/Gemma-it/Granite). --mcq-direct =
    the no_think preset + the standard 'letter only' suffix. The return value is pure (no I/O) → testable.

    Model-agnostic: we emit BOTH disable keys — Qwen/Gemma-it read `enable_thinking`,
    Granite reads `thinking`; the foreign key = an unused jinja variable (ignored).
    """
    nt = bool(no_think or mcq_direct)
    suffix = answer_suffix or (MCQ_SUFFIX if mcq_direct else None)
    if abstain_affordance:
        suffix = (suffix or "") + ABSTAIN_AFFORDANCE_SUFFIX
    return suffix, ({"enable_thinking": False, "thinking": False} if nt else None)


def run(args) -> None:
    # SLICE TRAINER: --slice → target_layers='late_slice' (read strictly below the lowest CA).
    # Requires late CA (an explicit list or 'late'); 'late_slice' is resolved by the framework.
    target_layers = "late_slice" if getattr(args, "slice", False) else args.target_layers
    answer_suffix, chat_template_kwargs = resolve_prompt_flags(
        args.no_think, args.mcq_direct, args.answer_suffix,
        abstain_affordance=getattr(args, "abstain_affordance", False))

    manifest = {
        "format_version": C.MANIFEST_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "device": args.device,
        "target_layers": C.parse_layers(target_layers),
        "cross_attn_layers": C.parse_layers(args.cross_attn_layers),
        "encoder_type": args.encoder_type,
        "prompt_format": args.prompt_format,
        "attn_implementation": args.attn_implementation,
        "quantize_lm_head": args.quantize_lm_head,
        "answer_suffix": answer_suffix,
        "chat_template_kwargs": chat_template_kwargs,
        "dataset": args.dataset,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
    }
    collect_stage(
        manifest, args.run_dir,
        collect_batch=args.collect_batch, collect_chunk=args.collect_chunk,
        offset=args.offset, max_new_tokens=args.max_new_tokens,
        loader_path=args.loader_path, device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        max_memory=C.parse_max_memory(args.max_memory),
    )
