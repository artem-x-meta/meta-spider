"""`metaloom build-universal` — one command to build a *general* uncertainty Doubter for any model.

The "universal Doubter factory" (`docs/results/qwen-14b/diverse-train-balanced.md`): instead of a
narrow wrapper over-fit to one task, assemble a *balanced* diverse agentic mix (commit + hold across
the decision space) and run the full collect → train pipeline, model-agnostically.

    metaloom build-universal --run-dir runs/qwen14b-uni --model-name Qwen/Qwen2.5-14B-Instruct \
        --quantization nf4 --dtype bfloat16 --per-class 70 --epochs 6 \
        --suite suite_v1.json            # excludes the suite's questions from training (leakage guard)

Produces <run-dir>/doubter_checkpoint.pt + run.json (a publishable wrapper). Evaluate per-axis with the
held-out suite separately (the two-method suite eval lives in lab / the eval scripts).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from meta_loom.cli import _common as C
from meta_loom.cli import collect as collect_mod
from meta_loom.cli import train as train_mod


def _suite_exclude(suite_path: Optional[str]) -> set:
    """Normalized question set from a held-out suite json (list of {question,...}) — leakage guard."""
    if not suite_path:
        return set()
    from meta_loom.data.agentic_mix import _norm
    data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    return {_norm(x["question"]) for x in data if x.get("question")}


def build_universal_stage(
    run_dir: str,
    model_name: str,
    *,
    per_class: int = 70,
    cap_tok: int = 320,
    epochs: int = 6,
    learning_rate: float = 1e-4,
    target_layers: str = "late",
    cross_attn_layers: str = "late",
    encoder_type: str = "selective",
    dtype: str = "bfloat16",
    quantization: Optional[str] = None,
    init_from: Optional[str] = None,
    suite_path: Optional[str] = None,
    tool_call_format: str = "auto",
    eval_suite: bool = False,
    eval_axes: Optional[list] = None,
    export_gguf: bool = False,
    device: Optional[str] = None,
    gradient_checkpointing: bool = False,
    max_memory: Optional[dict] = None,
    pipeline=None,
    tokenizer=None,
    mix: Optional[tuple] = None,
    verbose: bool = True,
) -> Path:
    """Assemble the diverse mix, collect activations, train the wrapper → publishable run-dir.

    pipeline/tokenizer/mix are injectable (CPU tests without GPU/HF). Returns the checkpoint path.
    """
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    C.write_status(run_dir, stage="build-universal", state="running")

    # 1) model-agnostic manifest skeleton ('late' layers resolved by the pipeline; no-think for
    #    thinking models; sizes filled once the mix is built).
    manifest = {
        "format_version": C.MANIFEST_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "dtype": dtype,
        "quantization": quantization,
        "device": device or "auto",
        "target_layers": C.parse_layers(target_layers),
        "cross_attn_layers": C.parse_layers(cross_attn_layers),
        "encoder_type": encoder_type,
        "prompt_format": "auto",
        "attn_implementation": None,
        "quantize_lm_head": False,
        "answer_suffix": None,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
        "dataset": "diverse-agentic-mix",
        "train_size": 0, "val_size": 0, "test_size": 0,  # set after the mix is built
    }

    # 2) pipeline once (its tokenizer builds the mix → model loaded a single time)
    if pipeline is None:
        cfg = C.build_meta_config(manifest, device=device,
                                  gradient_checkpointing=gradient_checkpointing,
                                  max_memory=max_memory)
        from meta_core import MetaSpiderPipeline
        pipeline = MetaSpiderPipeline.from_pretrained(cfg)
    tok = tokenizer or pipeline.tokenizer

    # 3) the diverse mix (disjoint from the held-out suite)
    if mix is None:
        from meta_loom.data.agentic_mix import build_training_mix
        exclude = _suite_exclude(suite_path)
        if verbose and exclude:
            print(f"  leakage guard: excluding {len(exclude)} suite questions from training", flush=True)
        prompts, specs = build_training_mix(tok, per_class=per_class, cap_tok=cap_tok,
                                            exclude_questions=exclude,
                                            tool_call_format=tool_call_format, verbose=verbose)
    else:
        prompts, specs = mix

    n = len(prompts)
    n_val = max(2, n // 6)
    manifest["train_size"] = n - n_val
    manifest["val_size"] = n_val
    manifest["test_size"] = 0
    if verbose:
        print(f"  mix: {n} items → train {n - n_val} / val {n_val}", flush=True)

    # 4) collect activations (inject the mix; targets ride along as ground_truth specs)
    collect_mod.collect_stage(
        manifest, run_dir, pipeline=pipeline,
        questions=prompts, ground_truths=specs,
        max_new_tokens=4, collect_chunk=max(50, per_class), verbose=verbose,
    )

    # 5) train on explicit multi-action targets (optionally continuing an existing wrapper)
    ckpt = train_mod.train_stage(
        run_dir, epochs=epochs, batch_size=1, grad_accumulation=16,
        learning_rate=learning_rate, max_seq_len=cap_tok + 96,
        agentic_targets=True, init_from=init_from,
        pipeline=pipeline, device=device, verbose=verbose,
    )

    # 6) publishable README in the run-dir
    import collections
    labs = collections.Counter(json.loads(s)["label"] for s in specs)
    (rd / "README.md").write_text(
        f"# Universal Doubter for `{model_name}`\n\n"
        f"A diverse-trained meta-attention Doubter (balanced commit + hold across the agentic decision "
        f"space). Built with `metaloom build-universal`.\n\n"
        f"- mix: {n} items, labels {dict(labs)}\n"
        f"- layers: {list(pipeline.config.target_layers)} (read + inject), encoder `{encoder_type}`\n"
        f"- load with `meta_core` (see run.json); the `gain` knob tunes caution at inference.\n",
        encoding="utf-8")

    # 7) optional per-axis eval on the held-out suite (base vs the trained wrapper)
    if eval_suite and suite_path:
        from meta_core import Doubter
        from meta_loom.evaluation.agentic_suite import compare_base_vs_doubter
        suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
        d = Doubter.from_checkpoint(str(ckpt))
        if verbose:
            print(f"  EVAL on suite ({len(suite)} items)…", flush=True)
        report = compare_base_vs_doubter(pipeline, d, suite, axes=eval_axes, verbose=verbose)
        (rd / "suite_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        C.mark_artifact(rd / "suite_eval.json")

    # 8) optional GGUF sidecar (llama.cpp / CPU / edge) — needs meta-deploy installed
    if export_gguf:
        try:
            from meta_deploy.export import export_from_run_dir
            gguf = export_from_run_dir(run_dir, verbose=verbose)
            C.mark_artifact(gguf)
            if verbose:
                print(f"  GGUF sidecar → {gguf}", flush=True)
        except ImportError:
            print("  [warn] --export-gguf needs meta-deploy installed (`pip install -e meta-deploy`); "
                  "skipped.", flush=True)

    C.write_status(run_dir, stage="build-universal", state="done", checkpoint=str(ckpt),
                   mix_size=n, labels=dict(labs))
    C.mark("STAGE_DONE", "build-universal")
    if verbose:
        print(f"  build-universal done → {ckpt}", flush=True)
    return ckpt


def add_args(p) -> None:
    p.add_argument("--run-dir", required=True, help="artifact directory (dataset.pt + run.json + ckpt)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--per-class", type=int, default=70, help="items per class-source (memory ≈ 2×)")
    p.add_argument("--cap-tok", type=int, default=320, help="skip mix items longer than this many tokens")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-layers", default="late", help="'late' / 'all' / comma indices")
    p.add_argument("--cross-attn-layers", default="late")
    p.add_argument("--encoder-type", default="selective",
                   choices=["selective", "multi_token", "transformer"])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default=None, choices=["int8", "nf4", "fp4"])
    p.add_argument("--init-from", default=None,
                   help="start from an existing Doubter checkpoint (e.g. continue a QA wrapper)")
    p.add_argument("--suite", default=None,
                   help="held-out suite json — its questions are EXCLUDED from training (leakage guard)")
    p.add_argument("--tool-call-format", default="auto",
                   help="native syntax of the tool-call TARGETS: auto (detect from the chat "
                        "template) / qwen / granite / llama")
    p.add_argument("--eval", action="store_true",
                   help="after training, run a per-axis log-prob eval (base vs wrapper) on --suite "
                        "→ suite_eval.json (floor / commit_mean). Action axes are exact; knowledge "
                        "axes via log-prob are approximate (see docs).")
    p.add_argument("--export-gguf", action="store_true",
                   help="also export a llama.cpp GGUF sidecar (needs meta-deploy installed)")
    p.add_argument("--device", default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-memory", default=None, help='offload budget JSON, e.g. {"0":"3GiB","cpu":"14GiB"}')


def run(args) -> None:
    build_universal_stage(
        args.run_dir, args.model_name,
        per_class=args.per_class, cap_tok=args.cap_tok, epochs=args.epochs,
        learning_rate=args.learning_rate,
        target_layers=args.target_layers, cross_attn_layers=args.cross_attn_layers,
        encoder_type=args.encoder_type, dtype=args.dtype, quantization=args.quantization,
        init_from=args.init_from, suite_path=args.suite,
        tool_call_format=args.tool_call_format,
        eval_suite=args.eval, export_gguf=args.export_gguf,
        device=args.device, gradient_checkpointing=args.gradient_checkpointing,
        max_memory=C.parse_max_memory(args.max_memory),
    )
