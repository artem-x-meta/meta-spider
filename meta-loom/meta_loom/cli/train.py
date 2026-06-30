"""`metaloom train` — train the Doubter on the collected activations (--run-dir).

A light stage: forward+backward through the FROZEN base over cached activations. On auto-device
(cpu on a laptop). It reads the architecture (layers/encoder) from the run.json manifest — flags
are not duplicated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from meta_loom.cli import _common as C


def train_stage(
    run_dir: str,
    *,
    epochs: int = 6,
    batch_size: int = 1,
    grad_accumulation: int = 16,
    learning_rate: float = 2e-4,
    max_seq_len: int = 128,
    early_stop_patience: int = 3,
    device: Optional[str] = None,
    gradient_checkpointing: bool = False,
    max_memory: Optional[dict] = None,
    agentic_targets: bool = False,
    init_from: Optional[str] = None,
    pipeline=None,
    samples: Optional[list] = None,
    verbose: bool = True,
) -> Path:
    """Train per the run-dir manifest → <run-dir>/doubter_checkpoint.pt + history.json.

    pipeline/samples are injectable (tests). Returns the path to the checkpoint.

    agentic_targets — multi-action routing (the universal-Doubter factory): derive explicit
      (target_text, label) per sample from its ground_truth spec instead of the QA selective path.
    init_from — start from an existing Doubter checkpoint (e.g. continue a QA wrapper into agentic).
    """
    from meta_core import Doubter, DoubterConfig, MetaSpiderPipeline
    from meta_loom import ActivationDatasetCollector, Trainer, TrainerConfig

    rd = Path(run_dir)
    manifest = C.read_manifest(run_dir)
    C.write_status(run_dir, stage="train", state="running")

    if samples is None:
        ds = rd / "dataset.pt"
        if not ds.exists():
            raise FileNotFoundError(f"missing {ds} — run `metaloom collect --run-dir {run_dir}` first")
        samples = ActivationDatasetCollector.load(str(ds))
    tr, va, te = manifest["train_size"], manifest["val_size"], manifest["test_size"]
    from meta_loom.data.splits import split_samples
    train_s, val_s, _ = split_samples(samples, tr, va, te)  # disjoint holdout + leakage guard

    # (determined before the pipeline-injection branch — also needed in TrainerConfig below)
    cut = manifest.get("slice_cut_layer")
    slice_offload = cut is not None and max_memory is not None
    if pipeline is None:
        cfg = C.build_meta_config(manifest, device=device,
                                  gradient_checkpointing=gradient_checkpointing,
                                  max_memory=max_memory)
        # SLICE TRAINER + offload (8-12B on 4GB): top→GPU (the slice is computed), bottom+embed→cpu.
        # Auto-offload would do the opposite → build a model-aware device_map (paths from the real
        # structure: Gemma-4 = model.language_model.layers). + Gemma fit: nf4-lm_head, eager.
        if slice_offload:
            from accelerate import init_empty_weights
            from transformers import AutoConfig
            from transformers import AutoModelForCausalLM as _AM

            from meta_core.slice_forward import slice_device_map_for_model
            _mcfg = AutoConfig.from_pretrained(manifest["model_name"])
            with init_empty_weights():
                _meta = _AM.from_config(_mcfg)
            cfg.device_map = slice_device_map_for_model(_meta, cut)
            cfg.max_memory = None              # device_map is self-contained
            cfg.cpu_offload_fp32 = True        # quant layers partly on cpu
            cfg.quantize_lm_head = True        # nf4-lm_head (Gemma 2GB→0.5GB)
            cfg.attn_implementation = "eager"  # Gemma local/global, FA2 unsupported
            del _meta
            print(f"  SLICE+offload: model-aware device_map cut={cut}, nf4-lm_head, eager, Adam8bit",
                  flush=True)
        pipeline = MetaSpiderPipeline.from_pretrained(cfg)

    selective = manifest["encoder_type"] == "selective"
    n_target = len(C.parse_layers(manifest["target_layers"])) \
        if isinstance(manifest["target_layers"], (list, tuple)) \
        else len(pipeline.config.target_layers)
    num_cog = n_target if selective else 8

    if init_from:
        # continue an existing wrapper (e.g. QA → agentic); rebuilds its own config from the ckpt
        doubter = Doubter.from_checkpoint(init_from)
        if verbose:
            print(f"  init_from: {init_from}", flush=True)
    else:
        doubter = Doubter(DoubterConfig(
            encoder_type=manifest["encoder_type"],
            encoder_bottleneck=256, encoder_gate_init=0.3,
            ca_bottleneck_dim=256, ca_num_heads=8, ca_dropout=0.1, ca_gate_init=0.3,
            num_cognitive_tokens=num_cog, token_preference_init=0.0,
            correction_ratio=0.0, enable_self_correction=False,
        ))
    pipeline.attach(doubter)

    # agentic factory: explicit per-sample (target_text, label) from the ground_truth spec
    tr_tgt = va_tgt = None
    if agentic_targets:
        from meta_loom.data.agentic_mix import targets_from_samples
        tr_tgt = targets_from_samples(train_s)
        va_tgt = targets_from_samples(val_s) if val_s else None

    trainer = Trainer(doubter, pipeline, TrainerConfig(
        epochs=epochs, batch_size=batch_size, grad_accumulation=grad_accumulation,
        learning_rate=learning_rate, gate_lr_multiplier=5.0, warmup_ratio=0.05,
        weight_decay=0.01, correction_ratio=0.0,
        pretrain_projectors=selective, early_stop_patience=early_stop_patience,
        max_seq_len=max_seq_len,
        # SLICE TRAINER: cut from the manifest → Pass 2 runs only the slice from the cached cut_hidden
        slice_cut_layer=manifest.get("slice_cut_layer"),
        optimizer=("adam8bit" if slice_offload else "adamw"),
    ))

    t0 = time.time()
    history = trainer.train(train_s, val_samples=(val_s or None),
                            targets_by_sample=tr_tgt, val_targets_by_sample=va_tgt,
                            checkpoint_dir=str(rd / "checkpoints"))
    ckpt = rd / "doubter_checkpoint.pt"
    doubter.save_checkpoint(str(ckpt))

    hist = {
        "train_loss": [float(x) for x in history.get("train_loss", [])],
        "val_loss": [float(x) for x in history.get("val_loss", [])],
        "best_val_loss": (float(history["best_val_loss"])
                          if history.get("best_val_loss") is not None else None),
        "ca_gate_map_per_epoch": [
            {int(k): float(v) for k, v in g.items()}
            for g in history.get("ca_gate_map", [])
        ],
    }
    (rd / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    C.write_status(run_dir, stage="train", state="done", checkpoint=str(ckpt),
                   best_val_loss=hist["best_val_loss"],
                   train_sec=round(time.time() - t0, 1))
    C.mark_artifact(ckpt)
    C.mark_peak_vram()
    C.mark("STAGE_DONE", "train")
    if verbose:
        print(f"  train done → {ckpt} (best_val_loss={hist['best_val_loss']})", flush=True)
    return ckpt


def add_args(p) -> None:
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--device", default="auto",
                   help="auto = cuda if available, otherwise cpu (the wrapper trains on cpu too)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-memory", default=None,
                   help='offload budget JSON, e.g. {"0":"3GiB","cpu":"14GiB"}')


def run(args) -> None:
    train_stage(
        args.run_dir, epochs=args.epochs, batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation, learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len, early_stop_patience=args.early_stop_patience,
        device=args.device, gradient_checkpointing=args.gradient_checkpointing,
        max_memory=C.parse_max_memory(args.max_memory),
    )
