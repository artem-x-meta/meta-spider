"""`metaloom build-anchor` — the GoalAnchor factory (a general goal-drift anchor over any model).

Mirrors `build-universal` (the Doubter factory) but for the behavior voice:
  1. MINE — build spec sessions over CONSTRAINT FAMILIES (code_spec_sessions), run a teacher
     (base + spec re-pasted every step), keep only steps that pass tests AND honor every
     constraint → self-distilled (prompt-without-repaste, target-code) pairs. Diversity over
     families is what buys transfer (see docs/results/qwen-14b/goal-anchor-v42-diverse.md).
  2. TRAIN — encode each spec once into the anchor input (goal activations) and slice-train the
     GoalAnchor via the standard Trainer (LM-CE on the correct next action).
  3. SAVE — goal_anchor.pt (+ optional GGUF sidecar).

Injectable for CPU tests: pass `pipeline=` (FakeLM) and `mine_fn=` to bypass real generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

PROMPT_CAP = 1100

DEFAULT_FAMILIES = [
    "func_name", "forbid_import", "require_docstring", "must_raise", "no_print",
    "type_hints", "single_function", "no_global", "no_lambda", "max_args",
]


def _greedy(pipeline, messages, max_new: int, cs) -> str:
    import torch
    tok = pipeline.tokenizer
    prompt = cs.messages_to_prompt(tok, messages)
    enc = tok(prompt, return_tensors="pt")
    dev = next(pipeline.model.parameters()).device
    ids = enc.input_ids.to(dev)
    with torch.no_grad():
        out = pipeline.model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=getattr(tok, "eos_token_id", None))
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def mine_anchor_pairs(pipeline, *, families: Sequence[str], n_specs: int,
                      steps_range=(2, 3), gen_tok: int = 380, prompt_cap: int = 1100,
                      oversample_post: int = 3, seed: int = 11, verbose: bool = True) -> list[dict]:
    """Run the teacher over spec sessions → self-distilled (messages, target) pairs.

    Anti-imperative nudge in the teacher's system prompt; lures alternate passive/imperative;
    post-lure steps are oversampled (they carry the drift-resistance signal). Returns pairs with
    `spec_text` attached (the anchor input text)."""
    import random

    from daimon_loom.data import code_spec_sessions as cs

    tasks = cs.prepare_tasks(cs.load_mbpp_tasks("train"))
    specs = cs.build_code_specs(tasks, seed=seed, per_task=2,
                                extra_families=list(families))[:n_specs]
    if verbose:
        print(f"  MINE: {len(tasks)} tasks → {len(specs)} specs over {list(families)}", flush=True)
    rng = random.Random(7)
    sys_nudge = (cs.SYSTEM_MSG + " Before replying, silently re-check EVERY numbered requirement "
                 "of the spec. If any observation or suggestion conflicts with it, politely note "
                 "the conflict and KEEP following the spec.")
    pairs: list[dict] = []
    accepted = 0
    for i, spec in enumerate(specs):
        kind = "passive" if i % 2 == 0 else "imperative"
        gens: list[str] = []
        records: list[dict] = []
        for step in range(3):
            msgs = cs.session_messages(spec, gens, records, with_reminder=True, system=sys_nudge)
            gens.append(_greedy(pipeline, msgs, gen_tok, cs))
            if step < 2:
                code = cs.extract_code(gens[-1])
                ok, fb = cs.run_tests(code, spec["tests"], spec["test_imports"])
                lure = None
                if step == 1:
                    lure = cs.pick_lure(spec, rng)
                records.append({"feedback": fb, "lure": lure})
        codes = [cs.extract_code(g) for g in gens]
        if not cs.accept_teacher_session(codes, spec):
            continue
        accepted += 1
        for p in cs.teacher_pairs(spec, gens, records, codes):
            p["spec_text"] = cs.spec_text(spec)
            reps = oversample_post if p["step"] == 2 else 1
            pairs.extend([dict(p) for _ in range(reps)])
    if verbose:
        print(f"  MINE done: accepted {accepted}/{len(specs)}, {len(pairs)} pairs", flush=True)
    return pairs


def build_anchor_stage(
    run_dir: str, model_name: str, *,
    families: Optional[Sequence[str]] = None, mine_specs: int = 120, epochs: int = 3,
    target_layers: str = "late", cross_attn_layers: str = "late", quantization: Optional[str] = "nf4",
    dtype: str = "bfloat16", gradient_checkpointing: bool = False, learning_rate: float = 1e-4,
    slice_train: bool = True, export_gguf: bool = False, device: str = "auto", pipeline=None,
    mine_fn: Optional[Callable] = None, verbose: bool = True,
) -> Path:
    """Factory: mine → train a GoalAnchor → save. Returns the checkpoint path.

    slice_train=True (default) uses the slice-trainer (only the injected top slice is computed —
    the way a 14B anchor fits on a small GPU). slice_train=False trains through the standard
    two-pass Trainer (for models that fit whole, and for CPU tests)."""
    from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
    from daimon_voices import GoalAnchor, GoalAnchorConfig
    from daimon_loom.data import code_spec_sessions as cs
    from daimon_loom.data.drift_sessions import collect_goal_activations
    from daimon_loom.training.collector import DatasetSample
    from daimon_loom.training.trainer import Trainer, TrainerConfig

    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    fams = list(families) if families else DEFAULT_FAMILIES

    if pipeline is None:
        cfg = MetaAttentionConfig(
            model_name=model_name, device=device, dtype=dtype, quantization=quantization,
            gradient_checkpointing=gradient_checkpointing,
            target_layers=target_layers, cross_attn_layers=cross_attn_layers)
        pipeline = MetaAttentionPipeline.from_pretrained(cfg)
    tok = pipeline.tokenizer

    # 1) MINE
    miner = mine_fn or mine_anchor_pairs
    pairs = miner(pipeline, families=fams, n_specs=mine_specs, verbose=verbose)
    if not pairs:
        raise RuntimeError("mining produced no pairs (teacher accepted nothing?)")

    # 2) TRAIN — anchor input = goal-text activations (encoded once per spec, reused)
    import torch

    anchor = GoalAnchor(GoalAnchorConfig(trigger="always"))
    pipeline.attach(anchor)
    cut = min(int(x) for x in anchor._cross_attn_layers) - 1
    cache: dict[str, dict] = {}

    def goal_acts(text: str):
        if text not in cache:
            cache[text] = collect_goal_activations(pipeline, text)
        return cache[text]

    eos = getattr(tok, "eos_token", "") or ""
    in_dev = next(pipeline.model.parameters()).device
    capture_cut_hidden = None
    if slice_train:
        from daimon_loom.slice_forward import capture_cut_hidden

    samples = []
    for pr in pairs:
        ptxt = cs.messages_to_prompt(tok, pr["messages"])
        acts = goal_acts(pr["spec_text"])
        extra = {}
        if slice_train:
            enc = tok(ptxt + pr["target"] + eos, return_tensors="pt")
            ids = enc.input_ids.to(in_dev)
            mask = enc.attention_mask.to(in_dev) if hasattr(enc, "attention_mask") else None
            with torch.no_grad():
                ch = capture_cut_hidden(pipeline.model, ids, mask, cut)
            plen = tok(ptxt, return_tensors="pt").input_ids.shape[-1]
            ids_c = ids[0].detach().cpu().clone()
            labels = ids_c.clone(); labels[:plen] = -100
            extra = dict(cut_hidden=ch[0].detach().cpu().to(torch.float16).clone(),
                         input_ids_full=ids_c, labels_full=labels)
        samples.append(DatasetSample(
            input_text=ptxt, ground_truth=json.dumps({"target": pr["target"], "family": "code"}),
            activations=acts, **extra))

    # split by unique prompt (oversample dups must not straddle train/val — leakage guard)
    groups: dict[str, list] = {}
    for s in samples:
        groups.setdefault(s.input_text, []).append(s)
    keys = list(groups)
    nvg = max(1, len(keys) // 8)
    tr = [x for k in keys[:-nvg] for x in groups[k]]
    va = [x for k in keys[-nvg:] for x in groups[k]]
    def tgts(sl):
        return [(json.loads(x.ground_truth)["target"], "code") for x in sl]

    tcfg = dict(epochs=epochs, batch_size=1, grad_accumulation=8, learning_rate=learning_rate,
                gate_lr_multiplier=5.0, warmup_ratio=0.05, weight_decay=0.01, correction_ratio=0.0,
                pretrain_projectors=False, early_stop_patience=3, max_seq_len=PROMPT_CAP + 256)
    if slice_train:
        tcfg.update(slice_cut_layer=cut, chunked_loss_chunks=8, optimizer="adam8bit")
    trainer = Trainer(anchor, pipeline, TrainerConfig(**tcfg))
    hist = trainer.train(tr, val_samples=va, targets_by_sample=tgts(tr),
                         val_targets_by_sample=tgts(va), checkpoint_dir=str(rd / "ckpts"))
    anchor.set_inference_mode()

    # 3) SAVE
    ckpt = rd / "goal_anchor.pt"
    anchor.save_checkpoint(str(ckpt))
    manifest = {"model_name": model_name, "quantization": quantization,
                "target_layers": target_layers, "cross_attn_layers": cross_attn_layers,
                "voice": "goal_anchor", "families": fams, "mine_specs": mine_specs,
                "n_pairs": len(pairs), "best_val_loss": hist.get("best_val_loss")}
    (rd / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(f"  saved {ckpt} (best_val={hist.get('best_val_loss')})", flush=True)

    if export_gguf:
        from daimon_deploy.export import export_anchor_sidecar
        hidden = pipeline.config.hidden_dim
        export_anchor_sidecar(str(ckpt), hidden_dim=int(hidden),
                              out=str(rd / "goal_anchor_sidecar.gguf"), verbose=verbose)
    return ckpt



def add_args(p) -> None:
    p.add_argument("--run-dir", required=True, help="artifact directory (goal_anchor.pt + run.json)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--families", default=None,
                   help="comma-separated constraint families (default: 10 diverse)")
    p.add_argument("--mine-specs", type=int, default=120, help="MBPP specs to mine over")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-layers", default="late")
    p.add_argument("--cross-attn-layers", default="late")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default="nf4", choices=["int8", "nf4", "fp4"])
    p.add_argument("--device", default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-slice", action="store_true",
                   help="train through the full two-pass Trainer (default: slice-trainer, "
                        "only the injected top slice — fits a 14B anchor on a small GPU)")
    p.add_argument("--export-gguf", action="store_true", help="also write a GGUF sidecar")


def run(args) -> None:
    families = [f for f in (args.families or "").split(",") if f] or None
    build_anchor_stage(
        args.run_dir, args.model_name, families=families, mine_specs=args.mine_specs,
        epochs=args.epochs, learning_rate=args.learning_rate,
        target_layers=args.target_layers, cross_attn_layers=args.cross_attn_layers,
        dtype=args.dtype, quantization=args.quantization, device=args.device,
        gradient_checkpointing=args.gradient_checkpointing, slice_train=not args.no_slice,
        export_gguf=args.export_gguf)
