"""Milestone 0 — the "calibration vs GGUF quant" curve (offline).

Two modes:
  --mode cossim     (local, encoder only, no base): activation cos-sim
                    fp16↔GGUF + cognitive-token cos-sim across quants. Core of the de-risk.
  --mode behavioral (needs an fp16 base in PyTorch, T4): for each question fills the
                    cognitive-token buffer from GGUF activations, runs Pass-2 generate,
                    computes refusal_precision / selective_accuracy.

Built-in sanity gate: Q8_0 activation cos-sim must be ≥0.999 vs fp16, otherwise
the ggml tensor tap was identified incorrectly (the script reports FAIL explicitly).

Usage (local):
  python offline_curve.py --mode cossim \
      --results-dir lab/experiments/llamacpp-deploy/results \
      --checkpoint <doubter_checkpoint.pt> \
      --framework-path meta-spider-framework

Usage (Kaggle, behavioral):
  python offline_curve.py --mode behavioral --device cuda \
      --results-dir <dir> --checkpoint <ckpt> --framework-path <fw> \
      --model-name google/gemma-2-2b-it
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
TARGET_LAYERS = [10, 14, 18, 22, 25]
# Sanity: tap alignment confirmed cross-layer (diagonal = max per layer).
# f16-GGUF vs HF fp16 = ~0.994 (inherent numerical engine difference, not quantization),
# so the absolute 0.999 threshold doesn't apply. The quantization baseline is F16 GGUF.
ENGINE_BASELINE = "F16"          # llama.cpp f16 — isolates quantization from the engine difference
GONOGO_QUANT = "Q4_K_M"
GONOGO_PRECISION = 0.90
FP16_REF_PRECISION = 0.975  # v11


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cos-sim for [N, D]."""
    at = torch.from_numpy(a).float()
    bt = torch.from_numpy(b).float()
    return F.cosine_similarity(at, bt, dim=-1).numpy()


def load_gguf_acts(results: Path, quant: str, n: int, n_layers: int, hidden: int):
    """GGUF .bin [n, n_layers, hidden] f32 (C-order) → np array, or None."""
    p = results / f"gguf_activations_{quant}.bin"
    if not p.exists():
        return None
    raw = np.fromfile(p, dtype=np.float32)
    expected = n * n_layers * hidden
    assert raw.size == expected, f"{quant}: {raw.size} != {expected}"
    return raw.reshape(n, n_layers, hidden)


def build_encoder(checkpoint: str, hidden: int):
    """Load MultiTokenEncoder from a checkpoint (no base — encoder forward only)."""
    from meta_daimon import Doubter

    doubter = Doubter.from_checkpoint(checkpoint)
    cfg = doubter.config
    from meta_core.encoders.multi_token import MultiTokenEncoder

    enc = MultiTokenEncoder(
        hidden_dim=hidden,
        num_layers=len(TARGET_LAYERS),
        num_cognitive_tokens=cfg.num_cognitive_tokens,
        bottleneck_dim=cfg.encoder_bottleneck,
    )
    enc.load_state_dict(doubter._pending_encoder_state)
    enc.eval()
    return enc


def encode(enc, acts: np.ndarray) -> np.ndarray:
    """acts [N, n_layers, hidden] → cognitive tokens [N, num_cog, hidden]."""
    with torch.no_grad():
        x = torch.from_numpy(acts).float()  # [N, L, H]
        activation_list = [x[:, j, :] for j in range(x.shape[1])]  # L × [N, H]
        cog = enc(activation_list)  # [N, num_cog, hidden]
    return cog.numpy()


def cog_cos_vs(enc, base_cog, acts):
    g_cog = encode(enc, acts)
    return float(np.mean([
        cos_per_row(base_cog[:, t, :].copy(), g_cog[:, t, :].copy()).mean()
        for t in range(base_cog.shape[1])
    ]))


def run_cossim(args, meta, results: Path):
    n, hidden = meta["n"], meta["hidden"]
    fp16 = np.load(results / "fp16_ref.npz")["activations"]  # HF fp16 reference [N,L,H]

    sys.path.insert(0, str(Path(args.framework_path).resolve()))
    enc = build_encoder(args.checkpoint, hidden)
    fp16_cog = encode(enc, fp16)

    nL = len(TARGET_LAYERS)
    # F16-GGUF baseline (engine floor) — isolates quantization from the llama.cpp↔HF difference
    f16g = load_gguf_acts(results, ENGINE_BASELINE, n, nL, hidden)
    f16g_cog = encode(enc, f16g) if f16g is not None else None

    def act_cos_vs(a, b):
        return float(np.mean([cos_per_row(a[:, j], b[:, j]).mean() for j in range(nL)]))

    rows = []
    # Cross-layer tap alignment check (on F16): the diagonal must be max
    aligned = None
    if f16g is not None:
        diag_ok = True
        for j in range(nL):
            sims = [cos_per_row(f16g[:, j], fp16[:, k]).mean() for k in range(nL)]
            if int(np.argmax(sims)) != j:
                diag_ok = False
        aligned = diag_ok

    for quant in [ENGINE_BASELINE] + QUANTS:
        g = load_gguf_acts(results, quant, n, nL, hidden)
        if g is None:
            print(f"  [skip] {quant}: bin missing")
            continue
        row = {
            "quant": quant,
            "act_cos_vs_hf": act_cos_vs(fp16, g),
            "act_cos_vs_f16gguf": act_cos_vs(f16g, g) if f16g is not None else None,
            "cog_cos_vs_hf": cog_cos_vs(enc, fp16_cog, g),
            "cog_cos_vs_f16gguf": (cog_cos_vs(enc, f16g_cog, g)
                                   if f16g_cog is not None else None),
            "act_cos_per_layer_vs_hf": {
                str(TARGET_LAYERS[j]): float(cos_per_row(fp16[:, j], g[:, j]).mean())
                for j in range(nL)},
        }
        rows.append(row)
        print(f"  {quant:8} act_vs_hf={row['act_cos_vs_hf']:.4f}  "
              f"act_vs_f16={row['act_cos_vs_f16gguf'] or 0:.4f}  "
              f"cog_vs_hf={row['cog_cos_vs_hf']:.4f}")

    verdict = {"tap_aligned": bool(aligned)}
    print(f"\nTAP ALIGNMENT (cross-layer diagonal = max): "
          f"{'PASS' if aligned else 'FAIL — tap identified incorrectly'}")
    print(f"Engine floor (F16-GGUF vs HF fp16): "
          f"{next((r['act_cos_vs_hf'] for r in rows if r['quant']==ENGINE_BASELINE), None)}")

    (results / "curve_cossim.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows, verdict


def run_behavioral(args, meta, results: Path):
    """Pass-2 eval with cognitive tokens from GGUF activations.

    Base in nf4 (fits a 4GB GPU; the Pass-2-nf4 cost was measured separately, v12 −0.2pp).
    Cognitive tokens — from GGUF-extracted activations → buffer → Pass-2 generate.
    Correctness — over all aliases (check_answer_correctness), refusal — classify_action.
    """
    from meta_core import MetaSpiderConfig, MetaSpiderPipeline
    from meta_daimon import Doubter
    from meta_loom.evaluation.harness import classify_action
    from meta_loom.data.dataset import check_answer_correctness

    n, hidden = meta["n"], meta["hidden"]
    prompts = meta["input_text"]
    aliases = meta["aliases"]
    gts = meta["ground_truth"]

    cfg = MetaSpiderConfig(
        model_name=args.model_name, device=args.device, dtype="float16",
        quantization=(args.quantization or None),
        target_layers=TARGET_LAYERS, cross_attn_layers=[6, 12, 18, 24],
    )
    pipeline = MetaSpiderPipeline.from_pretrained(cfg)
    doubter = Doubter.from_checkpoint(args.checkpoint)
    pipeline.attach(doubter)
    enc, tok, model = doubter.encoder, pipeline.tokenizer, pipeline.model
    device = next(model.parameters()).device

    def correct_of(pred, i):
        return check_answer_correctness(pred, aliases[i] or [gts[i]])

    # Conditions: HF fp16 reference (baseline, = v15 methodology) + selected quants
    conditions = {}
    fp16 = np.load(results / "fp16_ref.npz")["activations"]
    conditions["HF"] = fp16
    quants = args.quants_list.split(",") if args.quants_list else QUANTS
    for q in quants:
        g = load_gguf_acts(results, q, n, len(TARGET_LAYERS), hidden)
        if g is not None:
            conditions[q] = g

    rows = []
    for quant, g in conditions.items():
        actions, correct = [], []
        for i in range(n):
            acts = [torch.from_numpy(g[i, j]).float()[None].to(device)
                    for j in range(len(TARGET_LAYERS))]
            with torch.no_grad():
                cog = enc(acts)
            doubter.buffer.clear()
            doubter.buffer.fill(cog)
            if pipeline.collector is not None:
                pipeline.collector.freeze()
            enc_in = tok(prompts[i], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc_in, max_new_tokens=80, do_sample=False)
            if pipeline.collector is not None:
                pipeline.collector.unfreeze()
            doubter.buffer.clear()
            pred = tok.decode(out[0][enc_in.input_ids.shape[1]:],
                              skip_special_tokens=True).strip()
            actions.append(classify_action(pred))
            correct.append(correct_of(pred, i))
            if (i + 1) % 100 == 0:
                print(f"    {quant} {i+1}/{n}")

        n_ref = sum(a == "refuse" for a in actions)
        n_ans = n - n_ref
        sel = (sum(correct[i] for i in range(n) if actions[i] != "refuse") / n_ans
               if n_ans else 0.0)
        ref_prec = (sum(1 for i in range(n) if actions[i] == "refuse" and not correct[i])
                    / n_ref if n_ref else 0.0)
        row = {"quant": quant, "refusal_precision": round(ref_prec, 4),
               "selective_accuracy": round(sel, 4), "refusal_rate": round(n_ref / n, 4)}
        rows.append(row)
        print(f"  {quant:8} ref_prec={ref_prec:.4f} sel_acc={sel:.4f} ref_rate={n_ref/n:.4f}")

    verdict = {"gonogo": {"quant": GONOGO_QUANT, "threshold": GONOGO_PRECISION}}
    q4 = next((r for r in rows if r["quant"] == GONOGO_QUANT), None)
    if q4:
        verdict["gonogo"]["q4_precision"] = q4["refusal_precision"]
        verdict["gonogo"]["passed"] = q4["refusal_precision"] >= GONOGO_PRECISION
    (results / "curve_behavioral.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cossim", "behavioral"], required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--framework-path", default=None, help="(vestigial — meta_core/meta_loom are installed editable)")
    ap.add_argument("--dataset-loader-path", default=None, help="(vestigial — the native loader lives in meta_loom.data)")
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quantization", default=None, choices=["int8", "nf4", "fp4"],
                    help="behavioral: quantize the base (nf4 fits a 4GB GPU)")
    ap.add_argument("--quants-list", default=None,
                    help="behavioral: comma-separated subset of quants (default all)")
    args = ap.parse_args()

    results = Path(args.results_dir)
    meta = json.loads((results / "meta.json").read_text(encoding="utf-8"))

    if args.mode == "cossim":
        run_cossim(args, meta, results)
    else:
        run_behavioral(args, meta, results)


if __name__ == "__main__":
    main()
