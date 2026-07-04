"""M4: behavioral comparison of the llama.cpp deploy vs PyTorch on the same questions.

Design criterion: llama.cpp refusal_precision within −5pp of PyTorch (same sidecar,
quantized base). Honest metrics against the base oracle (would base be wrong).

Flow:
  1) N trivia_qa questions (+truths).
  2) llama.cpp: driver in base mode (oracle) and doubter mode (batch, \0-separated files).
  3) PyTorch nf4: pipeline base (detach) + doubter — the same questions.
  4) both sides → honest metrics, comparison.

Usage:
  python behavioral_compare.py --n 40 --offset 6000 \
    --exe /c/Users/Impi/llamacpp-build/build/bin/llama-meta-generate.exe \
    --gguf /c/Users/Impi/llamacpp-build/_gguf/gemma2b-Q4_K_M.gguf \
    --sidecar <doubter_sidecar.gguf> --checkpoint <doubter.pt> \
    --framework-path meta-spider-framework-dev --loader-path publish/github
"""

import argparse, os, subprocess, sys, math
from pathlib import Path


def metrics(base_texts, dbt_texts, truths, check, classify):
    n = len(truths)
    base_ok = [bool(check(base_texts[i], truths[i])) for i in range(n)]
    refused = [classify(dbt_texts[i]) == "refuse" for i in range(n)]
    dbt_ok  = [bool(check(dbt_texts[i], truths[i])) for i in range(n)]
    n_ref = sum(refused)
    answered = [i for i in range(n) if not refused[i]]
    ref_prec = (sum(1 for i in range(n) if refused[i] and not base_ok[i]) / n_ref) if n_ref else float("nan")
    over_ref = (sum(1 for i in range(n) if refused[i] and base_ok[i]) / n_ref) if n_ref else float("nan")
    sel_acc = (sum(dbt_ok[i] for i in answered) / len(answered)) if answered else float("nan")
    return dict(base_acc=sum(base_ok)/n, refusal_rate=n_ref/n, refusal_precision=ref_prec,
                over_refusal=over_ref, selective_acc=sel_acc, n_refused=n_ref)


def run_driver(exe, gguf, env_extra, prompts, tmp, tag):
    pf = Path(tmp)/f"mg_prompts_{tag}.bin"; of = Path(tmp)/f"mg_out_{tag}.bin"
    pf.write_bytes(b"\x00".join(p.encode("utf-8") for p in prompts))
    env = dict(os.environ); env.update(env_extra)
    env["META_PROMPTS"] = str(pf); env["META_OUT"] = str(of)
    # The driver is built statically (objdump: imports only KERNEL32/ADVAPI32/WS2_32/msvcrt),
    # libgomp/winpthreads are compiled into the .exe — it loads no DLLs from PATH, so the crash
    # cause was NOT Intel-OpenMP from PATH. The real cause: libgomp teardown
    # (gomp_team_end → gomp_mutex_destroy → pthread_mutex_destroy) on mingw winpthreads
    # corrupted the heap after many OpenMP commands (gdb backtrace), crashing deterministically
    # around the ~17th prompt. Fix: the driver was rebuilt with GGML_OPENMP=OFF (native ggml threadpool).
    # The minimal PATH is kept as harmless environment hygiene.
    env["PATH"] = r"C:\Users\Impi\gcc\bin;C:\Windows\System32;C:\Windows"
    cmd = [exe, "-m", gguf, "-t", "4", "-c", "2048", "-ngl", "0", "-fit", "off"]
    # retry — insurance against rare startup failures; the main crash is fixed in the driver itself
    for attempt in range(4):
        if of.exists(): of.unlink()
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0 and of.exists():
            break
        print(f"[driver exit {r.returncode}, attempt {attempt+1}/4]", flush=True)
    else:
        raise RuntimeError("driver failed after 4 attempts")
    blob = of.read_bytes()
    return [b.decode("utf-8", "replace") for b in blob.split(b"\x00")][:len(prompts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=int, default=6000)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--framework-path", default=None, help="(vestigial — meta_loom is installed editable)")
    ap.add_argument("--loader-path", default=None, help="(vestigial — the native loader lives in meta_loom.data)")
    ap.add_argument("--layers", default="10,14,18,22,25")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantization", default="nf4")
    args = ap.parse_args()

    from meta_loom.data.dataset import check_answer_correctness, load_qa_dataset
    from meta_loom.evaluation.harness import classify_action
    check, classify = check_answer_correctness, classify_action

    questions, truths = load_qa_dataset("trivia_qa", args.n, offset=args.offset)
    print(f"trivia_qa: {len(questions)} questions (offset {args.offset})", flush=True)

    # --- llama.cpp ---
    common = dict(META_SIDECAR=args.sidecar, META_LAYERS=args.layers, META_NGEN=str(args.ngen))
    print("llama.cpp base...", flush=True)
    l_base = run_driver(args.exe, args.gguf, {**common, "META_BASE": "1"}, questions, args.tmp, "base")
    print("llama.cpp doubter...", flush=True)
    l_dbt = run_driver(args.exe, args.gguf, common, questions, args.tmp, "dbt")
    l_metrics = metrics(l_base, l_dbt, truths, check, classify)

    # --- PyTorch nf4 ---
    print("PyTorch pipeline...", flush=True)
    from meta_core import MetaSpiderConfig, MetaSpiderPipeline
    from meta_daimon import Doubter
    cfg = MetaSpiderConfig(model_name="google/gemma-2-2b-it", device=args.device, dtype="float16",
                           quantization=args.quantization, target_layers=[10,14,18,22,25],
                           cross_attn_layers=[6,12,18,24])
    pipe = MetaSpiderPipeline.from_pretrained(cfg)
    doubter = Doubter.from_checkpoint(args.checkpoint)
    def pgen(q):
        return pipe.generate(q, max_new_tokens=args.ngen, apply_chat_template=True)
    p_base, p_dbt = [], []
    for i, q in enumerate(questions):
        pipe.detach_all(); p_base.append(pgen(q))
        pipe.attach(doubter); p_dbt.append(pgen(q)); pipe.detach_all()
        if (i+1) % 10 == 0: print(f"  pt {i+1}/{len(questions)}", flush=True)
    p_metrics = metrics(p_base, p_dbt, truths, check, classify)

    def show(tag, m):
        print(f"  {tag}: base_acc={m['base_acc']:.3f} refusal_rate={m['refusal_rate']:.3f} "
              f"ref_prec={m['refusal_precision']:.3f} over_ref={m['over_refusal']:.3f} "
              f"sel_acc={m['selective_acc']:.3f} (n_ref={m['n_refused']})", flush=True)
    print("\n================ M4: llama.cpp vs PyTorch (trivia, honest metrics) ================")
    show("llama.cpp", l_metrics)
    show("PyTorch  ", p_metrics)
    d = l_metrics["refusal_precision"] - p_metrics["refusal_precision"]
    print(f"\n  Δ refusal_precision (llama−pt): {d:+.3f} ({d*100:+.1f}pp)  "
          f"criterion: |Δ| ≤ 0.05 → {'PASS' if abs(d) <= 0.05 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
