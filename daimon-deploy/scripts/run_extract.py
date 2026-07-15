"""Extract activations from GGUF for all quants with auto-resume (works around cumulative crash).

llama-extract-activations sometimes segfaults after ~300 prompts (cumulative resource
exhaustion in llama.cpp). Here we restart with EXTRACT_SKIP until the full set is done.

Usage:
  python run_extract.py --exe <...llama-extract-activations.exe> \
      --gguf-dir <...> --results <...> --layers 10,14,18,22,25 [--quants Q8_0,...]
"""

import argparse
import os
import subprocess
from pathlib import Path

ALL_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
REC_FLOATS = None  # n_layers*n_embd, computed below


def records_in(path: Path, rec_bytes: int) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size // rec_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--layers", default="10,14,18,22,25")
    ap.add_argument("--quants", default=",".join(ALL_QUANTS))
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=2304)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=200, help="prompts per run")
    args = ap.parse_args()

    results = Path(args.results)
    prompts = results / "prompts.bin"
    n_layers = len(args.layers.split(","))
    rec_bytes = n_layers * args.hidden * 4

    for quant in args.quants.split(","):
        gguf = Path(args.gguf_dir) / f"gemma2b-{quant}.gguf"
        out = results / f"gguf_activations_{quant}.bin"
        if not gguf.exists():
            print(f"[skip] {quant}: {gguf} missing")
            continue
        # start fresh for this quant
        if out.exists() and records_in(out, rec_bytes) >= args.n_prompts:
            print(f"[done] {quant}: already {args.n_prompts} records")
            continue
        if out.exists():
            out.unlink()  # clean start (avoid a broken partial from an old format)

        print(f"=== {quant} ===")
        restarts = 0
        while True:
            done = records_in(out, rec_bytes)
            if done >= args.n_prompts:
                break
            if restarts > 20:
                print(f"  [abort] {quant}: too many restarts at {done}")
                break
            env = dict(os.environ)
            env["EXTRACT_LAYERS"] = args.layers
            env["EXTRACT_PROMPTS"] = str(prompts.resolve())
            env["EXTRACT_OUT"] = str(out.resolve())
            env["EXTRACT_SKIP"] = str(done)
            env["EXTRACT_MAX"] = str(args.chunk)
            r = subprocess.run(
                [args.exe, "-m", str(gguf), "-c", str(args.ctx), "-t", str(args.threads)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            new_done = records_in(out, rec_bytes)
            print(f"  skip={done} -> {new_done}/{args.n_prompts} (exit={r.returncode})")
            if new_done == done:
                restarts += 1  # no progress — possibly a crash on this prompt
                if restarts > 3:
                    print(f"  [stall] {quant} stuck at {done}")
                    break
            else:
                restarts = 0
        print(f"  {quant}: {records_in(out, rec_bytes)}/{args.n_prompts} records")


if __name__ == "__main__":
    main()
