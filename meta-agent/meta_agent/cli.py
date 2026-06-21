"""meta-agent CLI — practical use of the wrapper: interactive chat and agentic run.

  meta-agent chat --model Qwen/Qwen3.5-4B --checkpoint doubter.pt --quantization nf4
  meta-agent run  --model … --checkpoint … --stdtools "what is 17*23?"

Without --checkpoint — the bare base model (for comparison). --stdtools wires in calculator +
knowledge_base (demo tools) for agentic mode. Console-script: `meta-agent` (see pyproject [project.scripts]).
"""
from __future__ import annotations

import argparse
from typing import Optional


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None,
                   help="HF id of the base model (or pulled from --run-dir/run.json)")
    p.add_argument("--run-dir", default=None,
                   help="metaloom run-dir: auto-pull model/layers/checkpoint from run.json (after train)")
    p.add_argument("--checkpoint", default=None, help="Doubter .pt (without it — the bare base model)")
    p.add_argument("--quantization", default=None, choices=["int8", "nf4", "fp4"],
                   help="quantization of the frozen base model (default: none)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="auto", help="auto = cuda if available, else cpu")
    p.add_argument("--target-layers", default="late",
                   help="'late' / 'all' / COMMA-SEPARATED indices (e.g. 16,17,18 — no spaces)")
    p.add_argument("--cross-attn-layers", default="late", help="same as --target-layers")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--repetition-penalty", type=float, default=1.3,
                   help="repetition penalty (>1.0 damps the Doubter refusal loop 'I'm not confident…'); 1.0 = off")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--stdtools", action="store_true", help="wire in calculator + knowledge_base")
    p.add_argument("--max-memory", default=None,
                   help='offload budget JSON for a large base on a small GPU (8B on 4GB), '
                        'e.g. {"0":"3GiB","cpu":"16GiB"}')


def _layers(s: str):
    return s if s in ("late", "all") else [int(x) for x in s.split(",")]


def _resolve_from_run_dir(args) -> None:
    """--run-dir → pull model_name / resolved layers / checkpoint from run.json (explicit flags win)."""
    if not args.run_dir:
        return
    import json
    from pathlib import Path
    rj = Path(args.run_dir) / "run.json"
    if not rj.exists():
        raise SystemExit(f"no {rj} — this is not a metaloom run-dir (collect+train required)")
    m = json.loads(rj.read_text(encoding="utf-8"))
    if not args.model:
        args.model = m["model_name"]
    if args.target_layers == "late" and m.get("target_layers"):
        args.target_layers = ",".join(str(x) for x in m["target_layers"])
    if args.cross_attn_layers == "late" and m.get("cross_attn_layers"):
        args.cross_attn_layers = ",".join(str(x) for x in m["cross_attn_layers"])
    if not args.checkpoint:
        ck = Path(args.run_dir) / "doubter_checkpoint.pt"
        if ck.exists():
            args.checkpoint = str(ck)


def _build_agent(args):
    """args -> a ready MetaAgent. Lazy-import of Meta-Core (GPU path)."""
    import json
    _resolve_from_run_dir(args)
    if not args.model:
        raise SystemExit("--model (HF id of the base model) or --run-dir with run.json is required")
    from meta_core import MetaSpiderConfig  # lazy
    from .build import build_agent
    from .stdtools import calculator, knowledge_base
    mm = None
    if args.max_memory:
        raw = json.loads(args.max_memory)
        mm = {(int(k) if str(k).isdigit() else k): v for k, v in raw.items()}
    cfg = MetaSpiderConfig(
        model_name=args.model, device=args.device, dtype=args.dtype,
        quantization=args.quantization,
        target_layers=_layers(args.target_layers),
        cross_attn_layers=_layers(args.cross_attn_layers),
        max_memory=mm,                                  # F6: offload a large base on a small GPU
        cpu_offload_fp32=bool(mm and args.quantization))  # nf4+cpu-offload requires this
    tools = [calculator, knowledge_base] if args.stdtools else None
    return build_agent(cfg, args.checkpoint, tools=tools, system=args.system,
                       max_new_tokens=args.max_new_tokens, max_steps=args.max_steps,
                       repetition_penalty=args.repetition_penalty)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="meta-agent",
                                 description="Practical use of the two-pass wrapper: chat + agent.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("chat", help="interactive chat with an attached Doubter")
    _add_model_args(pc)
    pr = sub.add_parser("run", help="a single agentic task → answer")
    _add_model_args(pr)
    pr.add_argument("task", help="task/question text")
    return ap


def main(argv: Optional[list] = None) -> None:
    import sys
    if sys.platform == "win32":   # #1: --help with non-ASCII (→) crashed with UnicodeEncodeError on a cp1251 console
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    try:
        from meta_core.quiet import quiet_transformers
        quiet_transformers()                            # F1/F5: quiet transformers
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    agent = _build_agent(args)
    if args.cmd == "chat":
        from .chat import ChatLoop
        ChatLoop(agent).repl()
    elif args.cmd == "run":
        print(agent.run(args.task).answer)


if __name__ == "__main__":
    main()
