"""metadeploy CLI — deploy the Doubter sidecar into llama.cpp.

  metadeploy export --run-dir lab/runs/qwen35_mcq
  metadeploy export --checkpoint d.pt --target-layers 10 14 18 22 25 \
                    --cross-attn-layers 6 12 18 24 --hidden-dim 2304 --out sidecar.gguf
  metadeploy validate --framework-path ../.. --checkpoint d.pt   # numpy↔PyTorch (phase M2a)

`export` builds the GGUF sidecar (reads run.json with --run-dir). `validate` checks the numpy
encoder spec against PyTorch — that spec is what ggml/C++ (cpp/) translates. Console-script: `metadeploy`.
"""
from __future__ import annotations

import argparse
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metadeploy",
        description="Deploy the Doubter sidecar into llama.cpp: GGUF sidecar + ggml forward.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="checkpoint/run-dir → GGUF sidecar")
    pe.add_argument("--run-dir", default=None,
                    help="metaloom run-dir: reads run.json (hidden_dim/layers) + doubter_checkpoint.pt")
    pe.add_argument("--checkpoint", default=None, help="path to doubter .pt (when used without --run-dir)")
    pe.add_argument("--target-layers", type=int, nargs="+", default=None,
                    help="Pass 1 read taps, e.g. 10 14 18 22 25")
    pe.add_argument("--cross-attn-layers", type=int, nargs="+", default=None,
                    help="CA injection layers, e.g. 6 12 18 24")
    pe.add_argument("--hidden-dim", type=int, default=None, help="base hidden_dim")
    pe.add_argument("--out", default=None, help="output .gguf (default <run-dir>/doubter_sidecar.gguf)")

    pv = sub.add_parser("validate", help="numpy encoder spec == PyTorch (phase M2a)")
    pv.add_argument("--framework-path", default=None,
                    help="fallback path to meta-spider-framework when meta_core is not installed")
    pv.add_argument("--checkpoint", required=True)
    pv.add_argument("--hidden-dim", type=int, default=2304)
    pv.add_argument("--num-layers", type=int, default=5)
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "export":
        from .export import export_from_run_dir, export_sidecar
        if args.run_dir:
            export_from_run_dir(args.run_dir, out=args.out)
        else:
            need = [n for n, v in (("--checkpoint", args.checkpoint),
                                   ("--target-layers", args.target_layers),
                                   ("--cross-attn-layers", args.cross_attn_layers),
                                   ("--hidden-dim", args.hidden_dim)) if not v]
            if need:
                raise SystemExit(f"without --run-dir these are required: {', '.join(need)}")
            export_sidecar(args.checkpoint, target_layers=args.target_layers,
                           cross_attn_layers=args.cross_attn_layers,
                           hidden_dim=args.hidden_dim, out=args.out or "doubter_sidecar.gguf")
    elif args.cmd == "validate":
        from .validate import validate_encoder
        validate_encoder(args.framework_path, args.checkpoint,
                         hidden_dim=args.hidden_dim, num_layers=args.num_layers)


if __name__ == "__main__":
    main()
