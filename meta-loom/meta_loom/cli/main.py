"""`metaloom` — stage dispatcher: collect / train / eval (cloud — Part C).

    metaloom collect --run-dir runs/it --model-name google/gemma-4-12B-it --dataset mmlu …
    metaloom train   --run-dir runs/it --epochs 6
    metaloom eval    --run-dir runs/it
"""
from __future__ import annotations

import argparse
import sys

from meta_loom.cli import build_anchor as _build_anchor
from meta_loom.cli import build_universal as _build_universal
from meta_loom.cli import collect as _collect
from meta_loom.cli import eval as _eval
from meta_loom.cli import train as _train

_STAGES = {
    "collect": _collect,
    "train": _train,
    "eval": _eval,
    "build-universal": _build_universal,
    "build-anchor": _build_anchor,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metaloom", description="Meta-Loom stages for training the meta-attention wrapper")
    sub = p.add_subparsers(dest="stage", required=True)
    for name, mod in _STAGES.items():
        sp = sub.add_parser(name, help=mod.__doc__.splitlines()[0] if mod.__doc__ else name)
        mod.add_args(sp)
        sp.set_defaults(_run=mod.run)
    # cloud — safe provision/destroy of vast instances (Part C)
    from meta_loom import cloud as _cloud
    sp = sub.add_parser("cloud", help="safe provision/destroy of vast instances (kill-switch)")
    _cloud.add_args(sp)
    sp.set_defaults(_run=_cloud.run)
    return p


def main(argv=None) -> int:
    # UTF-8 stdout on the Windows console (docstrings/markers contain non-ASCII + '→')
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    try:
        from meta_core.quiet import quiet_transformers
        quiet_transformers()                            # F1/F5: quiet transformers (pad spam, [ERROR])
    except Exception:
        pass
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    args._run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
