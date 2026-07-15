"""`metaloom eval` — selective evaluation (BaselineComparison: base vs Doubter) → report.json.

Reads the architecture and test split from the run.json manifest; the checkpoint comes from
the run-dir (or --checkpoint). Metrics are honest: selective_accuracy, refusal_rate,
refusal_precision (oracle=pass1), over_refusal.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from daimon_loom.cli import _common as C


def eval_stage(
    run_dir: str,
    *,
    checkpoint: Optional[str] = None,
    max_tokens: int = 80,
    device: Optional[str] = None,
    max_memory: Optional[dict] = None,
    pipeline=None,
    samples: Optional[list] = None,
    check_fn: Optional[Callable] = None,
    loader_path: Optional[str] = None,
    verbose: bool = True,
):
    """Run BaselineComparison on the test split → <run-dir>/report.json. Returns the report."""
    from meta_attention import MetaAttentionPipeline
    from daimon_voices import Doubter
    from daimon_loom import BaselineComparison, BenchmarkTask, QABenchmark

    rd = Path(run_dir)
    manifest = C.read_manifest(run_dir)
    C.write_status(run_dir, stage="eval", state="running")

    if samples is None:
        from daimon_loom import ActivationDatasetCollector
        ds = rd / "dataset.pt"
        if not ds.exists():
            raise FileNotFoundError(f"missing {ds} — run `metaloom collect --run-dir {run_dir}` first")
        samples = ActivationDatasetCollector.load(str(ds))
    tr, va, te = manifest["train_size"], manifest["val_size"], manifest["test_size"]
    from daimon_loom.data.splits import split_samples
    _, _, test_s = split_samples(samples, tr, va, te)  # held-out test + leakage guard

    if pipeline is None:
        # max_memory → offload (12B inference on 4GB). attn_implementation is taken from
        # the manifest (collect set eager for Gemma). Doubter inference = a full 2-pass
        # forward (NOT a slice — slicing is train-only), hence the whole model needs offload.
        cfg = C.build_meta_config(manifest, device=device, max_memory=max_memory)
        pipeline = MetaAttentionPipeline.from_pretrained(cfg)

    ckpt = checkpoint or str(rd / "doubter_checkpoint.pt")
    doubter = Doubter.from_checkpoint(ckpt)
    pipeline.attach(doubter)

    if check_fn is None:
        is_gsm = manifest.get("dataset") == "gsm8k"
        _, check_answer, check_gsm8k = C.resolve_dataset_loader(loader_path)
        check_fn = check_gsm8k if is_gsm else check_answer

    tasks = [
        BenchmarkTask(
            task_id=f"qa_{i}", prompt=s.input_text, expected_answer=s.ground_truth,
            check_fn=lambda pred, gts=(s.aliases or [s.ground_truth]): bool(check_fn(pred, gts)),
        )
        for i, s in enumerate(test_s)
    ]
    bench = QABenchmark(name=f"{manifest.get('dataset','qa')}_eval", tasks=tasks, scoring="custom")

    # input_text was already templated at collection time → apply_chat_template=False
    cmp = BaselineComparison(pipeline, bench, max_tokens=max_tokens, apply_chat_template=False)
    t0 = time.time()
    report = cmp.run(verbose=verbose)

    report_path = rd / "report.json"
    report.save_json(str(report_path))

    C.write_status(run_dir, stage="eval", state="done", report=str(report_path),
                   eval_sec=round(time.time() - t0, 1))
    C.mark_artifact(report_path)
    C.mark_peak_vram()
    C.mark("STAGE_DONE", "eval")
    if verbose:
        bm, mm = report.base_metrics, report.modified_metrics
        print(f"  eval done → {report_path}\n"
              f"    base sel_acc={bm.get('selective_accuracy')}  "
              f"doubter sel_acc={mm.get('selective_accuracy')} "
              f"refusal_prec={mm.get('refusal_precision')}", flush=True)
    return report


def add_args(p) -> None:
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run-dir>/doubter_checkpoint.pt")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-memory", default=None,
                   help='offload budget JSON (inference of a large base on a small GPU), '
                        'e.g. {"0":"3GiB","cpu":"13GiB"}')
    p.add_argument("--loader-path", default=None)


def run(args) -> None:
    eval_stage(
        args.run_dir, checkpoint=args.checkpoint, max_tokens=args.max_tokens,
        device=args.device, max_memory=C.parse_max_memory(args.max_memory),
        loader_path=args.loader_path,
    )
