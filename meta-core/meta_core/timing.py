"""StageTimer — a "how long does the pipeline take on your hardware" report.

Collects per-stage: wall-time + peak VRAM (if CUDA), plus a hardware profile
(GPU, VRAM, bf16 support) and run context (dtype, quantization, sizes).
Output is a human-readable table and JSON alongside the results.

Usage:
    timer = StageTimer(context={"dtype": "float16", "train_size": 5000})
    with timer.stage("dataset_collection"):
        ...
    with timer.stage("training"):
        ...
    print(timer.summary())
    timer.save_json("results/timing_report.json")
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch

__all__ = ["StageTimer"]


def _fmt_duration(seconds: float) -> str:
    """123.4 → '2m 03s'; 4521 → '1h 15m'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m"


class StageTimer:
    """Per-stage timing and peak-VRAM accounting for the pipeline.

    Args:
        context: arbitrary run parameters for the report
            (dtype, quantization, dataset sizes, epochs...).
    """

    def __init__(self, context: Optional[dict[str, Any]] = None):
        self.context: dict[str, Any] = dict(context or {})
        self.stages: list[dict[str, Any]] = []
        self.hardware = self._collect_hardware()
        self._t_created = time.time()

    @staticmethod
    def _collect_hardware() -> dict[str, Any]:
        hw: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw["gpu"] = props.name
            hw["vram_gb"] = round(props.total_memory / 1e9, 1)
            hw["cuda"] = torch.version.cuda
            # By compute capability, NOT is_bf16_supported(): the latter returns True
            # even on Turing (T4), where bf16 is emulated via fp32 and many times slower.
            hw["bf16_native"] = torch.cuda.get_device_capability(0)[0] >= 8
        return hw

    @contextmanager
    def stage(self, name: str):
        """Context manager for a single stage: time + peak VRAM."""
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            entry: dict[str, Any] = {
                "name": name,
                "seconds": round(time.time() - t0, 1),
            }
            if cuda:
                entry["peak_vram_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 2
                )
            self.stages.append(entry)

    # ============================================================
    # Reports
    # ============================================================

    @property
    def total_seconds(self) -> float:
        return round(sum(s["seconds"] for s in self.stages), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware,
            "context": self.context,
            "stages": self.stages,
            "total_seconds": self.total_seconds,
            "total_human": _fmt_duration(self.total_seconds),
        }

    def summary(self) -> str:
        """Human-readable table to print at the end of a run."""
        lines = ["", "=" * 62, "PIPELINE TIMING REPORT", "=" * 62]

        hw = self.hardware
        if hw.get("gpu"):
            bf16 = "bf16 native" if hw.get("bf16_native") else "NO bf16 (use fp16)"
            lines.append(f"GPU: {hw['gpu']} ({hw['vram_gb']} GB, {bf16})")
        else:
            lines.append("GPU: none (CPU run)")
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            lines.append(f"Run: {ctx}")

        lines.append("-" * 62)
        has_vram = any("peak_vram_gb" in s for s in self.stages)
        header = f"{'stage':<28}{'time':>10}"
        if has_vram:
            header += f"{'peak VRAM':>14}"
        lines.append(header)
        for s in self.stages:
            row = f"{s['name']:<28}{_fmt_duration(s['seconds']):>10}"
            if has_vram:
                vram = f"{s['peak_vram_gb']:.2f} GB" if "peak_vram_gb" in s else "-"
                row += f"{vram:>14}"
            lines.append(row)
        lines.append("-" * 62)
        lines.append(f"{'TOTAL':<28}{_fmt_duration(self.total_seconds):>10}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
