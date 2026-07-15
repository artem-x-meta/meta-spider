"""Shared CLI helpers: layer parsing, prompt format (chat_template gap), device-auto,
run.json manifest, status.json, grep markers for CC orchestration, dataset loader.

Goal — remove the duplication that used to be copy-pasted across the lab scripts: `_layers()`,
the manual Gemma format, sys.path/loader-path boilerplate, device selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

MANIFEST_NAME = "run.json"
STATUS_NAME = "status.json"
MANIFEST_VERSION = "1.0"


# ────────────────────────── layers / format / device ──────────────────────────

def parse_layers(s: Any) -> Any:
    """'late'/'all'/'late_slice' → as-is (resolved by the framework); '34,38'/[34,38] → list[int]."""
    if s in ("late", "all", "late_slice"):
        return s
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(",")]


def resolve_prompt(questions: list[str], prompt_format: str, has_chat_template: bool):
    """Return (wrapped_questions, apply_chat_template_flag).

    auto    — if the tokenizer has a chat_template, return raw questions (the collector applies
              the template); otherwise the manual gemma2 format.
    gemma2  — <start_of_turn>user…<end_of_turn> (base/Gemma-2/3).
    gemma4_direct — <|turn>user…<turn|> (Gemma-4-IT direct, without the thinking channel).
    """
    if prompt_format == "auto":
        if has_chat_template:
            return list(questions), True
        prompt_format = "gemma2"
    if prompt_format == "gemma2":
        return ([f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
                 for q in questions], False)
    if prompt_format == "gemma4_direct":
        return ([f"<|turn>user\n{q}<turn|>\n<|turn>model\n" for q in questions], False)
    raise ValueError(f"unknown prompt_format: {prompt_format!r}")


def auto_device(device: Optional[str]) -> str:
    """'auto'/None → cuda if available, otherwise cpu. Otherwise — as passed in."""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ────────────────────────── manifest / status / markers ──────────────────────────

def write_manifest(run_dir: str, manifest: dict) -> Path:
    rd = Path(run_dir); rd.mkdir(parents=True, exist_ok=True)
    p = rd / MANIFEST_NAME
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_manifest(run_dir: str) -> dict:
    p = Path(run_dir) / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"missing {p} — run `metaloom collect` first")
    return json.loads(p.read_text(encoding="utf-8"))


def write_status(run_dir: str, **fields) -> None:
    """Live status.json for CC monitors (merges fields)."""
    rd = Path(run_dir); rd.mkdir(parents=True, exist_ok=True)
    p = rd / STATUS_NAME
    cur: dict = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(fields)
    cur["updated"] = time.time()
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


def mark(kind: str, value: Any) -> None:
    """Grep marker for background CC monitors (no tail — line-buffered flush)."""
    print(f"{kind} {value}", flush=True)


def mark_artifact(path: Any) -> None:
    mark("ARTIFACT", str(Path(path).resolve()))


def mark_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            mark("PEAK_VRAM", f"{torch.cuda.max_memory_allocated() / 1e9:.2f}")
    except Exception:
        pass


# ────────────────────────── configs from the manifest ──────────────────────────

def build_meta_config(manifest: dict, *, device: Optional[str] = None,
                      gradient_checkpointing: bool = False,
                      max_memory: Optional[dict] = None):
    """MetaAttentionConfig from the manifest (device-auto). max_memory — Part B (offload)."""
    from meta_attention import MetaAttentionConfig
    kw = dict(
        model_name=manifest["model_name"],
        device=auto_device(device if device is not None else manifest.get("device")),
        dtype=manifest.get("dtype", "float16"),
        quantization=manifest.get("quantization"),
        gradient_checkpointing=gradient_checkpointing,
        target_layers=parse_layers(manifest["target_layers"]),
        cross_attn_layers=parse_layers(manifest["cross_attn_layers"]),
    )
    # F3: Gemma in fp16 → NaN logits (softcapping/norms overflow fp16) → silently empty output.
    # Force bf16 (native on Ampere+). Catches the common silent Gemma failure out of the box.
    if "gemma" in manifest["model_name"].lower() and kw["dtype"] in ("float16", "fp16", "half"):
        print(f"  [warn] {manifest['model_name']}: fp16 on Gemma produces NaN logits → "
              f"switching dtype to bfloat16", flush=True)
        kw["dtype"] = "bfloat16"
    # F7: nf4/int8 on a model that fits in bf16 — overhead with no benefit. bitsandbytes dequant
    # loads the GPU COMPUTE on every forward; on a Windows/WDDM laptop GPU (the display shares the
    # chip) that = UI stutter for negligible VRAM savings. Warn if the bf16 weights fit the card
    # with headroom.
    if kw.get("quantization") and str(kw["device"]).startswith("cuda") and max_memory is None:
        try:
            import torch
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(manifest["model_name"])
            tc = getattr(c, "text_config", None)
            g = lambda k, d=0: getattr(c, k, None) or (getattr(tc, k, None) if tc else None) or d
            h, nl, v = g("hidden_size"), g("num_hidden_layers"), g("vocab_size")
            inter = g("intermediate_size", 4 * h)
            params = v * h * 2 + nl * (4 * h * h + 3 * h * inter)  # rough estimate
            bf16_gb = params * 2 / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if params and bf16_gb * 1.6 < total_gb:   # fits with headroom for activations/cache
                print(f"  [warn] {manifest['model_name']} ~{params/1e9:.1f}B fits in bf16 "
                      f"(~{bf16_gb:.1f}GB) on GPU {total_gb:.0f}GB — {kw['quantization']} adds "
                      f"dequant overhead (on a Windows/WDDM laptop GPU = UI stutter) with no memory win. "
                      f"For models that fit, drop --quantization (keep --dtype bfloat16).", flush=True)
        except Exception:
            pass
    kw["max_memory"] = max_memory  # None → whole model on device; dict → offload (Part B)
    if max_memory is not None and manifest.get("quantization"):
        # bnb 4bit + CPU offload fails without this ("Some modules dispatched on CPU/disk").
        kw["cpu_offload_fp32"] = True
    if manifest.get("quantize_lm_head"):
        # nf4-lm_head for models with a huge vocabulary (Gemma 262K / Qwen3.5 248K) WITHOUT slice —
        # lm_head 2GB→0.5GB, fits in 4GB under standard training (outputs.loss dequants on its own).
        kw["quantize_lm_head"] = True
    if manifest.get("attn_implementation"):
        kw["attn_implementation"] = manifest["attn_implementation"]  # Gemma collect → "eager"
    return MetaAttentionConfig(**kw)


def parse_max_memory(s: Optional[str]) -> Optional[dict]:
    """'{"0":"3GiB","cpu":"14GiB"}' → {0:"3GiB","cpu":"14GiB"} (int keys for GPU indices)."""
    if not s:
        return None
    raw = json.loads(s)
    return {(int(k) if str(k).isdigit() else k): v for k, v in raw.items()}


# ────────────────────────── dataset loader ──────────────────────────

def resolve_dataset_loader(loader_path: Optional[str] = None):
    """(load_qa_dataset, check_answer_correctness, check_gsm8k_answer).

    By default — the package's NATIVE loader (`daimon_loom.data.dataset`): a random dev needs
    nothing external (fixes the F4 blocker — the loader used to live in publish/github outside
    the packages). `--loader-path` → your own external loader with the same API (module
    `src/utils/dataset.py`).
    """
    if loader_path is None:
        from daimon_loom.data.dataset import (
            check_answer_correctness, check_gsm8k_answer, load_qa_dataset,
        )
        return load_qa_dataset, check_answer_correctness, check_gsm8k_answer
    # external loader (custom data): a directory containing src/utils/dataset.py
    p = str(Path(loader_path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    from src.utils.dataset import (  # noqa: E402
        check_answer_correctness, check_gsm8k_answer, load_qa_dataset,
    )
    return load_qa_dataset, check_answer_correctness, check_gsm8k_answer
