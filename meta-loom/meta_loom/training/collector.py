"""ActivationDatasetCollector — collects base-model activations for training.

Runs the dataset through the frozen base once and caches:
  - input_text (question) + ground_truth (correct answer)
  - hidden states [n_layers, hidden_dim] at target_layers
  - pass1_text — what the model produced without modifiers (Pass 1 baseline)
  - pass1_correct — flag for whether the model got it right on its own

Saved to .pt — the main training works on the activations directly,
without recomputing the base forward each time (60-70 min GPU on 12k MMLU → seconds from cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

if TYPE_CHECKING:
    from meta_core.pipeline import MetaSpiderPipeline

__all__ = ["ActivationDatasetCollector", "DatasetSample"]


@dataclass
class DatasetSample:
    """One dataset example.

    Attributes:
        input_text: prompt text (already with the chat template if needed).
        ground_truth: canonical correct answer (for building the target).
        activations: dict[layer_idx → tensor[hidden_dim]] — Pass 1 snapshot.
        pass1_text: what the model produced without modifiers.
        pass1_correct: whether it got it right (for building the correction target).
        aliases: alternative phrasings of the answer (TriviaQA-style) — for
            correctness checking; the target is built from ground_truth.
        cut_hidden: [seq, H] — full-seq output of the cut layer for prompt+target (SLICE TRAINER,
            format 1.3). None in normal mode. The cached input of the slice `layers[cut+1:]`.
        input_ids_full: [seq] — prompt+target tokens (deterministic target at
            correction_ratio=0). The Trainer uses them and does NOT re-tokenize (consistent with the cache).
        labels_full: [seq] — labels with the prompt masked (for the slice chunked_lm_loss).
    """

    input_text: str
    ground_truth: str
    activations: dict[int, torch.Tensor]
    pass1_text: str = ""
    pass1_correct: bool = False
    aliases: Optional[list[str]] = None
    cut_hidden: Optional[torch.Tensor] = None
    input_ids_full: Optional[torch.Tensor] = None
    labels_full: Optional[torch.Tensor] = None


class ActivationDatasetCollector:
    """Collect activations + Pass 1 answers for training the modifier.

    Args:
        pipeline: MetaSpiderPipeline WITHOUT attached modifiers (clean base).
        max_new_tokens: limit for Pass 1 generation (for the correctness check).
        apply_chat_template: apply the chat template to the questions.
        check_correctness: callable (pred_text, truths) → bool, where truths is a
            list[str] of aliases. If None — normalized substring against any alias.
        batch_size: collection batch size (left-padding); 1 = one at a time.

    Note:
        `collect()` accepts ground_truths as list[str] (one answer per question)
        or list[list[str]] (aliases, TriviaQA-style). The correctness check runs
        against all aliases; the first (canonical) one goes into DatasetSample.ground_truth.
    """

    def __init__(
        self,
        pipeline: "MetaSpiderPipeline",
        max_new_tokens: int = 50,
        apply_chat_template: bool = True,
        check_correctness: Optional[Callable[[str, str], bool]] = None,
        batch_size: int = 8,
        chat_template_kwargs: Optional[dict] = None,
    ):
        if pipeline.collector is None:
            raise ValueError("Pipeline without an ActivationCollector. Use from_pretrained.")
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.apply_chat_template = apply_chat_template
        self.check_correctness = check_correctness or self._default_check
        self.batch_size = max(1, batch_size)
        # extra kwargs for apply_chat_template — e.g. {"enable_thinking": False} for thinking models
        # (Qwen3.5 / Gemma-it): without it the template opens <think> and a 16-token pass1 never
        # reaches the answer → pass1_correct=0 → the Doubter collapses into permanent refusal.
        self.chat_template_kwargs = chat_template_kwargs or {}

    @staticmethod
    def _default_check(pred: str, truths: list[str]) -> bool:
        """Default checker: normalized substring against any alias.

        Normalization (lower + punctuation removal) is critical: chat models
        wrap answers in markdown (**Karl Marx**), breaking a naive substring.
        """
        import re

        def norm(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"[^\w\s]", "", s)
            return re.sub(r"\s+", " ", s).strip()

        pred_n = norm(pred)
        if not pred_n:
            return False
        for truth in truths:
            t = norm(truth)
            if t and t in pred_n:
                return True
        return False

    def collect(
        self,
        questions: list[str],
        ground_truths: list[str],
        verbose: bool = True,
        slice_cut_layer: Optional[int] = None,
        correction_ratio: float = 0.0,
    ) -> list[DatasetSample]:
        """Run every question through the base: capture activations + generate the Pass 1 answer.

        Args:
            questions: list of texts.
            ground_truths: the corresponding correct answers.
            verbose: print progress.
            slice_cut_layer: if set (SLICE TRAINER) — additionally capture the full-seq
                cut_hidden at this layer for prompt+target + cache input_ids/labels.
            correction_ratio: for building the deterministic target (slice mode);
                0.0 = refuse-only (Phase-1R), as in run_gemma12b.

        Returns:
            list[DatasetSample] — ready to feed into the Trainer.
        """
        assert len(questions) == len(ground_truths), "Length mismatch"

        # Normalize ground_truths to alias lists:
        # list[str] → list[list[str]] (one alias); list[list[str]] — as-is.
        alias_lists: list[list[str]] = [
            gt if isinstance(gt, (list, tuple)) else [gt] for gt in ground_truths
        ]

        pipeline = self.pipeline
        tokenizer = pipeline.tokenizer
        collector = pipeline.collector
        device = self._infer_device()

        samples: list[DatasetSample] = []
        bs = self.batch_size

        for batch_start in range(0, len(questions), bs):
            if verbose and (batch_start // bs) % 10 == 0:
                print(f"  Collecting {batch_start}/{len(questions)}...")

            batch_q = questions[batch_start:batch_start + bs]
            batch_truth = alias_lists[batch_start:batch_start + bs]

            # Prepare inputs
            texts = []
            for q in batch_q:
                if self.apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": q}]
                    texts.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        **self.chat_template_kwargs,
                    ))
                else:
                    texts.append(q)

            # LEFT padding is mandatory: the hook captures hidden[:, -1, :] — with left
            # padding, position -1 of every batch row = its real last token.
            old_side = getattr(tokenizer, "padding_side", "right")
            tokenizer.padding_side = "left"
            try:
                inputs = tokenizer(
                    texts, return_tensors="pt", padding=len(texts) > 1,
                )
            finally:
                tokenizer.padding_side = old_side
            input_ids = inputs.input_ids.to(device)
            attention_mask = (inputs.attention_mask.to(device)
                              if hasattr(inputs, "attention_mask") else None)

            # position_ids from the mask: with left padding, a bare HF forward does NOT recompute
            # positions on its own (unlike generate) — without this RoPE is shifted by the padding.
            position_ids = None
            if attention_mask is not None and input_ids.shape[0] > 1:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

            # Pass 1: forward + capture activations (whole batch)
            collector.clear()
            collector.unfreeze()
            fwd_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if position_ids is not None:
                fwd_kwargs["position_ids"] = position_ids
            with torch.no_grad():
                pipeline.model(**fwd_kwargs)
            snap = collector.get_snapshot()  # {layer: [B, hidden]}

            # Pass 1 baseline generation in a batch (to determine correctness)
            collector.freeze()
            with torch.no_grad():
                outputs = pipeline.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            collector.unfreeze()

            prompt_len = input_ids.shape[1]
            for j, (text, truth_aliases) in enumerate(zip(texts, batch_truth)):
                generated_ids = outputs[j][prompt_len:]
                pass1_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                pass1_correct = self.check_correctness(pass1_text, list(truth_aliases))

                # Per-sample shape [hidden_dim] (1D): when the Trainer stacks N samples
                # it gets [N, hidden] (2D), not [N, 1, hidden].
                activations = {
                    idx: t[j].detach().cpu().clone() for idx, t in snap.items()
                }

                samples.append(DatasetSample(
                    input_text=text,
                    ground_truth=truth_aliases[0],
                    activations=activations,
                    pass1_text=pass1_text,
                    pass1_correct=pass1_correct,
                    aliases=list(truth_aliases),
                ))

            # SLICE TRAINER (format 1.3): extra forward prompt+target → full-seq cut_hidden
            if slice_cut_layer is not None:
                self._capture_cut_hidden_batch(
                    samples[-len(texts):], slice_cut_layer, correction_ratio, device)

        if verbose:
            n_correct = sum(1 for s in samples if s.pass1_correct)
            print(f"  Done: {len(samples)} samples ({n_correct} pass1_correct = "
                  f"{100*n_correct/len(samples):.1f}%)")
        return samples

    def _capture_cut_hidden_batch(
        self, batch_samples: list[DatasetSample], cut_layer: int,
        correction_ratio: float, device: torch.device,
    ) -> None:
        """SLICE TRAINER: capture full-seq cut_hidden + cache input_ids/labels.

        A separate forward over prompt+target (deterministic target). RIGHT-padding (real
        tokens at the start → standard position_ids are correct; trailing pad is masked),
        per-sample trim to the real length. cut_hidden is the input of the slice `layers[cut+1:]`.
        """
        import random as _random

        from meta_core.slice_forward import capture_cut_hidden
        from meta_loom.training.losses import build_correction_target

        tokenizer = self.pipeline.tokenizer
        rng = _random.Random(0)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        full_texts, prompt_lens = [], []
        for s in batch_samples:
            target_text, _ = build_correction_target(
                s.ground_truth, s.pass1_correct, correction_ratio, rng)
            full_texts.append(s.input_text + target_text)
            prompt_lens.append(tokenizer(s.input_text, return_tensors="pt").input_ids.shape[-1])

        old_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "right"
        try:
            enc = tokenizer(full_texts, return_tensors="pt", padding=len(full_texts) > 1)
        finally:
            tokenizer.padding_side = old_side
        input_ids = enc.input_ids.to(device)
        attention_mask = (enc.attention_mask.to(device)
                          if hasattr(enc, "attention_mask") else None)

        cut_hidden = capture_cut_hidden(self.pipeline.model, input_ids, attention_mask, cut_layer)
        for j, s in enumerate(batch_samples):
            real_len = (int(attention_mask[j].sum().item())
                        if attention_mask is not None else input_ids.shape[1])
            ids = input_ids[j, :real_len].detach().cpu().clone()
            labels = ids.clone()
            labels[:min(prompt_lens[j], real_len)] = -100  # prompt mask
            labels[ids == pad_id] = -100
            s.cut_hidden = cut_hidden[j, :real_len].detach().cpu().to(torch.float16).clone()
            s.input_ids_full = ids
            s.labels_full = labels

    def _infer_device(self) -> torch.device:
        for p in self.pipeline.model.parameters():
            return p.device
        return torch.device("cpu")

    # ============================================================
    # Persistence
    # ============================================================

    @staticmethod
    def save(samples: list[DatasetSample], path: str,
             config: Optional[dict] = None) -> None:
        """Save to .pt. config (the run manifest) — makes the artifact self-describing."""
        has_slice = any(s.cut_hidden is not None for s in samples)
        payload = {
            "format_version": "1.3" if has_slice else "1.2",
            "config": config,  # None in format ≤1.1 — backward compat
            "samples": [
                {
                    "input_text": s.input_text,
                    "ground_truth": s.ground_truth,
                    "activations": s.activations,
                    "pass1_text": s.pass1_text,
                    "pass1_correct": s.pass1_correct,
                    "aliases": s.aliases,
                    # SLICE TRAINER (1.3); None in ≤1.2 — backward compat
                    "cut_hidden": s.cut_hidden,
                    "input_ids_full": s.input_ids_full,
                    "labels_full": s.labels_full,
                }
                for s in samples
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load_config(path: str) -> Optional[dict]:
        """Read the embedded config (run manifest) from dataset.pt; None if absent."""
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return payload.get("config")

    @staticmethod
    def load(path: str) -> list[DatasetSample]:
        """Load a saved dataset."""
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if "format_version" not in payload:
            raise ValueError("Checkpoint without 'format_version' — possibly a legacy format.")
        return [
            DatasetSample(
                input_text=d["input_text"],
                ground_truth=d["ground_truth"],
                activations=d["activations"],
                pass1_text=d.get("pass1_text", ""),
                pass1_correct=d.get("pass1_correct", False),
                aliases=d.get("aliases"),  # None in format 1.0 — backward compat
                cut_hidden=d.get("cut_hidden"),  # None in ≤1.2 — backward compat
                input_ids_full=d.get("input_ids_full"),
                labels_full=d.get("labels_full"),
            )
            for d in payload["samples"]
        ]
