"""Trainer — two-pass training loop for modifiers.

One training step = two forwards of the same model + backward on top:

    Pass 1 (read, cached):
      activation_list ← from ActivationDatasetCollector (pre-collected)
      cognitive_tokens ← modifier.encoder(activation_list)
      modifier.buffer.fill(cognitive_tokens)

    Pass 2 (write + loss):
      logits, loss ← base.forward(input_ids+target, labels=labels_with_prompt_mask)
                     # the modifier's CA hooks see buffer.is_filled and inject

    Backward:
      loss.backward()        # through the frozen base → CA → cog tokens → encoder
      optimizer.step()       # updates ONLY the wrapper

5 parameter groups (per the Phase 2 record):
  1. Encoder weights (no gates)  — base LR
  2. CA weights (no gates, no preferences) — base LR
  3. Encoder gates — ×5 LR
  4. CA gates — ×5 LR
  5. CA token preferences — ×5 LR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import random as _random

import torch
import torch.nn as nn
from torch.optim import AdamW

from meta_loom.training.collector import DatasetSample
from meta_loom.training.losses import (
    build_correction_target,
    make_labels_with_prompt_mask,
)

if TYPE_CHECKING:
    from meta_core.modifiers.doubter import Doubter
    from meta_core.pipeline import MetaSpiderPipeline

__all__ = ["Trainer", "TrainerConfig"]


@dataclass
class TrainerConfig:
    """Training hyperparameters (defaults from the Phase 2 Selective record on Llama-8B).

    Defaults validated on MMLU full (12042 train) — sel_acc 89.1% (base 64.6%):
      - lr 2e-4 (×5 for gates and token_preferences)
      - effective batch = 32 (batch=2, grad_accum=16)
      - epochs 10, early stop patience 5
      - cosine schedule, warmup 5%
      - AdamW + weight_decay 0.01
      - correction_ratio = 0.5 (Phase 2 self-correction)
    """

    epochs: int = 10
    batch_size: int = 2
    grad_accumulation: int = 16
    learning_rate: float = 2e-4
    gate_lr_multiplier: float = 5.0
    warmup_ratio: float = 0.05
    early_stop_patience: int = 5
    weight_decay: float = 0.01
    max_seq_len: int = 256
    grad_clip: float = 1.0

    correction_ratio: float = 0.5

    # SLICE TRAINER: if set (= config.slice_cut_layer()), Pass 2 runs ONLY the slice
    # layers[cut+1:] from the cached cut_hidden (dataset 1.3) + chunked_lm_loss, instead of a full
    # base forward. The bottom isn't computed → an 8-12B wrapper trains on 4GB. None = normal mode.
    slice_cut_layer: Optional[int] = None
    chunked_loss_chunks: int = 4
    # "adamw" (default) | "adam8bit" (bitsandbytes, 4× less optimizer memory — for
    # training an 8-12B wrapper on 4GB, otherwise fp32 Adam is fatter than the slice).
    optimizer: str = "adamw"

    pretrain_projectors: bool = True
    pretrain_epochs: int = 5
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 32

    # GradScaler against gradient underflow with an fp16 base (T4/P100 without bf16).
    # None = auto: enabled if the base is in fp16.
    use_grad_scaler: Optional[bool] = None

    seed: int = 42

    extras: dict[str, Any] = field(default_factory=dict)


# ============================================================
# ConfidenceHead — a temporary classifier for the pretrain projectors
# ============================================================


class _ConfidenceHead(nn.Module):
    """Small head encoder_dim → P(correct). Discarded after pretrain."""

    def __init__(self, bottleneck_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


# ============================================================
# Trainer
# ============================================================


class Trainer:
    """Two-pass trainer for the Doubter (or another CA-based modifier).

    Args:
        modifier: a Doubter already attached to the pipeline (via `pipeline.attach`).
        pipeline: MetaSpiderPipeline.
        config: TrainerConfig.
    """

    def __init__(
        self,
        modifier: "Doubter",
        pipeline: "MetaSpiderPipeline",
        config: Optional[TrainerConfig] = None,
    ):
        if modifier not in pipeline.modifiers:
            raise ValueError("Modifier must be attached to the pipeline before training.")
        self.modifier = modifier
        self.pipeline = pipeline
        self.config = config or TrainerConfig()
        self._rng = _random.Random(self.config.seed)

    # ============================================================
    # Optimizer + scheduler
    # ============================================================

    def build_optimizer(self) -> AdamW:
        """5-group AdamW (per the Phase 2 record)."""
        cfg = self.config
        mod = self.modifier

        encoder = mod.encoder
        if encoder is None or not isinstance(encoder, nn.Module):
            raise RuntimeError("Modifier.encoder is not initialized (call pipeline.attach).")

        # Group 1: encoder weights without gates
        encoder_weight_params: list = []
        encoder_gate_params: list = []
        for n, p in encoder.named_parameters():
            if "layer_gates" in n:
                encoder_gate_params.append(p)
            else:
                encoder_weight_params.append(p)

        # Groups 2, 4, 5: CA weights, gates, preferences
        ca_weight_params: list = []
        ca_gate_params: list = []
        ca_pref_params: list = []
        for ca in mod.ca_modules.values():
            for n, p in ca.named_parameters():
                if n == "gate":
                    ca_gate_params.append(p)
                elif n == "token_preference":
                    ca_pref_params.append(p)
                else:
                    ca_weight_params.append(p)

        gate_lr = cfg.learning_rate * cfg.gate_lr_multiplier

        param_groups = [
            {"params": encoder_weight_params, "lr": cfg.learning_rate},
            {"params": ca_weight_params, "lr": cfg.learning_rate},
        ]
        if encoder_gate_params:
            param_groups.append({"params": encoder_gate_params, "lr": gate_lr})
        if ca_gate_params:
            param_groups.append({"params": ca_gate_params, "lr": gate_lr})
        if ca_pref_params:
            param_groups.append({"params": ca_pref_params, "lr": gate_lr})

        # Adam8bit (bitsandbytes): 2 bytes/param instead of 8 (fp32 Adam). On 4GB for a
        # 100-200M wrapper that's ~0.4GB instead of ~1.6GB — otherwise the optimizer is fatter than the slice.
        if cfg.optimizer == "adam8bit":
            from bitsandbytes.optim import Adam8bit
            return Adam8bit(param_groups, weight_decay=cfg.weight_decay)
        return AdamW(param_groups, weight_decay=cfg.weight_decay)

    def build_scheduler(self, optimizer: AdamW, num_training_steps: int):
        """Cosine schedule with warmup. Lazy import transformers."""
        try:
            from transformers import get_cosine_schedule_with_warmup
        except ImportError as exc:
            raise ImportError(
                "The scheduler requires `transformers`. "
                "Install it: pip install transformers"
            ) from exc

        num_warmup = int(num_training_steps * self.config.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )

    # ============================================================
    # Pretrain projectors (per-layer BCE on P(correct))
    # ============================================================

    def pretrain_projectors(
        self,
        samples: list[DatasetSample],
        verbose: bool = True,
    ) -> dict[int, float]:
        """Separately pretrain each per-layer encoder projector on P(correct) BCE.

        Only for SelectiveEncoder (TransformerEncoder uses a different architecture).

        Returns:
            {layer_idx: probe_accuracy} — for logging.
        """
        from meta_core.encoders.selective import SelectiveEncoder

        cfg = self.config
        encoder = self.modifier.encoder
        if not isinstance(encoder, SelectiveEncoder):
            if verbose:
                print("  pretrain_projectors is only supported for SelectiveEncoder; skip.")
            return {}

        device = self._infer_device()
        results: dict[int, float] = {}

        labels = torch.tensor(
            [int(s.pass1_correct) for s in samples],
            dtype=torch.float32,
            device=device,
        )

        layer_indices = sorted(samples[0].activations.keys())
        n_samples = len(samples)
        bs = cfg.pretrain_batch_size

        for li, layer_idx in enumerate(layer_indices):
            if li >= len(encoder.layer_projectors):
                continue
            projector = encoder.layer_projectors[li].to(device)
            head = _ConfidenceHead(encoder.bottleneck_dim).to(device)

            params = list(projector.parameters()) + list(head.parameters())
            opt = AdamW(params, lr=cfg.pretrain_lr)
            bce = nn.BCEWithLogitsLoss()

            # Stack activations [N, hidden]; cast to float — projector weights are fp32,
            # activations may be bf16 (from a bf16 base), and LayerNorm does not reconcile dtypes.
            acts = torch.stack([s.activations[layer_idx].to(device).float() for s in samples])

            for epoch in range(cfg.pretrain_epochs):
                # Shuffle
                perm = torch.randperm(n_samples, device=device)
                acts_shuffled = acts[perm]
                labels_shuffled = labels[perm]
                for i in range(0, n_samples, bs):
                    x = acts_shuffled[i:i+bs]
                    y = labels_shuffled[i:i+bs]
                    z = projector(x)
                    logits = head(z)
                    loss = bce(logits, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Final accuracy
            with torch.no_grad():
                z = projector(acts)
                preds = (torch.sigmoid(head(z)) > 0.5).float()
                acc = (preds == labels).float().mean().item()
            results[layer_idx] = acc
            if verbose:
                print(f"  pretrain L{layer_idx}: acc={acc:.3f}", flush=True)

        if verbose and results:
            print(f"  Pretrain done — best layers: "
                  f"{sorted(results.items(), key=lambda x: -x[1])[:3]}", flush=True)
        return results

    # ============================================================
    # Main train loop
    # ============================================================

    def train(
        self,
        train_samples: list[DatasetSample],
        val_samples: Optional[list[DatasetSample]] = None,
        targets_by_sample: Optional[list[tuple[str, str]]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Main training loop with the two-pass structure.

        Args:
            train_samples: list of DatasetSample (from ActivationDatasetCollector).
            val_samples: optional, for validation loss + early stop.
            targets_by_sample: optional pre-built (target_text, action_type)
                per sample. If None — generated via
                `build_correction_target(s.ground_truth, s.pass1_correct, ...)`.
            checkpoint_dir: where to save best_model.pt. None = don't save.

        Returns:
            history: dict with train_loss / val_loss / per-epoch CA gate map.
        """
        cfg = self.config
        pipeline = self.pipeline
        modifier = self.modifier

        # Optional: pretrain projectors
        if cfg.pretrain_projectors and train_samples:
            self.pretrain_projectors(train_samples)

        # Optimizer + scheduler
        optimizer = self.build_optimizer()
        steps_per_epoch = max(1, len(train_samples) // cfg.batch_size // cfg.grad_accumulation)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = None
        try:
            scheduler = self.build_scheduler(optimizer, total_steps)
        except ImportError:
            pass  # no scheduler — acceptable in tests without transformers

        # Targets
        if targets_by_sample is None:
            targets_by_sample = [
                build_correction_target(
                    s.ground_truth, s.pass1_correct,
                    correction_ratio=cfg.correction_ratio,
                    rng=self._rng,
                )
                for s in train_samples
            ]

        # GradScaler: auto with an fp16 base, otherwise off (bf16/fp32 don't need it).
        # The first floating-point parameter = the compute dtype (skip quantized int weights).
        use_scaler = cfg.use_grad_scaler
        if use_scaler is None:
            base_dtype = next(
                (p.dtype for p in pipeline.model.parameters()
                 if p.dtype.is_floating_point),
                torch.float32,
            )
            use_scaler = base_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        if use_scaler:
            print("  GradScaler enabled (fp16 base)")

        # History
        history: dict[str, Any] = {
            "train_loss": [], "val_loss": [], "ca_gate_map": [], "epoch_seconds": [],
        }
        best_val_loss = float("inf")
        patience = 0

        import time as _time

        for epoch in range(cfg.epochs):
            t_epoch = _time.time()
            modifier.encoder.train()
            modifier.ca_modules.train()

            indices = list(range(len(train_samples)))
            self._rng.shuffle(indices)

            optimizer.zero_grad()
            total_loss = 0.0
            n_batches = 0
            opt_step = 0

            for batch_start in range(0, len(indices), cfg.batch_size):
                batch_idx = indices[batch_start:batch_start + cfg.batch_size]
                batch_samples = [train_samples[i] for i in batch_idx]
                batch_targets = [targets_by_sample[i] for i in batch_idx]

                loss = self._train_step(batch_samples, batch_targets)
                if loss is None:
                    continue

                scaler.scale(loss / cfg.grad_accumulation).backward()
                total_loss += loss.item()
                n_batches += 1

                if (n_batches % cfg.grad_accumulation) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        modifier.get_trainable_parameters(), cfg.grad_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    opt_step += 1

            train_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if val_samples is not None and len(val_samples) > 0:
                val_targets = [
                    build_correction_target(
                        s.ground_truth, s.pass1_correct,
                        correction_ratio=cfg.correction_ratio,
                        rng=self._rng,
                    )
                    for s in val_samples
                ]
                val_loss = self._validate(val_samples, val_targets)
                history["val_loss"].append(val_loss)

            # CA gate map snapshot
            history["ca_gate_map"].append(modifier.get_ca_gate_map())

            epoch_sec = _time.time() - t_epoch
            history["epoch_seconds"].append(round(epoch_sec, 1))

            print(f"Epoch {epoch+1}/{cfg.epochs}: train_loss={train_loss:.4f}"
                  + (f", val_loss={val_loss:.4f}" if val_loss is not None else "")
                  + f" [{epoch_sec/60:.1f} min]", flush=True)

            # Early stop + best checkpoint
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                if checkpoint_dir:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    modifier.save_checkpoint(str(Path(checkpoint_dir) / "best_model.pt"))
            elif val_loss is not None:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"  Early stop at epoch {epoch+1}", flush=True)
                    break

        history["best_val_loss"] = best_val_loss if best_val_loss < float("inf") else None
        return history

    # ============================================================
    # Internal: one train step (two-pass)
    # ============================================================

    def _slice_softcap(self) -> Optional[float]:
        """final_logit_softcapping from the base config (Gemma-4 has it in `text_config`, top-level None).
        Llama/Mistral → None. Cached."""
        if hasattr(self, "_softcap_cached"):
            return self._softcap_cached
        cfg = getattr(self.pipeline.model, "config", None)
        cap = getattr(cfg, "final_logit_softcapping", None)
        if cap is None and hasattr(cfg, "text_config"):
            cap = getattr(cfg.text_config, "final_logit_softcapping", None)
        self._softcap_cached = cap
        return cap

    def _collate_slice(self, samples: list[DatasetSample], device):
        """SLICE TRAINER: right-pad the cached input_ids_full/labels_full/cut_hidden to the batch max.

        cut_hidden is stored as fp16 (disk savings) → cast to the base compute dtype before
        injection. Right-pad: real tokens at the start → standard position_ids are correct,
        trailing pad is masked (labels=-100, attn=0) and doesn't affect the real ones (causal).
        """
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        base_dtype = dtype_map.get(self.pipeline.config.dtype, torch.bfloat16)
        pad_id = getattr(self.pipeline.tokenizer, "pad_token_id", 0) or 0
        if any(s.cut_hidden is None for s in samples):
            raise RuntimeError(
                "slice_cut_layer is set, but the dataset has no cut_hidden — re-collect with "
                "`metaloom collect --slice` (the dataset must be format 1.3).")
        maxlen = max(int(s.input_ids_full.shape[0]) for s in samples)
        H = samples[0].cut_hidden.shape[-1]
        B = len(samples)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        cut_hidden = torch.zeros((B, maxlen, H), dtype=base_dtype)
        for i, s in enumerate(samples):
            n = int(s.input_ids_full.shape[0])
            input_ids[i, :n] = s.input_ids_full
            labels[i, :n] = s.labels_full
            attn[i, :n] = 1
            cut_hidden[i, :n] = s.cut_hidden.to(base_dtype)
        return (input_ids.to(device), attn.to(device),
                labels.to(device), cut_hidden.to(device))

    def _train_step(
        self,
        samples: list[DatasetSample],
        targets: list[tuple[str, str]],
    ) -> Optional[torch.Tensor]:
        """One training step: Pass 1 (cached activations) → encoder → buffer → Pass 2.

        Returns:
            the loss tensor, or None if NaN.
        """
        modifier = self.modifier
        pipeline = self.pipeline
        tokenizer = pipeline.tokenizer
        device = self._infer_device()
        max_seq_len = self.config.max_seq_len
        cut = self.config.slice_cut_layer

        cut_hidden = None
        if cut is not None:
            # SLICE TRAINER: use the cached input_ids/labels/cut_hidden (do NOT re-tokenize —
            # consistency with what cut_hidden was captured on in the collector).
            input_ids, attention_mask, labels, cut_hidden = self._collate_slice(samples, device)
        else:
            # Prepare input_ids + labels (mask prompt)
            input_texts = [s.input_text for s in samples]
            target_texts = [t[0] for t in targets]
            full_texts = [a + b for a, b in zip(input_texts, target_texts)]
            encodings = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            input_ids = encodings.input_ids.to(device)
            attention_mask = (encodings.attention_mask.to(device)
                              if hasattr(encodings, "attention_mask") else None)
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

            # Labels with prompt masking
            labels = input_ids.clone()
            for i, inp_text in enumerate(input_texts):
                inp_ids = tokenizer(inp_text, return_tensors="pt").input_ids
                inp_len = min(inp_ids.shape[-1], input_ids.shape[-1])
                labels[i, :inp_len] = -100
            labels[labels == pad_id] = -100

        # Pass 1: cached activations → encoder → buffer
        target_layers = sorted(samples[0].activations.keys())
        activation_list = []
        for layer_idx in target_layers:
            layer_acts = torch.stack([s.activations[layer_idx].to(device).float()
                                      for s in samples])
            activation_list.append(layer_acts)

        modifier.buffer.clear()
        cognitive_tokens = modifier.encoder(activation_list)
        modifier.buffer.fill(cognitive_tokens)

        # Pass 2: forward with labels → outputs.loss (CA hooks inject)
        # First freeze the ActivationCollector — otherwise it would overwrite the buffer during the forward
        if pipeline.collector is not None:
            pipeline.collector.freeze()
        try:
            if cut is not None:
                # SLICE: forward only layers[cut+1:] from cut_hidden (CA hooks inject
                # on the slice layers) → chunked_lm_loss. The bottom isn't computed; backward — the wrapper.
                from meta_core.slice_forward import slice_last_hidden
                from meta_loom.training.losses import chunked_lm_loss
                slice_hidden = slice_last_hidden(
                    pipeline.model, input_ids, attention_mask, cut_hidden, cut)
                lm_head = pipeline.model.get_output_embeddings()
                # nf4-lm_head (Gemma on 4GB): pass the MODULE (Linear4bit dequants);
                # bf16-lm_head (Llama): .weight (F.linear). softcap for Gemma from config.
                is_4bit = (hasattr(getattr(lm_head, "weight", None), "quant_state")
                           or "4bit" in type(lm_head).__name__.lower())
                head_arg = lm_head if is_4bit else lm_head.weight
                loss = chunked_lm_loss(slice_hidden, head_arg, labels,
                                       n_chunks=self.config.chunked_loss_chunks,
                                       softcap=self._slice_softcap())
            else:
                outputs = pipeline.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        finally:
            if pipeline.collector is not None:
                pipeline.collector.unfreeze()
            modifier.buffer.clear()

        if torch.isnan(loss):
            return None
        return loss

    def _validate(
        self,
        samples: list[DatasetSample],
        targets: list[tuple[str, str]],
    ) -> float:
        """One pass over val without backward."""
        modifier = self.modifier
        modifier.encoder.eval()
        modifier.ca_modules.eval()

        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(samples), self.config.batch_size):
                batch_s = samples[i:i+self.config.batch_size]
                batch_t = targets[i:i+self.config.batch_size]
                loss = self._train_step(batch_s, batch_t)
                if loss is not None:
                    total += loss.item()
                    n += 1
        return total / max(n, 1)

    def _infer_device(self) -> torch.device:
        for p in self.pipeline.model.parameters():
            return p.device
        return torch.device("cpu")

    # ============================================================
    # Persistence (for resume training)
    # ============================================================

    def save_checkpoint(self, path: str, epoch: Optional[int] = None) -> None:
        """Save the modifier state_dict + the current epoch."""
        payload = {
            "format_version": "1.0",
            "epoch": epoch,
            "encoder_state": (self.modifier.encoder.state_dict()
                              if isinstance(self.modifier.encoder, nn.Module) else None),
            "ca_state": {k: m.state_dict() for k, m in self.modifier.ca_modules.items()},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        """Load the state_dict into the modifier (encoder + CA)."""
        state = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(self.modifier.encoder, nn.Module) and state.get("encoder_state"):
            self.modifier.encoder.load_state_dict(state["encoder_state"])
        for k, st in state.get("ca_state", {}).items():
            if k in self.modifier.ca_modules:
                self.modifier.ca_modules[k].load_state_dict(st)
