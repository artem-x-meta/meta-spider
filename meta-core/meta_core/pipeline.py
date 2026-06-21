"""MetaSpiderPipeline — the framework's main entry point.

A wrapper around a HuggingFace causal LM that modifiers (Doubter) attach to.
Manages the lifecycle of the ActivationCollector hooks, the two forward passes (read + write),
and inference with active modifiers.

Target UX:

    pipeline = MetaSpiderPipeline.from_pretrained(config)
    pipeline.attach(Doubter.from_checkpoint("..."))
    text = pipeline.generate("prompt", max_new_tokens=100)

Two-pass forward (per the Phase 1 Selective canon):
  Pass 1: base.forward(prompt) → ActivationCollector captures activations
                              → each modifier.on_post_forward(activations)
                              → its encoder fills its own ReflexionBuffer
  Pass 2: ActivationCollector.freeze() — do not overwrite the buffer
        → base.generate(prompt, ...) — the modifiers' CA hooks read the buffers and inject
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from meta_core.config import MetaSpiderConfig
from meta_core.hooks import ActivationCollector

if TYPE_CHECKING:
    from meta_core.modifiers.base import Modifier

__all__ = ["MetaSpiderPipeline"]


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    """Map "bfloat16" / "float16" / "float32" → torch.dtype."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype string: {dtype_str!r}")
    return mapping[dtype_str]


def _patch_bnb_params4bit_is_hf_initialized() -> None:
    """transformers ≥5.6, when loading some nf4 models (the Gemma-4 multimodal loads its
    sub-models RECURSIVELY), passes `_is_hf_initialized` into `Params4bit.__new__`, but bnb
    0.49.2 (the LATEST release, nothing newer) does not accept it → `TypeError: unexpected
    keyword argument`. We absorb the kwarg and set it as an attribute (the transformers
    behavior — marking the "parameter as loaded" — is preserved). Idempotent. Remove once a
    bnb > 0.49.2 with support is released.

    Verified on gemma-4-12b-it nf4+offload: load is ok, sanity generation is correct.
    """
    try:
        import bitsandbytes.nn as bnn
    except ImportError:
        return
    P = bnn.Params4bit
    if getattr(P, "_ishf_absorb_patched", False):
        return
    _orig_new = P.__new__

    def _new(cls, *a, **kw):
        flag = kw.pop("_is_hf_initialized", None)
        obj = _orig_new(cls, *a, **kw)
        if flag is not None:
            obj._is_hf_initialized = flag
        return obj

    P.__new__ = _new
    P._ishf_absorb_patched = True


def _build_quantization_config(config: Any, compute_dtype: torch.dtype) -> Any:
    """Build a BitsAndBytesConfig from the compression ladder in MetaSpiderConfig.

    int8 → LLM.int8 (~2× smaller than bf16); nf4 / fp4 → 4-bit (~4×), with optional
    double quantization. Only the base's Linear layers are quantized; embeddings,
    layernorms and our wrapper stay in the compute dtype.
    """
    _patch_bnb_params4bit_is_hf_initialized()  # bnb 0.49.2 ↔ transformers ≥5.6 nf4-load fix
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "quantization requires transformers with bitsandbytes support."
        ) from exc
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"quantization={config.quantization!r} requires bitsandbytes. "
            "Install it: pip install bitsandbytes"
        ) from exc

    # Force-quant lm_head (Gemma-4 on 4GB): an empty skip list overrides the auto-detect,
    # which keeps lm_head (tied) in bf16. cpu_offload_fp32 — for a device_map with a cpu part.
    extra: dict = {}
    if getattr(config, "quantize_lm_head", False):
        extra["llm_int8_skip_modules"] = []
    if getattr(config, "cpu_offload_fp32", False):
        extra["llm_int8_enable_fp32_cpu_offload"] = True

    if config.quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, **extra)
    if config.quantization in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization,
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
            **extra,
        )
    raise ValueError(
        f"Unknown quantization: {config.quantization!r}. "
        "Allowed: None, 'int8', 'nf4', 'fp4'."
    )


def _infer_num_layers(model: Any) -> int:
    """Auto-detect the number of transformer layers (incl. the multimodals' nested text_config)."""
    from meta_core.model_utils import infer_num_layers
    return infer_num_layers(model)


def _infer_hidden_dim(model: Any) -> int:
    """Auto-detect the hidden state dimensionality (incl. the multimodals' nested text_config)."""
    from meta_core.model_utils import infer_hidden_dim
    return infer_hidden_dim(model)


class MetaSpiderPipeline:
    """Wrap an HF causal LM + attached modifiers + two-pass forward.

    Attributes:
        config: MetaSpiderConfig.
        model: HF AutoModelForCausalLM (frozen base).
        tokenizer: HF AutoTokenizer.
        collector: ActivationCollector with hooks on the target_layers.
        modifiers: list of attached modifiers in registration order.
    """

    def __init__(
        self,
        config: MetaSpiderConfig,
        model: Any,
        tokenizer: Any,
        collector: Optional[ActivationCollector] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.collector = collector
        self.modifiers: list["Modifier"] = []

    @classmethod
    def from_pretrained(
        cls,
        config: MetaSpiderConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> "MetaSpiderPipeline":
        """Load the base from the HF Hub + auto-detect dimensions + freeze the weights.

        Args:
            config: MetaSpiderConfig with model_name.
            model: pre-loaded HF model (for tests / for an already loaded model).
                If None — loaded via `AutoModelForCausalLM.from_pretrained`.
            tokenizer: pre-loaded HF tokenizer (if None — `AutoTokenizer.from_pretrained`).

        Returns:
            MetaSpiderPipeline ready for `attach()`-ing modifiers.
        """
        # device "auto" → cuda if available, else cpu (wrapper training runs on cpu too).
        if config.device == "auto":
            import torch as _torch
            config.device = "cuda" if _torch.cuda.is_available() else "cpu"

        if model is None or tokenizer is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "`from_pretrained` requires `transformers`. "
                    "Install it via `pip install transformers` or pass "
                    "already-loaded `model` and `tokenizer` directly."
                ) from exc

            dtype = _dtype_from_string(config.dtype)
            if model is None:
                load_kw = dict(torch_dtype=dtype)
                if getattr(config, "attn_implementation", None):
                    load_kw["attn_implementation"] = config.attn_implementation
                if config.quantization is not None:
                    # Quantize on load (bnb). Already-quantized (pre-quant) models
                    # load with quantization=None — transformers picks up the baked-in config.
                    load_kw["quantization_config"] = _build_quantization_config(config, dtype)

                if config.device_map is not None:
                    # Explicit device_map (slice-offload: top→GPU, bottom→cpu). Without .to().
                    load_kw["device_map"] = config.device_map
                    if config.max_memory:
                        load_kw["max_memory"] = config.max_memory
                    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kw)
                elif config.max_memory:
                    # Auto-offload by budget: accelerate distributes layers GPU↔RAM.
                    # Works for both quantized and pre-quant (without .to()).
                    load_kw["device_map"] = "auto"
                    load_kw["max_memory"] = config.max_memory
                    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kw)
                elif config.quantization is not None:
                    # Quantized without offload: bnb distributes; .to() is forbidden.
                    load_kw["device_map"] = {"": config.device}
                    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kw)
                else:
                    # Non-quantized without offload → load and move to device.
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_name, **load_kw).to(config.device)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Standard pattern for Llama / Mistral: pad_token is unset → use EOS.
        # Needed for batch tokenization in the Trainer and for decoding generated outputs.
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

        # Auto-detect dimensions + fill the config defaults
        base_num_layers = _infer_num_layers(model)
        base_hidden_dim = _infer_hidden_dim(model)
        config.resolve_defaults(base_num_layers, base_hidden_dim)

        # Freeze the base (if required)
        if config.freeze_base_model:
            for p in model.parameters():
                p.requires_grad = False

        # Gradient checkpointing: activation memory ↓ at the cost of a recomputed forward
        # during backward. CA injection via forward hooks is compatible: the hook fires
        # on recompute too, and the result is deterministic (the same buffer).
        if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False  # incompatible with checkpointing
            # The base is frozen → without this, checkpointing won't propagate the gradient to
            # the wrapper through the frozen layers (no requires_grad at the graph's input).
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Init ActivationCollector
        target_layers = config.target_layers or list(range(config.num_layers))
        collector = ActivationCollector(model, target_layers=target_layers)

        return cls(config=config, model=model, tokenizer=tokenizer, collector=collector)

    # ============================================================
    # Attaching modifiers
    # ============================================================

    def attach(self, modifier: "Modifier") -> "MetaSpiderPipeline":
        """Attach a modifier. Returns self for chaining."""
        modifier.on_attach(self)
        self.modifiers.append(modifier)
        return self

    def detach(self, modifier: "Modifier") -> None:
        """Detach a modifier and remove its associated hooks."""
        modifier.on_detach()
        if modifier in self.modifiers:
            self.modifiers.remove(modifier)

    def detach_all(self) -> list["Modifier"]:
        """Detach all modifiers. Returns their snapshot for a possible re-attach."""
        snapshot = list(self.modifiers)
        for m in snapshot:
            m.on_detach()
        self.modifiers.clear()
        return snapshot

    # ============================================================
    # Two-pass forward
    # ============================================================

    def _run_pass1(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> None:
        """Pass 1: forward to collect activations + fill the modifiers' buffers."""
        if self.collector is None:
            raise RuntimeError("Pipeline without an ActivationCollector. Use from_pretrained.")

        # Each modifier clears its own buffer
        for m in self.modifiers:
            m.on_pre_forward()

        # Clear the ActivationCollector and enable recording
        self.collector.clear()
        self.collector.unfreeze()

        # Forward without generation
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Pass the activations to all modifiers — they fill their buffers
        activations = self.collector.get_snapshot()
        for m in self.modifiers:
            m.on_post_forward(activations)

    def _run_pass2(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Pass 2: model.generate with the modifiers' CA hooks active."""
        # Freeze the ActivationCollector — otherwise on each generation step it would
        # overwrite the buffers with the new (generated) token
        if self.collector is not None:
            self.collector.freeze()

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Unfreeze the collector for the next call. Do NOT clear the modifiers' buffers —
        # each modifier decides for itself in `on_pre_forward` (Doubter clears its buffer).
        if self.collector is not None:
            self.collector.unfreeze()

        return outputs

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        apply_chat_template: bool = True,
        dynamic_refresh: bool = False,
        refresh_threshold: float = 0.5,
        refresh_min_interval: int = 3,
        refresh_max_interval: int = 20,
        refresh_eos_latch: float = 0.0,
        **generate_kwargs: Any,
    ) -> str:
        """Two-pass inference with active modifiers.

        Args:
            prompt: the user text.
            max_new_tokens: token limit for Pass 2.
            apply_chat_template: apply `tokenizer.apply_chat_template` (for instruct
                models). False = feed the raw prompt.
            dynamic_refresh: if True — per-token adaptive refresh of the cognitive tokens
                via IntrospectionCache (for long reasoning / agentic chains, where the
                uncertainty evolves over time). Otherwise static injection
                (the prompt's cognitive tokens are frozen for the whole answer).
            refresh_threshold / refresh_min_interval / refresh_max_interval: IntrospectionCache
                parameters (see meta_core.dynamic). Default threshold 0.5.
            **generate_kwargs: forwarded to `model.generate` (do_sample, temperature, etc.).

        Returns:
            The decoded generated text (without the prompt part).
            Dynamic-refresh stats (if dynamic_refresh) — in `self.last_dynamic_stats`.
        """
        if self.tokenizer is None:
            raise RuntimeError("Pipeline without a tokenizer — cannot run generate.")

        # Tokenize
        if apply_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")

        device = self._infer_device()
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None

        # dynamic_refresh requires a modifier with encoder+buffer; without one (e.g.
        # a base run in BaselineComparison) — silently fall back to static.
        if dynamic_refresh and self._has_buffer_modifier():
            outputs = self._generate_dynamic(
                input_ids, attention_mask, max_new_tokens,
                refresh_threshold, refresh_min_interval, refresh_max_interval,
                eos_latch=refresh_eos_latch,
                **generate_kwargs,
            )
        else:
            # Pass 1: read activations → fill modifiers' buffers
            self._run_pass1(input_ids, attention_mask)
            # Pass 2: generate with CA-hooks active
            outputs = self._run_pass2(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **generate_kwargs,
            )

        # Decode only the generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _has_buffer_modifier(self) -> bool:
        return any(
            getattr(m, "buffer", None) is not None and getattr(m, "encoder", None) is not None
            for m in self.modifiers
        )

    def _buffer_modifier(self):
        """The first attached modifier with encoder + buffer (for dynamic refresh)."""
        for m in self.modifiers:
            if getattr(m, "buffer", None) is not None and getattr(m, "encoder", None) is not None:
                return m
        raise RuntimeError(
            "dynamic_refresh requires an attached modifier with encoder+buffer (Doubter)."
        )

    def _generate_dynamic(
        self, input_ids, attention_mask, max_new_tokens,
        threshold, min_interval, max_interval, eos_latch=0.0, **generate_kwargs,
    ):
        """Per-token generation with adaptive cognitive-token refresh — on a KV cache.

        The chunk/refresh logic is as in `phase3_dynamic_llama8b.per_token_model`, but WITHOUT
        re-prefilling the context on every chunk. We keep TWO KV caches of the base:
          • gen_pkv — injected generation (buffer filled → CA injection active),
            grows one token at a time (greedy), the prompt prefill is done once;
          • clean_pkv — clean "reading" of the context (buffer empty → no injection), needed
            for the activation snapshot when deciding on a refresh; advanced incrementally
            with only the NEW tokens.
        This removes O(n²) (each chunk = 2 full passes) → O(n): one prefill + one cached
        forward per token + cheap incremental snapshots.

        Greedy-only (do_sample is not supported on this path — all our evals run greedy).
        """
        from meta_core.dynamic import IntrospectionCache
        from transformers import DynamicCache

        mod = self._buffer_modifier()
        cache = IntrospectionCache(threshold, min_interval, max_interval)
        target_layers = sorted(self.config.target_layers)
        device = input_ids.device

        # The full set of stop tokens — as in model.generate (the static path). Just
        # tokenizer.eos_token_id is not enough: for Gemma-it the turn is closed by <end_of_turn>,
        # which lives in generation_config.eos_token_id. Without it the manual loop does not
        # stop and runs on to max_tokens (a garbage tail after the stop).
        stop_ids: set[int] = set()
        gc = getattr(self.model, "generation_config", None)
        if gc is not None and getattr(gc, "eos_token_id", None) is not None:
            e = gc.eos_token_id
            stop_ids.update(e if isinstance(e, (list, tuple)) else [e])
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            stop_ids.add(tok_eos)

        def snapshot(clean_pkv, new_ids):
            """Incremental clean forward of `new_ids` on top of clean_pkv (buffer empty →
            CA does not inject). Returns (last token's activations, last token's clean
            logits). The logits = the BASE's next-token distribution without injection
            — for the clean-EOS latch (the base wants to close, but injection interferes)."""
            mod.buffer.clear()
            self.collector.clear()
            self.collector.unfreeze()
            seen = clean_pkv.get_seq_length()
            cur = new_ids.shape[1]
            am = torch.ones((1, seen + cur), dtype=torch.long, device=device)
            pos = torch.arange(seen, seen + cur, device=device)
            with torch.no_grad():
                out = self.model(input_ids=new_ids, attention_mask=am,
                                 past_key_values=clean_pkv, use_cache=True,
                                 cache_position=pos)
            self.collector.freeze()
            snap = self.collector.get_snapshot()
            clean_last = _logits(out)[:, -1, :].detach()
            return [snap[L].float() for L in target_layers], clean_last

        def encode_fill(acts):
            cog = mod.encoder(acts)
            mod.buffer.fill(cog)
            return cog

        def _logits(out):
            # A real HF model → ModelOutput with .logits; FakeLM (test) → a bare hidden.
            return out.logits if hasattr(out, "logits") else out

        def gen_step(gen_pkv, tok):
            """One cached step of injected generation (buffer filled)."""
            seen = gen_pkv.get_seq_length()
            am = torch.ones((1, seen + 1), dtype=torch.long, device=device)
            pos = torch.arange(seen, seen + 1, device=device)
            with torch.no_grad():
                out = self.model(input_ids=tok, attention_mask=am,
                                 past_key_values=gen_pkv, use_cache=True,
                                 cache_position=pos)
            return _logits(out)[:, -1, :].argmax(-1, keepdim=True)

        # --- Initial introspection: clean prefill of the prompt → cognitive tokens ---
        clean_pkv = DynamicCache()
        acts, _ = snapshot(clean_pkv, input_ids)
        clean_len = input_ids.shape[1]
        prompt_cog = encode_fill(acts)        # buffer filled; remember the static version
        cache.store(acts, prompt_cog)

        # --- Injected prefill of the prompt (a separate cache) → first token ---
        self.collector.freeze()  # during injected passes snapshots are not needed
        gen_pkv = DynamicCache()
        am = torch.ones_like(input_ids)
        pos = torch.arange(0, input_ids.shape[1], device=device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=am,
                             past_key_values=gen_pkv, use_cache=True,
                             cache_position=pos)
        next_tok = _logits(out)[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([input_ids, next_tok], dim=1)
        cache.tick(1)
        tokens_left = max_new_tokens - 1
        stop_hit = next_tok.item() in stop_ids
        latched = False              # clean-EOS latch: refresh off, buffer = static
        latch_count = 0

        current_batch = min_interval
        while tokens_left > 0 and not stop_hit:
            steps = min(current_batch, tokens_left)
            for _ in range(steps):
                next_tok = gen_step(gen_pkv, next_tok)
                generated = torch.cat([generated, next_tok], dim=1)
                tokens_left -= 1
                cache.tick(1)
                if next_tok.item() in stop_ids:
                    stop_hit = True
                    break
            if stop_hit or tokens_left <= 0:
                break
            # Clean snapshot of the current context (increment clean_pkv with new tokens)
            new_ids = generated[:, clean_len:]
            cur_acts, clean_last = snapshot(clean_pkv, new_ids)
            clean_len = generated.shape[1]

            # clean-EOS latch: if the BASE (without injection) wants to close the turn, but
            # injection interferes (an uncertainty loop) — we freeze the cognitive tokens at
            # the prompt version (static) and stop refreshing. Latent and language-agnostic:
            # P(stop) is taken from the clean logits over the set of stop tokens.
            if not latched and stop_ids and eos_latch > 0:
                p_stop = torch.softmax(clean_last.float(), dim=-1)[0, list(stop_ids)].sum().item()
                if p_stop >= eos_latch:
                    latched = True
            if latched:
                latch_count += 1
                mod.buffer.fill(prompt_cog)        # static until the end
                current_batch = max_interval        # no more pointless refreshing
                continue

            if cache.should_refresh(cur_acts):
                cache.store(cur_acts, encode_fill(cur_acts))
                current_batch = min_interval
            else:
                mod.buffer.fill(cache.last_cognitive_tokens)
                current_batch = min(current_batch + min_interval, max_interval)

        stats = cache.get_stats()
        stats["latched"] = latched
        stats["latch_steps"] = latch_count
        self.last_dynamic_stats = stats
        mod.buffer.clear()
        return generated

    def _infer_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    # ============================================================
    # Training (delegated to the Trainer)
    # ============================================================

    def train(self, modifier: "Modifier", dataset: Any, **kwargs: Any) -> dict:
        """Train the modifier on user data.

        Delegates to `meta_loom.training.trainer.Trainer`. The base stays frozen.
        Implementation is ported in Level 3.
        """
        raise NotImplementedError("Trainer port — Level 3")
