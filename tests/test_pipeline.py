"""Level 2 smoke tests: Pipeline + Doubter + Config.resolve_defaults.

Use FakeLM + FakeTokenizer from conftest.py — without HF transformers.
"""

import pytest
import torch

from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_voices import Doubter, DoubterConfig


# ============================================================
# MetaAttentionConfig.resolve_defaults
# ============================================================


def test_config_resolve_defaults_auto_fills():
    """resolve_defaults fills None fields from the base model's dimensions."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    assert cfg.hidden_dim is None
    assert cfg.num_layers is None
    assert cfg.target_layers is None
    assert cfg.cross_attn_layers is None

    cfg.resolve_defaults(base_num_layers=4, base_hidden_dim=64)
    assert cfg.hidden_dim == 64
    assert cfg.num_layers == 4
    assert cfg.target_layers == [0, 1, 2, 3]
    assert cfg.cross_attn_layers == [0, 1, 2, 3]


def test_config_resolve_defaults_preserves_user_set():
    """If the user set target_layers — resolve_defaults does not overwrite it."""
    cfg = MetaAttentionConfig(
        model_name="fake/model", device="cpu",
        target_layers=[1, 3],
    )
    cfg.resolve_defaults(base_num_layers=4, base_hidden_dim=64)
    assert cfg.target_layers == [1, 3]
    assert cfg.cross_attn_layers == [1, 3]


def test_config_resolve_defaults_hidden_mismatch_raises():
    """A provided hidden_dim ≠ base → ValueError."""
    cfg = MetaAttentionConfig(model_name="fake/model", hidden_dim=128, device="cpu")
    with pytest.raises(ValueError, match="hidden_dim"):
        cfg.resolve_defaults(base_num_layers=4, base_hidden_dim=64)


# ============================================================
# Quantization config (compression ladder) + gradient checkpointing
# ============================================================


def test_config_quantization_defaults():
    """By default compression is off, checkpointing is off."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    assert cfg.quantization is None
    assert cfg.double_quant is True
    assert cfg.gradient_checkpointing is False


def test_build_quantization_config_unknown_raises():
    """An unknown compression level → ValueError (if bnb is installed) or
    ImportError with a clear message (if not)."""
    from meta_attention.pipeline import _build_quantization_config

    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    cfg.quantization = "int3"  # no such level exists
    with pytest.raises((ValueError, ImportError)):
        _build_quantization_config(cfg, torch.float16)


def test_gradient_checkpointing_flag_applied(fake_lm, fake_tokenizer):
    """The gradient_checkpointing flag calls enable on the model (if it supports it)."""
    calls = []
    fake_lm.gradient_checkpointing_enable = lambda *a, **k: calls.append(1)

    cfg = MetaAttentionConfig(
        model_name="fake/model", device="cpu", gradient_checkpointing=True,
    )
    MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    assert calls, "gradient_checkpointing_enable was not called"


def test_infer_dtype_skips_quantized_params(fake_lm, fake_tokenizer):
    """_infer_dtype returns the first floating dtype, skipping int parameters
    (imitating bnb-quantized Linear weights)."""
    import torch.nn as nn

    class IntFirstModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.fake_quant = nn.Parameter(
                torch.zeros(4, dtype=torch.uint8), requires_grad=False,
            )
            self.inner = inner

        def parameters(self, recurse=True):
            yield self.fake_quant
            yield from self.inner.parameters(recurse)

    wrapped = IntFirstModel(fake_lm)
    assert Doubter._infer_dtype(wrapped) == torch.float32


# ============================================================
# Pipeline.from_pretrained (with an already-loaded FakeLM)
# ============================================================


def test_pipeline_from_pretrained_with_injected_model(fake_lm, fake_tokenizer):
    """from_pretrained(model=fake_lm) bypasses HF, correctly sets up collector + freeze."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)

    assert pipeline.config.hidden_dim == 64
    assert pipeline.config.num_layers == 4
    assert pipeline.config.target_layers == [0, 1, 2, 3]
    assert pipeline.collector is not None
    # Base is frozen
    assert all(not p.requires_grad for p in fake_lm.parameters())


# ============================================================
# Doubter attach / detach
# ============================================================


def test_doubter_on_attach_creates_obvyazka(fake_lm, fake_tokenizer):
    """on_attach creates encoder + ca_modules + buffer + registers CA hooks."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)

    doubter_cfg = DoubterConfig(
        encoder_type="selective",
        encoder_bottleneck=16,
        ca_bottleneck_dim=16,
        ca_num_heads=2,
        num_cognitive_tokens=4,
    )
    doubter = Doubter(doubter_cfg)
    pipeline.attach(doubter)

    assert doubter.encoder is not None
    assert len(doubter.ca_modules) == 4  # one per cross_attn_layer
    assert doubter.buffer is not None
    assert doubter.buffer.num_slots == 4
    assert len(doubter._ca_hook_handles) == 4


def test_doubter_detach_removes_hooks(fake_lm, fake_tokenizer):
    """detach removes the CA hooks."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)

    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    pipeline.detach(doubter)
    assert len(doubter._ca_hook_handles) == 0
    assert doubter not in pipeline.injectors


def test_doubter_get_trainable_parameters(fake_lm, fake_tokenizer):
    """get_trainable_parameters returns encoder + 4 CA modules."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    params = doubter.get_trainable_parameters()
    assert len(params) > 0
    n_params = sum(p.numel() for p in params)
    # encoder + 4 CA modules — should be > 10k parameters
    assert n_params > 10_000, f"unexpected param count {n_params}"


# ============================================================
# Two-pass forward via Pipeline._run_pass1 / _run_pass2
# ============================================================


def test_pipeline_pass1_fills_doubter_buffer(fake_lm, fake_tokenizer):
    """Pass 1 → ActivationCollector captures activations → Doubter.on_post_forward → buffer.is_filled."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)

    # input_ids: batch=1, length=5
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    assert doubter.buffer.is_filled is False

    pipeline._run_pass1(input_ids)
    assert doubter.buffer.is_filled is True
    assert doubter.buffer.slots.shape == (1, 4, 64)


def test_pipeline_full_generate(fake_lm, fake_tokenizer):
    """Full generate() → returns decoded text."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)

    result = pipeline.generate("hello", max_new_tokens=3)
    # FakeTokenizer.decode returns "<N generated tokens>" — the point is that it didn't crash
    assert isinstance(result, str)
    assert "generated tokens" in result
    # The buffer persists after generate (clear is done by the Doubter in on_pre_forward).
    # We verify that a repeat generate clears and recreates it correctly:
    assert doubter.buffer.is_filled is True
    pipeline.generate("again", max_new_tokens=2)
    assert doubter.buffer.is_filled is True  # filled again after the second Pass 1


def test_pipeline_detach_all_clean(fake_lm, fake_tokenizer):
    """detach_all disables all voices and returns a snapshot."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    d1 = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(d1)
    assert len(pipeline.injectors) == 1

    snapshot = pipeline.detach_all()
    assert len(snapshot) == 1
    assert len(pipeline.injectors) == 0
    assert len(d1._ca_hook_handles) == 0


# ============================================================
# Checkpoint save/load round-trip
# ============================================================


def test_doubter_checkpoint_round_trip(tmp_path, fake_lm, fake_tokenizer):
    """save_checkpoint → from_checkpoint → on_attach → weights restored."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline_a = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter_cfg = DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    )
    doubter_a = Doubter(doubter_cfg)
    pipeline_a.attach(doubter_a)

    # Save
    path = tmp_path / "doubter.pt"
    doubter_a.save_checkpoint(str(path))
    assert path.exists()

    # Load + attach into a new pipeline
    cfg_b = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    fake_lm_b = type(fake_lm)(hidden_dim=64, num_layers=4)
    pipeline_b = MetaAttentionPipeline.from_pretrained(cfg_b, model=fake_lm_b, tokenizer=fake_tokenizer)
    doubter_b = Doubter.from_checkpoint(str(path))
    pipeline_b.attach(doubter_b)

    # Encoder weights should match
    for k_a, v_a in doubter_a.encoder.state_dict().items():
        v_b = doubter_b.encoder.state_dict()[k_a]
        assert torch.allclose(v_a, v_b), f"encoder mismatch on {k_a}"
    # CA gates too
    for k_a in doubter_a.ca_modules.keys():
        ca_a = doubter_a.ca_modules[k_a]
        ca_b = doubter_b.ca_modules[k_a]
        assert torch.allclose(ca_a.gate, ca_b.gate), f"CA gate mismatch on {k_a}"


# ============================================================
# Encoder types
# ============================================================


def test_doubter_with_transformer_encoder(fake_lm, fake_tokenizer):
    """encoder_type='transformer' creates a TransformerEncoder."""
    from meta_attention import TransformerEncoder

    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="transformer",
        transformer_encoder_dim=32,
        transformer_num_blocks=1,
        transformer_num_heads=2,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
    ))
    pipeline.attach(doubter)
    assert isinstance(doubter.encoder, TransformerEncoder)


# ============================================================
# Diagnostics
# ============================================================


def test_doubter_ca_gate_map_after_attach(fake_lm, fake_tokenizer):
    """get_ca_gate_map returns the gate value for each CA layer (init ≈ tanh(0.3))."""
    import math

    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipeline = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(
        encoder_type="selective", encoder_bottleneck=16,
        ca_bottleneck_dim=16, ca_num_heads=2, num_cognitive_tokens=4,
        ca_gate_init=0.3,
    ))
    pipeline.attach(doubter)
    gate_map = doubter.get_ca_gate_map()
    assert set(gate_map.keys()) == {0, 1, 2, 3}
    expected = math.tanh(0.3)
    for v in gate_map.values():
        assert abs(v - expected) < 1e-5


# ============================================================
# Re-attach preserves trained weights (canary v9 bug)
# ============================================================


def test_reattach_preserves_trained_weights(fake_lm, fake_tokenizer):
    """detach → attach (the BaselineComparison cycle) must NOT recreate the wrapper.

    Canary v9: on_attach unconditionally recreated encoder+CA → the modified run
    used random weights → Δ=0.0000 across all metrics.
    """
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    pipe = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(encoder_type="selective"))
    pipe.attach(doubter)

    # "Training": mark the weights with a recognizable value
    ca = doubter.ca_modules["0"]
    with torch.no_grad():
        ca.gate.fill_(0.777)
    encoder_param = next(doubter.encoder.parameters())
    with torch.no_grad():
        encoder_param.fill_(0.123)

    # BaselineComparison cycle: detach → re-attach
    doubter.on_detach()
    pipe.injectors.remove(doubter)
    doubter.on_attach(pipe)
    pipe.injectors.append(doubter)

    assert doubter.ca_modules["0"].gate.item() == pytest.approx(0.777), \
        "CA weights recreated on re-attach!"
    assert next(doubter.encoder.parameters()).flatten()[0].item() == pytest.approx(0.123), \
        "Encoder weights recreated on re-attach!"
    # Hooks restored and not duplicated
    assert len(doubter._ca_hook_handles) == len(doubter.ca_modules)


def test_baseline_comparison_uses_trained_weights(fake_lm, fake_tokenizer):
    """Full BaselineComparison.run: afterwards the voice weights are the same."""
    from daimon_loom import BaselineComparison, BenchmarkTask, QABenchmark

    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu")
    pipe = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(encoder_type="selective"))
    pipe.attach(doubter)

    with torch.no_grad():
        doubter.ca_modules["0"].gate.fill_(0.555)

    tasks = [
        BenchmarkTask(task_id=f"t{i}", prompt=f"question {i}",
                      expected_answer="x", check_fn=lambda pred: True)
        for i in range(2)
    ]
    bench = QABenchmark(name="t", tasks=tasks, scoring="custom")
    report = BaselineComparison(pipe, bench, max_tokens=3).run(verbose=False)

    assert doubter.ca_modules["0"].gate.item() == pytest.approx(0.555), \
        "BaselineComparison clobbered the trained weights!"
    assert len(report.modified_results) == 2


# ============================================================
# Layer presets: "all" / "late"
# ============================================================


def test_layer_preset_all_and_late():
    """The 'all'/'late' presets resolve from the model depth."""
    cfg = MetaAttentionConfig(
        model_name="fake/model", device="cpu",
        target_layers="all", cross_attn_layers="late",
    )
    cfg.resolve_defaults(base_num_layers=26, base_hidden_dim=64)
    assert cfg.target_layers == list(range(26))
    assert cfg.cross_attn_layers == list(range(17, 26))  # top third of 26 layers

    cfg32 = MetaAttentionConfig(
        model_name="fake/model", device="cpu", cross_attn_layers="late",
    )
    cfg32.resolve_defaults(base_num_layers=32, base_hidden_dim=64)
    assert cfg32.cross_attn_layers == list(range(21, 32))


def test_layer_preset_unknown_raises():
    cfg = MetaAttentionConfig(
        model_name="fake/model", device="cpu", cross_attn_layers="early",
    )
    with pytest.raises(ValueError, match="early"):
        cfg.resolve_defaults(base_num_layers=8, base_hidden_dim=64)


def test_layer_preset_late_pipeline_attach(fake_lm, fake_tokenizer):
    """Doubter on the 'late' preset attaches CA only to the late layers."""
    cfg = MetaAttentionConfig(
        model_name="fake/model", device="cpu", cross_attn_layers="late",
    )
    pipe = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    doubter = Doubter(DoubterConfig(encoder_type="selective"))
    pipe.attach(doubter)
    # FakeLM: 4 layers → late = top third = [2, 3]
    assert sorted(int(k) for k in doubter.ca_modules.keys()) == [2, 3]


# ============================================================
# Dynamic refresh (IntrospectionCache) — Phase 3 port
# ============================================================


def test_introspection_cache_refresh_logic():
    """IntrospectionCache: first call refresh, min/max interval, threshold."""
    import torch
    from meta_attention import IntrospectionCache

    c = IntrospectionCache(threshold=0.5, min_interval=3, max_interval=10)
    a = [torch.ones(4)]
    assert c.should_refresh(a) is True          # no cache → refresh
    c.store(a, torch.zeros(1, 2, 4))
    c.tick(1)
    assert c.should_refresh(a) is False          # tokens_since=1 < min_interval
    c.tick(3)                                     # tokens_since=4 >= min_interval
    assert c.should_refresh([torch.ones(4)]) is False   # sim=1.0 > threshold → no refresh
    assert c.should_refresh([-torch.ones(4)]) is True   # sim=-1 < threshold → refresh
    c.tick(20)
    assert c.should_refresh(a) is True           # >= max_interval → force
    stats = c.get_stats()
    assert "cache_hit_rate" in stats and "total_refreshes" in stats


def test_generate_dynamic_runs(fake_lm, fake_tokenizer):
    """pipeline.generate(dynamic_refresh=True) runs end-to-end on FakeLM + Doubter."""
    cfg = MetaAttentionConfig(model_name="fake/model", device="cpu", dtype="float32")
    pipe = MetaAttentionPipeline.from_pretrained(cfg, model=fake_lm, tokenizer=fake_tokenizer)
    pipe.attach(Doubter(DoubterConfig(encoder_type="selective")))
    out = pipe.generate("Solve step by step", max_new_tokens=12, dynamic_refresh=True,
                        refresh_min_interval=2, refresh_max_interval=6)
    assert isinstance(out, str)
    assert hasattr(pipe, "last_dynamic_stats")
    assert pipe.last_dynamic_stats["total_tokens"] > 0
