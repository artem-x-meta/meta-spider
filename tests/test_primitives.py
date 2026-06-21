"""Level 1 primitive smoke tests.

Verify that each primitive instantiates correctly, forward returns the right
shapes, and the basic invariants (gate init, freeze, fill/clear) hold.

No HF model loading — we use FakeLM (see conftest.py) for ActivationCollector.
"""

import math

import pytest
import torch

from meta_core import ActivationCollector, BottleneckCrossAttention, ReflexionBuffer, SelectiveEncoder, TransformerEncoder


# ============================================================
# ReflexionBuffer
# ============================================================


def test_reflexion_buffer_fill_clear():
    """fill → is_filled=True; expand_batch correct shape; clear → is_filled=False."""
    buf = ReflexionBuffer(num_slots=4, hidden_dim=64, device="cpu")
    assert buf.is_filled is False
    assert buf.num_slots == 4
    assert buf.hidden_dim == 64

    tokens = torch.randn(1, 4, 64)
    buf.fill(tokens)
    assert buf.is_filled is True

    expanded = buf.expand_batch(batch_size=3)
    assert expanded.shape == (3, 4, 64)

    buf.clear()
    assert buf.is_filled is False


def test_reflexion_buffer_fill_wrong_shape_raises():
    """fill with the wrong num_slots → AssertionError."""
    buf = ReflexionBuffer(num_slots=4, hidden_dim=64, device="cpu")
    tokens_wrong = torch.randn(1, 5, 64)  # 5 instead of 4 slots
    with pytest.raises(AssertionError):
        buf.fill(tokens_wrong)


# ============================================================
# ActivationCollector
# ============================================================


def test_activation_collector_capture(fake_lm, dummy_input):
    """forward → snapshot non-empty with the right int keys + [batch, hidden] shape."""
    collector = ActivationCollector(fake_lm, target_layers=[0, 1, 2, 3])
    fake_lm(dummy_input)  # dummy_input.shape = [2, 8, 64]
    snap = collector.get_snapshot()
    assert set(snap.keys()) == {0, 1, 2, 3}
    for layer_idx, act in snap.items():
        # The hook takes hidden_states[:, -1, :] → [batch, hidden_dim]
        assert act.shape == (2, 64), (
            f"layer_{layer_idx} unexpected shape {act.shape}"
        )
    collector.remove()


def test_activation_collector_freeze(fake_lm, dummy_input):
    """freeze() → the next forward does not overwrite the buffer."""
    collector = ActivationCollector(fake_lm, target_layers=[0, 1])
    fake_lm(dummy_input)
    snap_before = collector.get_snapshot()
    saved = snap_before[0].clone()

    collector.freeze()
    # A second forward with different input must not record anything
    other_input = torch.randn(2, 8, 64)
    fake_lm(other_input)
    snap_after = collector.get_snapshot()
    assert torch.allclose(snap_after[0], saved), "freeze() did not stop recording"
    collector.remove()


def test_activation_collector_multi_layer_tensor(fake_lm, dummy_input):
    """get_multi_layer_tensor → [batch, n_layers, hidden]."""
    collector = ActivationCollector(fake_lm, target_layers=[0, 2, 3])
    fake_lm(dummy_input)  # batch=2
    stacked = collector.get_multi_layer_tensor()
    # batch=2, n_layers=3, hidden=64
    assert stacked.shape == (2, 3, 64), f"unexpected shape {stacked.shape}"
    collector.remove()


def test_activation_collector_activation_list(fake_lm, dummy_input):
    """get_activation_list → a list of [B, hidden] tensors ready for Encoder.forward."""
    collector = ActivationCollector(fake_lm, target_layers=[0, 1, 2, 3])
    fake_lm(dummy_input)
    acts = collector.get_activation_list()
    assert len(acts) == 4
    for a in acts:
        assert a.shape == (2, 64)
    collector.remove()


# ============================================================
# BottleneckCrossAttention
# ============================================================


def test_bottleneck_ca_shape_preservation():
    """input shape preserved + output not identical to the residual."""
    B, seq, hidden = 2, 4, 64
    num_cog = 8

    ca = BottleneckCrossAttention(
        hidden_dim=hidden,
        bottleneck_dim=16,
        num_heads=2,
        num_cognitive_tokens=num_cog,
    )
    hidden_states = torch.randn(B, seq, hidden)
    cog_tokens = torch.randn(B, num_cog, hidden)

    out = ca(hidden_states, cog_tokens)
    assert out.shape == hidden_states.shape, f"shape mismatch: {out.shape}"
    # When gate ≠ 0 the output should differ from the residual
    assert not torch.allclose(out, hidden_states), "output identical to residual"


def test_bottleneck_ca_gate_init():
    """tanh(gate) ≈ tanh(0.3) ≈ 0.291 at init=0.3."""
    ca = BottleneckCrossAttention(
        hidden_dim=64, bottleneck_dim=16, num_heads=2, num_cognitive_tokens=4,
        gate_init=0.3,
    )
    gate_val = ca.get_gate_value()
    expected = math.tanh(0.3)
    assert abs(gate_val - expected) < 1e-5, (
        f"gate_value {gate_val} != tanh(0.3)={expected}"
    )


def test_bottleneck_ca_param_count():
    """Total parameter count in the expected range (sanity)."""
    ca = BottleneckCrossAttention(
        hidden_dim=64, bottleneck_dim=16, num_heads=2, num_cognitive_tokens=4,
    )
    n_params = sum(p.numel() for p in ca.parameters())
    # down(64*16)+q(16*16)+k(64*16)+v(64*16)+up(16*64) + gate(1) + tp(4) + 2x LN(64*2)
    # ≈ 1024 + 256 + 1024 + 1024 + 1024 + 1 + 4 + 256 ≈ 4613
    assert 3000 < n_params < 10000, f"unexpected param count {n_params}"


# ============================================================
# SelectiveEncoder
# ============================================================


def test_selective_encoder_forward(dummy_activations_4_layers):
    """activation_list of 4 → cognitive_tokens [B, 4, hidden]."""
    encoder = SelectiveEncoder(hidden_dim=64, num_layers=4, bottleneck_dim=16)
    cog_tokens = encoder(dummy_activations_4_layers)
    assert cog_tokens.shape == (2, 4, 64), f"unexpected shape {cog_tokens.shape}"


def test_selective_encoder_gate_diagnostics():
    """get_gate_values returns 4 per-layer tanh values, init ≈ tanh(0.3)."""
    encoder = SelectiveEncoder(hidden_dim=64, num_layers=4, gate_init=0.3)
    gates = encoder.get_gate_values()
    assert set(gates.keys()) == {f"layer_{i}" for i in range(4)}
    expected = math.tanh(0.3)
    for v in gates.values():
        assert abs(v - expected) < 1e-5


def test_selective_encoder_active_layers():
    """get_active_layers with a threshold above init returns an empty list."""
    encoder = SelectiveEncoder(hidden_dim=64, num_layers=4, gate_init=0.3)
    # tanh(0.3) ≈ 0.291 < 0.5
    active = encoder.get_active_layers(threshold=0.5)
    assert active == []
    # tanh(0.3) ≈ 0.291 > 0.1 — all 4 layers active
    active = encoder.get_active_layers(threshold=0.1)
    assert active == [0, 1, 2, 3]


# ============================================================
# TransformerEncoder
# ============================================================


def test_transformer_encoder_forward(dummy_activations_4_layers):
    """activation_list of 4 → cognitive_tokens [B, 4, hidden] + transformer blocks active."""
    encoder = TransformerEncoder(
        hidden_dim=64,
        num_layers=4,
        encoder_dim=32,
        num_blocks=1,
        num_heads=2,
    )
    cog_tokens = encoder(dummy_activations_4_layers)
    assert cog_tokens.shape == (2, 4, 64), f"unexpected shape {cog_tokens.shape}"


def test_transformer_encoder_pos_embed_optional(dummy_activations_4_layers):
    """use_layer_pos_embeddings=False → forward works (without position embedding)."""
    encoder = TransformerEncoder(
        hidden_dim=64,
        num_layers=4,
        encoder_dim=32,
        num_blocks=1,
        num_heads=2,
        use_layer_pos_embeddings=False,
    )
    assert not hasattr(encoder, "layer_pos_embed")
    cog_tokens = encoder(dummy_activations_4_layers)
    assert cog_tokens.shape == (2, 4, 64)


def test_transformer_encoder_shared_projector(dummy_activations_4_layers):
    """use_per_layer_projectors=False → shared projector works."""
    encoder = TransformerEncoder(
        hidden_dim=64,
        num_layers=4,
        encoder_dim=32,
        num_blocks=1,
        num_heads=2,
        use_per_layer_projectors=False,
    )
    assert hasattr(encoder, "shared_projector")
    assert not hasattr(encoder, "layer_projectors")
    cog_tokens = encoder(dummy_activations_4_layers)
    assert cog_tokens.shape == (2, 4, 64)


# ============================================================
# End-to-end mini chain
# ============================================================


def test_end_to_end_chain():
    """encoder → buffer → CA: shape preserved across the whole chain."""
    B, n_layers, hidden = 2, 4, 64
    encoder = SelectiveEncoder(hidden_dim=hidden, num_layers=n_layers, bottleneck_dim=16)
    ca = BottleneckCrossAttention(
        hidden_dim=hidden, bottleneck_dim=16, num_heads=2,
        num_cognitive_tokens=n_layers,
    )
    buf = ReflexionBuffer(num_slots=n_layers, hidden_dim=hidden, device="cpu")

    acts = [torch.randn(B, hidden) for _ in range(n_layers)]
    cog_tokens = encoder(acts)
    buf.fill(cog_tokens)
    hidden_states = torch.randn(B, 8, hidden)
    modified = ca(hidden_states, buf.expand_batch(B))
    assert modified.shape == hidden_states.shape


# ============================================================
# StageTimer
# ============================================================


def test_stage_timer_collects_stages(tmp_path):
    """StageTimer records stages, total, and a JSON report."""
    import json
    import time as _t

    from meta_core import StageTimer

    timer = StageTimer(context={"dtype": "float32"})
    with timer.stage("alpha"):
        _t.sleep(0.05)
    with timer.stage("beta"):
        pass

    assert [s["name"] for s in timer.stages] == ["alpha", "beta"]
    assert timer.stages[0]["seconds"] >= 0.0
    assert timer.total_seconds >= timer.stages[0]["seconds"]

    text = timer.summary()
    assert "alpha" in text and "TOTAL" in text

    out = tmp_path / "timing.json"
    timer.save_json(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["context"]["dtype"] == "float32"
    assert len(data["stages"]) == 2
    assert "hardware" in data


def test_stage_timer_survives_exception():
    """The stage is recorded even when an exception is raised inside."""
    import pytest as _pytest

    from meta_core import StageTimer

    timer = StageTimer()
    with _pytest.raises(RuntimeError):
        with timer.stage("boom"):
            raise RuntimeError("x")
    assert timer.stages and timer.stages[0]["name"] == "boom"
