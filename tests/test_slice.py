"""Slice-trainer tests: late_slice preset + cut_layer (Chunk 1), vocab-shard loss (Chunk 2),
collector cut-hidden (Chunk 3), partial slice-forward (Chunk 4, wiring on FakeLM).
"""
import pytest
import torch

from meta_core import MetaSpiderConfig


# ───────────────────────── Chunk 1: late_slice preset ─────────────────────────

def _resolved(num_layers, ca, target):
    cfg = MetaSpiderConfig(model_name="x", target_layers=target, cross_attn_layers=ca)
    cfg.resolve_defaults(base_num_layers=num_layers, base_hidden_dim=64)
    return cfg


def test_late_slice_reads_strictly_below_ca():
    cfg = _resolved(48, [34, 38, 42, 46], "late_slice")
    assert cfg.target_layers == list(range(16, 34))  # num_layers//3 .. min_ca-1
    assert all(t < 34 for t in cfg.target_layers)
    assert cfg.slice_cut_layer() == 33  # min_ca - 1


def test_late_slice_requires_explicit_ca():
    cfg = MetaSpiderConfig(model_name="x", target_layers="late_slice")  # ca None
    with pytest.raises(ValueError):
        cfg.resolve_defaults(48, 64)


def test_late_slice_rejects_low_ca():
    cfg = MetaSpiderConfig(model_name="x", target_layers="late_slice", cross_attn_layers=[2, 4])
    with pytest.raises(ValueError):
        cfg.resolve_defaults(48, 64)  # min_ca=2 <= num_layers//3 + 1


def test_slice_cut_layer_validates_reads_below_cut():
    cfg = _resolved(48, [34], [30, 40])  # read 40 > cut 33 → invalid for the slice
    with pytest.raises(ValueError):
        cfg.slice_cut_layer()


def test_normal_presets_unaffected_by_late_slice_code():
    cfg = _resolved(48, None, "late")  # ca defaults to target
    assert cfg.target_layers == list(range(32, 48))
    assert cfg.cross_attn_layers == cfg.target_layers
    cfg2 = _resolved(48, [34, 38], [10, 20, 30])  # both explicit
    assert cfg2.target_layers == [10, 20, 30]
    assert cfg2.cross_attn_layers == [34, 38]


# ───────────────────────── Chunk 2: chunked vocab loss ─────────────────────────

def test_chunked_lm_loss_matches_full_ce():
    from meta_loom.training.losses import chunked_lm_loss
    torch.manual_seed(0)
    B, S, H, V = 2, 7, 64, 50
    hidden = torch.randn(B, S, H, requires_grad=True)
    W = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    labels[:, 0] = -100  # mask
    logits = torch.nn.functional.linear(hidden.detach(), W)
    ref = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
    got = chunked_lm_loss(hidden, W, labels, n_chunks=3)
    assert torch.allclose(got, ref, atol=1e-4)
    got.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_chunked_lm_loss_all_masked_is_zero():
    from meta_loom.training.losses import chunked_lm_loss
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    loss = chunked_lm_loss(hidden, torch.randn(5, 8), torch.full((1, 4), -100))
    assert loss.item() == 0.0


def test_chunked_lm_loss_softcap():
    """softcap (Gemma final_logit_softcapping=30) == manual 30·tanh(logits/30) before CE."""
    import torch.nn.functional as F

    from meta_loom.training.losses import chunked_lm_loss
    torch.manual_seed(0)
    B, S, H, V = 2, 7, 16, 50
    hidden = torch.randn(B, S, H)
    W = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    labels[:, 0] = -100
    cap = 30.0
    logits = F.linear(hidden[:, :-1], W).float()
    logits = cap * torch.tanh(logits / cap)
    ref = F.cross_entropy(logits.reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    got = chunked_lm_loss(hidden, W, labels, n_chunks=3, softcap=cap)
    assert torch.allclose(got, ref, atol=1e-4)
    # without softcap the result is DIFFERENT (so the cap is actually applied)
    no_cap = chunked_lm_loss(hidden, W, labels, n_chunks=3, softcap=None)
    assert not torch.allclose(got, no_cap, atol=1e-3)


def test_chunked_lm_loss_callable_lm_head():
    """lm_head as a callable module (the path for nf4 Linear4bit) == passing the weight directly."""
    import torch.nn as nn

    from meta_loom.training.losses import chunked_lm_loss
    torch.manual_seed(0)
    B, S, H, V = 2, 6, 16, 40
    hidden = torch.randn(B, S, H)
    head = nn.Linear(H, V, bias=False)
    labels = torch.randint(0, V, (B, S))
    labels[:, 0] = -100
    via_weight = chunked_lm_loss(hidden, head.weight, labels, n_chunks=2)
    via_module = chunked_lm_loss(hidden, head, labels, n_chunks=2)  # callable
    assert torch.allclose(via_weight, via_module, atol=1e-5)


# ───────────── Chunk 3+4 wiring: --slice flag, dataset 1.3, collate ─────────────

def test_parse_layers_passes_late_slice():
    from meta_loom.cli._common import parse_layers
    assert parse_layers("late_slice") == "late_slice"
    assert parse_layers("late") == "late"
    assert parse_layers("34,38") == [34, 38]


def test_dataset_roundtrip_cut_hidden_v13(tmp_path):
    """cut_hidden/input_ids/labels survive save/load; format bumps 1.2→1.3."""
    from meta_loom import ActivationDatasetCollector
    from meta_loom.training.collector import DatasetSample

    s = DatasetSample(
        input_text="q", ground_truth="a", activations={5: torch.randn(8)},
        pass1_correct=True, cut_hidden=torch.randn(7, 8),
        input_ids_full=torch.arange(7), labels_full=torch.full((7,), -100),
    )
    p = tmp_path / "ds.pt"
    ActivationDatasetCollector.save([s], str(p))
    assert torch.load(str(p), weights_only=False)["format_version"] == "1.3"
    loaded = ActivationDatasetCollector.load(str(p))
    assert loaded[0].cut_hidden.shape == (7, 8)
    assert loaded[0].input_ids_full.shape == (7,)

    # without cut_hidden → stays 1.2 (backward compat)
    s2 = DatasetSample(input_text="q", ground_truth="a", activations={5: torch.randn(8)})
    p2 = tmp_path / "ds2.pt"
    ActivationDatasetCollector.save([s2], str(p2))
    assert torch.load(str(p2), weights_only=False)["format_version"] == "1.2"
    assert ActivationDatasetCollector.load(str(p2))[0].cut_hidden is None


def test_build_slice_device_map():
    """device_map: bottom (0..cut)→cpu, top + embed/rotary/norm/lm_head→GPU."""
    from meta_core.slice_forward import build_slice_device_map
    dm = build_slice_device_map(num_layers=32, cut_layer=20)
    assert dm["model.layers.0"] == "cpu" and dm["model.layers.20"] == "cpu"
    assert dm["model.layers.21"] == 0 and dm["model.layers.31"] == 0
    assert dm["model.embed_tokens"] == 0 and dm["model.rotary_emb"] == 0
    assert dm["model.norm"] == 0 and dm["lm_head"] == 0
    # all layers covered (accelerate requires full coverage)
    assert sum(1 for k in dm if k.startswith("model.layers.")) == 32


def test_collate_slice_right_pads():
    """_collate_slice: right-pad (real tokens first), labels/cut_hidden zeroed in the tail."""
    from types import SimpleNamespace

    from meta_loom import Trainer
    from meta_loom.training.collector import DatasetSample

    samples = [
        DatasetSample(input_text="a", ground_truth="x", activations={},
                      cut_hidden=torch.ones(3, 4), input_ids_full=torch.tensor([1, 2, 3]),
                      labels_full=torch.tensor([-100, 2, 3])),
        DatasetSample(input_text="b", ground_truth="y", activations={},
                      cut_hidden=torch.ones(2, 4) * 2, input_ids_full=torch.tensor([4, 5]),
                      labels_full=torch.tensor([-100, 5])),
    ]
    fake = SimpleNamespace(pipeline=SimpleNamespace(
        config=SimpleNamespace(dtype="float32"),
        tokenizer=SimpleNamespace(pad_token_id=0)))
    ids, attn, labels, cut_h = Trainer._collate_slice(fake, samples, "cpu")
    assert ids.shape == (2, 3) and cut_h.shape == (2, 3, 4)
    assert attn[1].tolist() == [1, 1, 0]                # right-pad: real→0..n, pad→tail
    assert labels[1].tolist() == [-100, 5, -100]        # pad position → -100
    assert cut_h[1, 2].abs().sum().item() == 0.0        # pad cut_hidden = 0
    assert cut_h[1, 0, 0].item() == 2.0                 # real cut_hidden preserved
