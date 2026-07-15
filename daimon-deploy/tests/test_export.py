"""CPU test of exporting the sidecar to a GGUF sidecar (no base/GPU): round-trip of tensor
names+values and reading run.json. Proves export is encoder-agnostic and metaloom run-dir compatible."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gguf")

from daimon_deploy import export_from_run_dir, export_sidecar


def _fake_checkpoint(path):
    """Minimal sidecar: 2 enc tensors + CA on two layers. encoder_type=selective."""
    enc = {"queries": torch.randn(4, 16), "output_norm.weight": torch.ones(8)}
    ca = {6: {"q_proj.weight": torch.randn(8, 8)},
          12: {"q_proj.weight": torch.randn(8, 8)}}
    ck = {"config": {"encoder_type": "selective", "num_cognitive_tokens": 4,
                     "ca_bottleneck_dim": 16, "ca_num_heads": 2, "transformer_num_heads": 2},
          "encoder_state": enc, "ca_state": ca}
    torch.save(ck, path)
    return enc, ca


def test_export_roundtrip(tmp_path):
    ckpt = tmp_path / "doubter.pt"
    enc, ca = _fake_checkpoint(ckpt)
    out = tmp_path / "sidecar.gguf"
    export_sidecar(str(ckpt), target_layers=[6, 12], cross_attn_layers=[6, 12],
                   hidden_dim=8, out=str(out), verbose=False)
    assert out.exists()

    from gguf import GGUFReader
    r = GGUFReader(str(out))
    names = {t.name for t in r.tensors}
    assert {"enc.queries", "enc.output_norm.weight",
            "ca.6.q_proj.weight", "ca.12.q_proj.weight"} <= names

    # values round-trip (axis order in gguf may differ → compare sorted)
    q = next(t for t in r.tensors if t.name == "enc.queries")
    assert np.allclose(np.sort(np.asarray(q.data).ravel()),
                       np.sort(enc["queries"].numpy().ravel()), atol=1e-5)


def test_export_from_run_dir(tmp_path):
    """export_from_run_dir reads hidden_dim/layers from run.json (metaloom convention)."""
    _fake_checkpoint(tmp_path / "doubter_checkpoint.pt")
    (tmp_path / "run.json").write_text(json.dumps({
        "model_name": "fake", "encoder_type": "selective", "hidden_dim": 8,
        "target_layers": [6, 12], "cross_attn_layers": [6, 12],
        "train_size": 1, "val_size": 1, "test_size": 1,
    }), encoding="utf-8")

    out = export_from_run_dir(str(tmp_path), verbose=False)
    assert Path(out).exists()


def test_run_dir_missing_hidden_dim(tmp_path):
    """Without hidden_dim in run.json — a clear error, not silent garbage."""
    _fake_checkpoint(tmp_path / "doubter_checkpoint.pt")
    (tmp_path / "run.json").write_text(json.dumps({
        "target_layers": [6], "cross_attn_layers": [6],
        "train_size": 1, "val_size": 1, "test_size": 1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden_dim"):
        export_from_run_dir(str(tmp_path), verbose=False)
