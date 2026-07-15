"""daimon-deploy: GGUF-экспорт GoalAnchor (kind/trigger-метадата + тензоры)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gguf")


def _fake_anchor_ckpt(tmp_path):
    """Минимальный goal_anchor-чекпоинт формата 1.1 (2 CA-слоя, крошечный энкодер)."""
    p = tmp_path / "ga.pt"
    torch.save({
        "format_version": "1.1", "kind": "goal_anchor",
        "config": {"encoder_dim": 8, "encoder_num_heads": 2, "ca_bottleneck_dim": 4,
                   "ca_num_heads": 2, "trigger": "fixed", "trigger_k": 77,
                   "trigger_decision_layer": 5},
        "target_layers": [2, 3], "cross_attn_layers": [2, 3],
        "encoder_state": {"queries": torch.randn(2, 8), "proj.weight": torch.randn(8, 16)},
        "ca_state": {"2": {"q_proj.weight": torch.randn(4, 16)},
                     "3": {"q_proj.weight": torch.randn(4, 16)}},
        "trigger_state": None,
    }, p)
    return str(p)


def test_export_anchor_sidecar(tmp_path):
    from gguf import GGUFReader
    from daimon_deploy.export import export_anchor_sidecar

    out = str(tmp_path / "ga.gguf")
    export_anchor_sidecar(_fake_anchor_ckpt(tmp_path), hidden_dim=16, out=out, verbose=False)

    r = GGUFReader(out)
    kv = {f.name: f for f in r.fields.values()}

    def s(name):  # строковое поле
        f = kv[name]; return bytes(f.parts[f.data[-1]]).decode()

    def u(name):  # uint32-поле
        f = kv[name]; return int(f.parts[f.data[-1]][0])

    assert s("daimon.kind") == "goal_anchor"
    assert s("daimon.encoder_type") == "transformer"      # у якоря жёстко
    assert s("daimon.trigger") == "fixed"
    assert u("daimon.trigger_k") == 77
    assert u("daimon.trigger_decision_layer") == 5
    assert u("daimon.num_cognitive_tokens") == 2           # = len(target_layers)
    assert u("daimon.enc_num_heads") == 2                  # из encoder_num_heads
    names = {t.name for t in r.tensors}
    assert "enc.queries" in names and "ca.2.q_proj.weight" in names and "ca.3.q_proj.weight" in names


def test_export_rejects_doubter_via_anchor_api(tmp_path):
    from daimon_deploy.export import export_anchor_sidecar
    p = tmp_path / "d.pt"
    torch.save({"format_version": "1.1", "kind": "doubter", "config": {},
                "encoder_state": {}, "ca_state": {}}, p)
    with pytest.raises(ValueError, match="not a GoalAnchor"):
        export_anchor_sidecar(str(p), hidden_dim=16, out=str(tmp_path / "x.gguf"), verbose=False)
