"""CPU validation of the SelectiveEncoder numpy spec against PyTorch (phase M2a, no base/GPU).

Pins the exact arithmetic of the current Doubter encoder BEFORE ggml/C++ translates it:
a port mismatch is caught against the numpy spec, not against elusive PyTorch.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("meta_attention")

from meta_attention.encoders.selective import SelectiveEncoder

from daimon_deploy import spec


def test_selective_numpy_matches_pytorch():
    hidden, num_layers, bottleneck = 32, 4, 8
    torch.manual_seed(0)
    enc = SelectiveEncoder(hidden_dim=hidden, num_layers=num_layers, bottleneck_dim=bottleneck)
    # different gates per layer (init 0.3 is uniform → check the scalar gate is accounted for)
    with torch.no_grad():
        for i, g in enumerate(enc.layer_gates):
            g.copy_(torch.tensor([0.1 * (i + 1)]))
    enc.eval()

    acts_t = [torch.randn(1, hidden) for _ in range(num_layers)]
    with torch.no_grad():
        ref = enc(acts_t)[0].numpy()                       # [num_layers, hidden]

    w = {k: v.detach().float().numpy().astype(np.float64) for k, v in enc.state_dict().items()}
    acts_np = [a[0].numpy().astype(np.float64) for a in acts_t]
    got = spec.selective_forward(acts_np, w, num_layers, bottleneck, hidden)

    diff = np.abs(ref - got)
    cos = float(ref.flatten() @ got.flatten() / (np.linalg.norm(ref) * np.linalg.norm(got)))
    assert ref.shape == got.shape == (num_layers, hidden)
    assert diff.max() < 1e-4, f"max abs diff {diff.max():.2e}, cos {cos:.6f}"
    assert cos > 0.99999


def test_selective_gate_zero_zeroes_token():
    """tanh(gate)=0 (gate=0) → the layer's contribution is zeroed before output_proj (scalar-gate sanity)."""
    hidden, num_layers, bottleneck = 16, 3, 8
    torch.manual_seed(1)
    enc = SelectiveEncoder(hidden_dim=hidden, num_layers=num_layers, bottleneck_dim=bottleneck)
    with torch.no_grad():
        for g in enc.layer_gates:
            g.zero_()
    w = {k: v.detach().float().numpy().astype(np.float64) for k, v in enc.state_dict().items()}
    acts_np = [np.random.randn(hidden) for _ in range(num_layers)]
    # projection stack is zero → output_proj sees zeros; the numpy spec must not crash
    got = spec.selective_forward(acts_np, w, num_layers, bottleneck, hidden)
    assert got.shape == (num_layers, hidden)
    assert np.isfinite(got).all()
