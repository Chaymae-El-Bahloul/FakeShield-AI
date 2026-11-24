import torch

from src.models.fusion import build_fusion


def test_fusion_shapes():
    t, i = torch.randn(4, 512), torch.randn(4, 512)
    for name in ["concat", "gated", "attention"]:
        f = build_fusion(name, 512)
        assert f(t, i).shape == (4, f.out_dim)
