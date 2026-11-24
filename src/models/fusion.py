import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.out_dim = 2 * dim

    def forward(self, t, i):
        return torch.cat([t, i], dim=-1)


class GatedFusion(nn.Module):
    """Learns a per-feature gate deciding how much to trust text vs image."""

    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.out_dim = dim

    def forward(self, t, i):
        g = self.gate(torch.cat([t, i], dim=-1))
        return g * t + (1 - g) * i


class AttentionFusion(nn.Module):
    """Treats the two modalities as a 2-token sequence and applies self-attention."""

    def __init__(self, dim=512, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, t, i):
        x = torch.stack([t, i], dim=1)
        a, _ = self.attn(x, x, x)
        return self.norm(x + a).mean(dim=1)


def build_fusion(name: str, dim: int = 512) -> nn.Module:
    return {"concat": ConcatFusion, "gated": GatedFusion, "attention": AttentionFusion}[name](dim)
