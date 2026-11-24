import torch
import torch.nn as nn

from .fusion import build_fusion
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class FakeShieldModel(nn.Module):
    """mode: 'text' | 'image' | 'multimodal'  (baselines + fusion model in one class)."""

    def __init__(self, mode="multimodal", fusion="concat", num_classes=2,
                 text_model="bert-base-uncased", dim=512, dropout=0.3, pretrained=True):
        super().__init__()
        assert mode in ("text", "image", "multimodal")
        self.mode, self.dim = mode, dim
        self.text_encoder = TextEncoder(text_model, dim, dropout) if mode != "image" else None
        self.image_encoder = ImageEncoder(dim, dropout, pretrained) if mode != "text" else None
        self.fusion = build_fusion(fusion, dim) if mode == "multimodal" else None
        in_dim = self.fusion.out_dim if self.fusion else dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, image=None, ablate=None):
        """ablate: None | 'text' | 'image' -> zeroes that modality's embedding (multimodal only)."""
        if self.mode == "text":
            return self.head(self.text_encoder(input_ids, attention_mask))
        if self.mode == "image":
            return self.head(self.image_encoder(image))
        t = self.text_encoder(input_ids, attention_mask)
        i = self.image_encoder(image)
        if ablate == "text":
            t = torch.zeros_like(t)
        elif ablate == "image":
            i = torch.zeros_like(i)
        return self.head(self.fusion(t, i))
