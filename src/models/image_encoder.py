import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """ResNet50 (ImageNet) -> 2048-d features -> projection to `out_dim`."""

    def __init__(self, out_dim=512, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, image):
        return self.proj(self.backbone(image))
