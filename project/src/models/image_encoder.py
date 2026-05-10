"""Frozen image encoder for the 3-modality (struct + text + image) experiment.

Uses ImageNet-pretrained ResNet18 with the final classification layer replaced
by an Identity, then a small projection MLP brings the 512-d ResNet18 features
down to a configurable dim (default 64) compatible with the FiLM hidden size.

The ResNet trunk is frozen by default (small dataset → no benefit from
fine-tuning a vision backbone); only the projection MLP trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class FrozenResNet18Encoder(nn.Module):
    """ImageNet-pretrained ResNet18 → projection MLP → out_dim features."""

    def __init__(self, out_dim: int = 64, freeze_trunk: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.trunk = torchvision.models.resnet18(weights=weights)
        feat_dim = self.trunk.fc.in_features  # 512
        self.trunk.fc = nn.Identity()
        self.preprocess = weights.transforms()  # standard 224×224 resize+normalize
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )
        # zero-init final projection layer → image branch starts neutral
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) — already preprocessed (use self.preprocess)."""
        feat = self.trunk(images)        # (B, 512)
        return self.proj(feat)            # (B, out_dim)
