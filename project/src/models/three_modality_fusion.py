"""Three-modality fusion: structure (CGCNN) + text (SciBERT) + image (ResNet18).

Architecture:
    Builds on MultiTaskFiLM's text-modulates-structure FiLM core. The image
    branch projects the ResNet18 feature into the same fusion hidden dim and
    is concatenated to the [h_modulated; h_t_proj] stack before the heads.
    Image has no influence on the FiLM gamma/beta — text remains the
    "modulator" — image is an additional information channel that the
    bandgap_head and metal_head can read independently.

Why this topology?
    - Keeps the FiLM SoTA pathway intact so warm-start from Exp 5b's
      checkpoints would (in principle) load cleanly.
    - Image information is global per-crystal (single feature vector);
      mixing it via FiLM γ/β would force a per-feature interaction that
      a 224×224 ball-stick PNG can't reliably support on 383 samples.
      Concat into the head is the lowest-risk integration.
    - Image branch's projection MLP is zero-init, so before any training
      the image channel contributes nothing — the model starts identical
      to MultiTaskFiLM and learns to use image only if it helps val.
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from typing import Tuple

import torch
import torch.nn as nn

from src.models.multitask_fusion import MultiTaskFiLM
from src.models.image_encoder import FrozenResNet18Encoder


class ThreeModalityFusion(MultiTaskFiLM):
    """MultiTaskFiLM + ResNet18 image branch concatenated into the head."""

    def __init__(
        self,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        metal_head_hidden: int = 64,
        image_dim: int = 64,
        freeze_image_trunk: bool = True,
        **encoder_kwargs,
    ):
        # Parent builds CGCNN, BERT, FiLM gen, proj_struct/proj_text,
        # head (Linear(2d → d) → ... → 1) and metal_head (Linear(2d → 64) → ...).
        # We REPLACE both heads to accept 2d + image_dim inputs.
        super().__init__(
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            metal_head_hidden=metal_head_hidden,
            **encoder_kwargs,
        )

        d = fusion_hidden_dim
        self.image_dim = image_dim
        self.image_encoder = FrozenResNet18Encoder(
            out_dim=image_dim, freeze_trunk=freeze_image_trunk,
        )

        head_in = 2 * d + image_dim
        # Replace the inherited heads with image-aware variants
        self.head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(head_in, d),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(d, 1),
        )
        self.metal_head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(head_in, metal_head_hidden),
            nn.GELU(),
            nn.Linear(metal_head_hidden, 1),
        )
        # Zero-init final layer of both heads so model starts neutral
        for m in (self.head[-1], self.metal_head[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
        input_ids, attention_mask,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_s, h_t = self.encode(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                                input_ids, attention_mask)

        h_s_proj = self.proj_struct(h_s)
        h_t_proj = self.proj_text(h_t)

        film_params = self.film_gen(h_t)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma + 1.0
        h_modulated = gamma * h_s_proj + beta
        h_modulated = self.mod_norm(h_modulated) + h_s_proj

        h_img = self.image_encoder(images)        # (B, image_dim)
        h_fused = torch.cat([h_modulated, h_t_proj, h_img], dim=1)

        bandgap_pred = self.head(h_fused).squeeze(-1)
        metal_logit = self.metal_head(h_fused).squeeze(-1)
        return bandgap_pred, metal_logit
