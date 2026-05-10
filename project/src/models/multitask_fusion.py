"""Multi-task FiLM fusion: extends LateFusionFiLM with an is_metal classification head."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import LateFusionFiLM


class MultiTaskFiLM(LateFusionFiLM):
    """LateFusionFiLM + auxiliary metal-classification head.

    Reuses the entire FiLM forward path (encoders, projections, modulation, text shortcut).
    Adds one small MLP head from the same fused (h_modulated; h_t_proj) representation
    that produces the band_gap prediction, predicting the metal/non-metal logit.

    Final layer of the metal head is zero-initialised so the model starts neutral
    (sigmoid(0) = 0.5) and only learns the metal signal as gradients flow.
    """

    def __init__(
        self,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        metal_head_hidden: int = 64,
        **encoder_kwargs,
    ):
        super().__init__(
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            output_dim=1,
            **encoder_kwargs,
        )
        d = fusion_hidden_dim
        self.metal_head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(2 * d, metal_head_hidden),
            nn.GELU(),
            nn.Linear(metal_head_hidden, 1),
        )
        nn.init.zeros_(self.metal_head[-1].weight)
        nn.init.zeros_(self.metal_head[-1].bias)

    def forward(
        self,
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
        input_ids, attention_mask,
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
        h_fused = torch.cat([h_modulated, h_t_proj], dim=1)

        bandgap_pred = self.head(h_fused).squeeze(-1)
        metal_logit = self.metal_head(h_fused).squeeze(-1)
        return bandgap_pred, metal_logit


def multitask_loss(
    bandgap_pred: torch.Tensor,
    metal_logit: torch.Tensor,
    target_bg: torch.Tensor,
    target_metal: torch.Tensor,
    alpha_metal: float = 0.3,
    huber_delta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined Huber regression + BCE classification loss.

    Returns (total_loss, regression_loss_detached, metal_loss_detached).
    """
    l_reg = F.huber_loss(bandgap_pred, target_bg, delta=huber_delta, reduction="mean")
    l_metal = F.binary_cross_entropy_with_logits(
        metal_logit, target_metal.float(), reduction="mean"
    )
    total = l_reg + alpha_metal * l_metal
    return total, l_reg.detach(), l_metal.detach()
