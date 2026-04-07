"""Fusion models: CGCNN (structure) + SciBERT (text) -> prediction.

Three fusion strategies:
  1. LateFusionConcat   — Concatenation + MLP (baseline)
  2. LateFusionGated    — Learned gate weighting between modalities
  3. LateFusionFiLM     — Text modulates structure features via FiLM
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.cgcnn import CGCNN


# ======================================================================
# Shared encoder wrapper
# ======================================================================

class _DualEncoder(nn.Module):
    """Base class: CGCNN + SciBERT encoders. Subclasses define fusion logic."""

    def __init__(
        self,
        # CGCNN
        orig_atom_fea_len: int,
        atom_fea_len: int = 64,
        nbr_fea_len: int = 41,
        n_conv: int = 3,
        cgcnn_h_fea_len: int = 128,
        cgcnn_n_h: int = 1,
        # SciBERT
        bert_model_name: str = "allenai/scibert_scivocab_uncased",
        freeze_bert: bool = False,
        bert_unfreeze_last_n: int = 2,
    ):
        super().__init__()

        # CGCNN encoder
        self.cgcnn = CGCNN(
            orig_atom_fea_len=orig_atom_fea_len,
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_conv=n_conv,
            h_fea_len=cgcnn_h_fea_len,
            n_h=cgcnn_n_h,
            output_dim=1,
        )
        self.cgcnn_out_dim = cgcnn_h_fea_len

        # SciBERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_out_dim = self.bert.config.hidden_size  # 768

        # Freeze strategy
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = False
            for layer in self.bert.encoder.layer[-bert_unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
                for p in self.bert.pooler.parameters():
                    p.requires_grad = True

    def encode(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
               input_ids, attention_mask):
        """Return (h_struct, h_text)."""
        h_struct = self.cgcnn.get_embedding(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = bert_out.last_hidden_state[:, 0, :]
        return h_struct, h_text

    def freeze_cgcnn(self):
        for p in self.cgcnn.parameters():
            p.requires_grad = False

    def unfreeze_cgcnn(self):
        for p in self.cgcnn.parameters():
            p.requires_grad = True


# ======================================================================
# 1. Concatenation Fusion (baseline)
# ======================================================================

class LateFusionConcat(_DualEncoder):
    """[h_struct; h_text] -> MLP -> prediction."""

    def __init__(self, fusion_hidden_dim: int = 256, fusion_dropout: float = 0.2,
                 output_dim: int = 1, **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        fused_dim = self.cgcnn_out_dim + self.bert_out_dim  # 128 + 768 = 896
        self.fusion_head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                input_ids, attention_mask):
        h_s, h_t = self.encode(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                                input_ids, attention_mask)
        h_fused = torch.cat([h_s, h_t], dim=1)
        return self.fusion_head(h_fused)


# ======================================================================
# 2. Gated Fusion
# ======================================================================

class LateFusionGated(_DualEncoder):
    """Learned gate with LayerNorm + residual for stable training.

    g = sigmoid(W·[h_s'; h_t']), fused = LN(g*h_s' + (1-g)*h_t') + skip
    """

    def __init__(self, fusion_hidden_dim: int = 256, fusion_dropout: float = 0.2,
                 output_dim: int = 1, **encoder_kwargs):
        super().__init__(**encoder_kwargs)

        d = fusion_hidden_dim

        # Project both modalities to same dim, with LayerNorm for stable gate input
        self.proj_struct = nn.Sequential(
            nn.Linear(self.cgcnn_out_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(self.bert_out_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Gate network: bias initialized to 0 → sigmoid(0)=0.5 → balanced start
        self.gate_linear = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate_linear.bias)

        # LayerNorm after gating for stable gradient flow
        self.fused_norm = nn.LayerNorm(d)

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(d, output_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                input_ids, attention_mask):
        h_s, h_t = self.encode(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                                input_ids, attention_mask)

        h_s_proj = self.proj_struct(h_s)  # (B, d)
        h_t_proj = self.proj_text(h_t)    # (B, d)

        g = torch.sigmoid(self.gate_linear(torch.cat([h_s_proj, h_t_proj], dim=1)))
        h_gated = g * h_s_proj + (1 - g) * h_t_proj

        # Residual: also pass through the mean of both, so gradient always flows
        h_skip = (h_s_proj + h_t_proj) * 0.5
        h_fused = self.fused_norm(h_gated + h_skip)

        return self.head(h_fused)

    def get_gate_values(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                        input_ids, attention_mask):
        """Return gate values for interpretability. g~1 means trusting structure more."""
        h_s, h_t = self.encode(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                                input_ids, attention_mask)
        h_s_proj = self.proj_struct(h_s)
        h_t_proj = self.proj_text(h_t)
        g = torch.sigmoid(self.gate_linear(torch.cat([h_s_proj, h_t_proj], dim=1)))
        return g.mean(dim=1)  # (B,) average gate per sample


# ======================================================================
# 3. FiLM Fusion (Feature-wise Linear Modulation)
# ======================================================================

class LateFusionFiLM(_DualEncoder):
    """Text modulates structure via FiLM, with LayerNorm + residual + text shortcut.

    h_modulated = LN(gamma * h_s' + beta) + h_s'   (residual)
    h_fused = [h_modulated; h_t']                   (text shortcut to head)
    """

    def __init__(self, fusion_hidden_dim: int = 256, fusion_dropout: float = 0.2,
                 output_dim: int = 1, **encoder_kwargs):
        super().__init__(**encoder_kwargs)

        d = fusion_hidden_dim

        # Project structure with LayerNorm
        self.proj_struct = nn.Sequential(
            nn.Linear(self.cgcnn_out_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Project text for shortcut to head
        self.proj_text = nn.Sequential(
            nn.Linear(self.bert_out_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # FiLM generator: text -> (gamma, beta)
        # Two-layer for more expressive modulation
        self.film_gen = nn.Sequential(
            nn.Linear(self.bert_out_dim, d),
            nn.GELU(),
            nn.Linear(d, d * 2),
        )
        # Init last layer bias to zero → gamma starts at 1, beta at 0
        nn.init.zeros_(self.film_gen[-1].bias)
        nn.init.zeros_(self.film_gen[-1].weight)

        # LayerNorm after modulation
        self.mod_norm = nn.LayerNorm(d)

        # Prediction head: takes [h_modulated; h_text_proj] = 2d input
        self.head = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(d, output_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                input_ids, attention_mask):
        h_s, h_t = self.encode(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                                input_ids, attention_mask)

        h_s_proj = self.proj_struct(h_s)     # (B, d)
        h_t_proj = self.proj_text(h_t)       # (B, d)

        film_params = self.film_gen(h_t)     # (B, 2d)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma + 1.0  # center around identity transform

        h_modulated = gamma * h_s_proj + beta
        h_modulated = self.mod_norm(h_modulated) + h_s_proj  # residual

        # Concat modulated structure + text shortcut → head sees both
        h_fused = torch.cat([h_modulated, h_t_proj], dim=1)

        return self.head(h_fused)


# ======================================================================
# Factory
# ======================================================================

FUSION_MODELS = {
    "concat": LateFusionConcat,
    "gated": LateFusionGated,
    "film": LateFusionFiLM,
}


def build_fusion_model(fusion_type: str = "concat", **kwargs) -> _DualEncoder:
    """Build a fusion model by name.

    Args:
        fusion_type: one of "concat", "gated", "film"
        **kwargs: passed to the model constructor

    Returns:
        Fusion model instance.
    """
    if fusion_type not in FUSION_MODELS:
        raise ValueError(f"Unknown fusion type '{fusion_type}'. Choose from {list(FUSION_MODELS.keys())}")
    return FUSION_MODELS[fusion_type](**kwargs)
