"""Cross-attention token fusion (Exp 6).

Per-atom CGCNN node embeddings (after conv stack, before pool) are queries;
per-token SciBERT last_hidden_state are keys/values. Each atom can attend
to the most relevant text tokens (e.g. its bond-length descriptions).
After cross-attention, mean-pool over atoms within each crystal → crystal
embedding → bandgap_head + metal_head (multi-task with Huber + BCE).

CGCNN trunk uses the same 83-dim one-hot atom features as Exp 1/3/5 so the
exp1 CGCNN warm-start checkpoint is directly loadable.
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.cgcnn import CGCNN


def gather_atoms_per_crystal(
    node_emb: torch.Tensor,
    crystal_atom_idx: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad concatenated per-atom embeddings into a (B, max_atoms, d) tensor.

    Args:
        node_emb: (N_total, d) — concatenated atom embeddings across batch.
        crystal_atom_idx: (N_total,) — atom→crystal mapping (in [0, batch_size)).
        batch_size: number of crystals in the batch.

    Returns:
        padded: (B, max_atoms, d) — atoms grouped per crystal, zero-padded.
        mask:   (B, max_atoms) bool — True for valid (non-padding) positions.
    """
    counts = torch.bincount(crystal_atom_idx, minlength=batch_size)
    max_atoms = int(counts.max().item())
    d = node_emb.shape[1]
    device = node_emb.device

    padded = torch.zeros(batch_size, max_atoms, d, device=device, dtype=node_emb.dtype)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = (crystal_atom_idx == b).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n > 0:
            padded[b, :n] = node_emb[idx]
            mask[b, :n] = True
    return padded, mask


class CrossAttnFusion(nn.Module):
    """Token-level cross-attention fusion of CGCNN nodes ↔ SciBERT tokens.

    Forward returns (bandgap_pred, metal_logit) — multi-task heads.
    """

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
        bert_unfreeze_last_n: int = 2,
        # Cross-attention
        attn_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.3,
        head_hidden_dim: int = 128,
        metal_head_hidden: int = 64,
    ):
        super().__init__()

        self.cgcnn_trunk = CGCNN(
            orig_atom_fea_len=orig_atom_fea_len,
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_conv=n_conv,
            h_fea_len=cgcnn_h_fea_len,
            n_h=cgcnn_n_h,
            output_dim=1,
        )
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        # BERT freezing strategy (same as fusion.py's _DualEncoder)
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-bert_unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

        # Project to common attention dim
        self.q_proj = nn.Linear(atom_fea_len, attn_dim)        # node → Q
        self.kv_proj = nn.Linear(self.bert_dim, attn_dim)      # token → shared K/V
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # Bandgap regression head — over crystal-pooled fused features
        self.bandgap_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        # Metal classification head
        self.metal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, metal_head_hidden),
            nn.GELU(),
            nn.Linear(metal_head_hidden, 1),
        )
        # Zero-init last layer of both heads → neutral start
        for m in (self.bandgap_head[-1], self.metal_head[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        """Returns (bandgap_pred, metal_logit) by default.

        If ``return_attention=True``, also returns a dict with the
        cross-attention weights and the per-crystal atom mask, suitable
        for the poster heatmap visualisation.
        """
        # 1. CGCNN node embeddings: (N_total, atom_fea_len)
        node_emb = self.cgcnn_trunk.get_node_embedding(atom_fea, nbr_fea, nbr_idx)

        # 2. SciBERT token embeddings: (B, T, 768)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = bert_out.last_hidden_state

        # 3. Project to common attention dim
        Q_unpadded = self.q_proj(node_emb)               # (N_total, attn_dim)
        KV = self.kv_proj(token_emb)                     # (B, T, attn_dim)

        # 4. Pad atoms-per-crystal into (B, max_atoms, attn_dim)
        batch_size = int(crystal_atom_idx.max().item()) + 1
        Q_padded, atom_mask = gather_atoms_per_crystal(Q_unpadded, crystal_atom_idx, batch_size)
        key_padding_mask = ~attention_mask.bool()

        # 5. Cross-attention: each atom queries the text tokens
        attn_out, attn_weights = self.attn(
            Q_padded, KV, KV, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        attn_out = self.attn_norm(attn_out + Q_padded)

        # 6. Mean pool over valid atoms per crystal → (B, attn_dim)
        atom_mask_f = atom_mask.unsqueeze(-1).float()
        pooled = (attn_out * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1)

        # 7. Heads
        bandgap_pred = self.bandgap_head(pooled).squeeze(-1)
        metal_logit = self.metal_head(pooled).squeeze(-1)
        if return_attention:
            return bandgap_pred, metal_logit, {
                "attn_weights": attn_weights,        # (B, max_atoms, T) — head-averaged
                "atom_mask": atom_mask,              # (B, max_atoms) — True = valid atom
                "token_mask": attention_mask.bool(), # (B, T) — True = valid token
            }
        return bandgap_pred, metal_logit
