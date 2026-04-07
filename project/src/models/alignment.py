"""CLIP-style contrastive alignment for crystal structure and text."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.cgcnn import CGCNN


class CLIPAlignmentModel(nn.Module):
    """CLIP-style model: project structure and text into shared space with InfoNCE."""

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
        # Projection
        proj_dim: int = 128,
        temperature_init: float = 0.07,
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
        self.bert_out_dim = self.bert.config.hidden_size

        # Freeze BERT, unfreeze last N layers
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-bert_unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

        # Projection heads: encoder_dim -> proj_dim, with LayerNorm
        self.proj_struct = nn.Sequential(
            nn.Linear(self.cgcnn_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(self.bert_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Learnable temperature (log scale for stability)
        self.log_temperature = nn.Parameter(
            torch.tensor([float(-torch.log(torch.tensor(temperature_init)))])
        )

    @property
    def temperature(self):
        # Clamp to avoid numerical issues
        return torch.exp(-self.log_temperature).clamp(min=0.01, max=100.0)

    def encode_structure(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx):
        """Return L2-normalized structure embedding."""
        h = self.cgcnn.get_embedding(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        z = self.proj_struct(h)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """Return L2-normalized text embedding."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        z = self.proj_text(h)
        return F.normalize(z, dim=-1)

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                input_ids, attention_mask):
        """Return (z_struct, z_text, temperature) for loss computation."""
        z_s = self.encode_structure(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        z_t = self.encode_text(input_ids, attention_mask)
        return z_s, z_t, self.temperature


def info_nce_loss(z_s, z_t, temperature):
    """Symmetric InfoNCE loss (CLIP-style).

    Args:
        z_s: (B, d) L2-normalized structure embeddings
        z_t: (B, d) L2-normalized text embeddings
        temperature: scalar

    Returns:
        loss: scalar
    """
    # Similarity matrix: (B, B)
    logits = (z_s @ z_t.T) * temperature  # higher temp = sharper distribution
    B = z_s.shape[0]
    labels = torch.arange(B, device=z_s.device)

    # Symmetric: structure->text and text->structure
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return (loss_s2t + loss_t2s) / 2


def compute_alignment_metrics(z_s, z_t):
    """Compute retrieval accuracy and mean similarity for monitoring."""
    sim = z_s @ z_t.T  # (B, B)
    B = sim.shape[0]
    labels = torch.arange(B, device=sim.device)

    # Top-1 retrieval accuracy
    s2t_acc = (sim.argmax(dim=1) == labels).float().mean().item()
    t2s_acc = (sim.argmax(dim=0) == labels).float().mean().item()

    # Mean positive pair similarity
    mean_pos_sim = sim.diag().mean().item()

    return {
        "s2t_acc": s2t_acc,
        "t2s_acc": t2s_acc,
        "mean_retrieval_acc": (s2t_acc + t2s_acc) / 2,
        "mean_pos_sim": mean_pos_sim,
    }


def soft_info_nce_loss(z_s, z_t, targets, temperature, margin=0.5):
    """Soft contrastive: down-weight negatives with similar band gap.

    Pairs with |gap_i - gap_j| < margin are masked out as negatives,
    so the model doesn't waste effort pushing similar materials apart.
    """
    B = z_s.shape[0]
    logits = (z_s @ z_t.T) * temperature  # (B, B)
    labels = torch.arange(B, device=z_s.device)

    # Mask: pairs with similar targets should not be hard negatives
    target_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()  # (B, B)
    soft_mask = (target_diff < margin).float()
    # Zero out diagonal (positive pairs) from mask
    soft_mask.fill_diagonal_(0.0)
    # Apply: add large negative value to logits for soft negatives
    logits = logits - soft_mask * 1e9

    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return (loss_s2t + loss_t2s) / 2


class JointAlignmentModel(CLIPAlignmentModel):
    """CLIP alignment + regression head, trained jointly.

    L = alpha * L_contrastive + (1 - alpha) * L_regression
    """

    def __init__(self, proj_dim: int = 128, reg_hidden_dim: int = 128,
                 reg_dropout: float = 0.1, **encoder_kwargs):
        super().__init__(proj_dim=proj_dim, **encoder_kwargs)

        # Lightweight regression head on concatenated aligned embeddings
        self.reg_head = nn.Sequential(
            nn.Dropout(reg_dropout),
            nn.Linear(proj_dim * 2, reg_hidden_dim),
            nn.GELU(),
            nn.Linear(reg_hidden_dim, 1),
        )

    def forward_joint(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                      input_ids, attention_mask):
        """Return (z_s, z_t, temperature, pred)."""
        z_s = self.encode_structure(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        z_t = self.encode_text(input_ids, attention_mask)
        pred = self.reg_head(torch.cat([z_s, z_t], dim=1)).squeeze(-1)
        return z_s, z_t, self.temperature, pred


class AlignedRegressor(nn.Module):
    """Regression head on top of aligned embeddings.

    Supports three modes:
      - 'concat': [z_s; z_t] -> MLP
      - 'sum':    z_s + z_t  -> MLP  (works because they're now in same space)
      - 'film':   text modulates structure (same as Exp 3 FiLM but on aligned space)
    """

    def __init__(
        self,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        mode: str = "concat",
        output_dim: int = 1,
    ):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            in_dim = proj_dim * 2
        elif mode == "sum":
            in_dim = proj_dim
        elif mode == "film":
            self.film_gen = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim * 2),
            )
            nn.init.zeros_(self.film_gen[-1].weight)
            nn.init.zeros_(self.film_gen[-1].bias)
            self.mod_norm = nn.LayerNorm(proj_dim)
            in_dim = proj_dim * 2  # modulated + text shortcut
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            h = torch.cat([z_s, z_t], dim=1)
        elif self.mode == "sum":
            h = z_s + z_t
        elif self.mode == "film":
            params = self.film_gen(z_t)
            gamma, beta = params.chunk(2, dim=1)
            gamma = gamma + 1.0
            h_mod = self.mod_norm(gamma * z_s + beta) + z_s
            h = torch.cat([h_mod, z_t], dim=1)
        return self.head(h)
