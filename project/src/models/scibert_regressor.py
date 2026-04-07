"""SciBERT-based regressor for material property prediction from text."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
import torch.nn as nn
from transformers import AutoModel


class SciBERTRegressor(nn.Module):
    """SciBERT [CLS] embedding -> MLP head -> scalar prediction."""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
        freeze_bert: bool = False,
        unfreeze_last_n_layers: int = 2,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size  # 768

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # Freeze all, then unfreeze last N encoder layers + pooler
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze last N layers
            for layer in self.bert.encoder.layer[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze pooler if exists
            if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            (B, output_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.head(cls_emb)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return [CLS] embedding before the MLP head."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
