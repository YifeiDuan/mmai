"""Multimodal dataset: crystal graph + text for fusion model."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pymatgen.core import Structure

from src.data.crystal_graph import (
    GaussianDistance,
    build_crystal_graph,
)


class FusionDataset(Dataset):
    """Combined crystal graph + tokenized text dataset."""

    def __init__(
        self,
        df,
        target_col: str = "band_gap",
        text_col: str = "robocrys_text",
        # Graph params
        radius: float = 8.0,
        max_neighbors: int = 12,
        gaussian: Optional[GaussianDistance] = None,
        # Text params
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_seq_len: int = 256,
        tokenizer=None,
    ):
        # Only keep text_ok samples
        self.df = df[df["text_ok"] == True].reset_index(drop=True)
        self.target_col = target_col
        self.text_col = text_col
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.gaussian = gaussian or GaussianDistance()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self._cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        target = float(row[self.target_col])

        # --- Graph ---
        structure = Structure.from_file(row["cif_path"])
        atom_fea, nbr_idx, nbr_fea, _ = build_crystal_graph(
            structure, self.radius, self.max_neighbors, self.gaussian
        )
        atom_fea = torch.tensor(atom_fea, dtype=torch.float32)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx, dtype=torch.long)

        # --- Text ---
        text = str(row[self.text_col])
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_seq_len, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        target_t = torch.tensor([target], dtype=torch.float32)

        result = (atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask, target_t)
        self._cache[idx] = result
        return result


def collate_fusion(batch):
    """Custom collate for variable-size graphs + fixed-size text tokens."""
    (atom_feas, nbr_feas, nbr_idxs,
     input_ids_list, attn_mask_list, targets) = zip(*batch)

    # --- Graph collation (same as crystal_graph.collate_crystal_graphs) ---
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    crystal_atom_idx = []

    offset = 0
    for i, (af, nf, ni) in enumerate(zip(atom_feas, nbr_feas, nbr_idxs)):
        n = af.shape[0]
        batch_atom_fea.append(af)
        batch_nbr_fea.append(nf)
        batch_nbr_idx.append(ni + offset)
        crystal_atom_idx.extend([i] * n)
        offset += n

    # --- Text collation (simple stack) ---
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attn_mask_list, dim=0)

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_idx, dim=0),
        torch.tensor(crystal_atom_idx, dtype=torch.long),
        input_ids,
        attention_mask,
        torch.cat(targets, dim=0),
    )
