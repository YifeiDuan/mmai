"""Multi-task fusion dataset: graph + text + (band_gap, is_metal)."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch

from src.data.fusion_dataset import FusionDataset, collate_fusion


class MultiTaskFusionDataset(FusionDataset):
    """FusionDataset that additionally returns is_metal as a binary float target."""

    def __getitem__(self, idx: int):
        # Reuse parent for graph + text + main target.
        if idx in self._cache:
            return self._cache[idx]
        atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask, target_t = \
            super().__getitem__(idx)

        row = self.df.iloc[idx]
        is_metal_t = torch.tensor([float(bool(row["is_metal"]))], dtype=torch.float32)

        result = (atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask,
                  target_t, is_metal_t)
        # Override parent cache (parent stored 6-tuple; replace with 7-tuple)
        self._cache[idx] = result
        return result


def collate_multitask_fusion(batch):
    """Like collate_fusion but threads is_metal through as the 8th element."""
    base_batch = [(af, nf, ni, ii, am, t) for (af, nf, ni, ii, am, t, _) in batch]
    is_metals = [item[6] for item in batch]

    (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
     input_ids, attention_mask, targets) = collate_fusion(base_batch)

    is_metal = torch.cat(is_metals, dim=0)

    return (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
            input_ids, attention_mask, targets, is_metal)
