"""5-seed ensemble forward oracle wrapping Exp 5b MultiTaskFiLM checkpoints.

Given a (cif_path, text) pair the oracle builds a single-sample crystal graph
plus tokenized text, runs all 5 trained MultiTaskFiLM checkpoints in eval mode,
and returns ensemble mean / std of the band-gap regression head plus mean
sigmoid of the is_metal head. The std doubles as an OOD confidence proxy that
the agent uses to flag low-confidence candidates.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from pymatgen.core import Structure
from transformers import AutoTokenizer

from src.data.crystal_graph import (
    GaussianDistance,
    NUM_ATOM_FEATURES,
    build_crystal_graph,
)
from src.models.multitask_fusion import MultiTaskFiLM

logger = logging.getLogger(__name__)


DEFAULT_SEEDS: Sequence[int] = (42, 1337, 2024, 7, 999)


class EnsembleForwardOracle:
    """5-seed ensemble of MultiTaskFiLM trained under Exp 5b (no warm start)."""

    def __init__(
        self,
        ckpt_dir: str,
        device: Optional[str] = None,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        bert_model_name: str = "allenai/scibert_scivocab_uncased",
        radius: float = 8.0,
        max_neighbors: int = 12,
        gaussian_dmin: float = 0.0,
        gaussian_dmax: float = 8.0,
        gaussian_step: float = 0.2,
        text_max_seq_len: int = 256,
        # MultiTaskFiLM hyperparameters (must match training config)
        atom_fea_len: int = 64,
        n_conv: int = 3,
        cgcnn_h_fea_len: int = 128,
        cgcnn_n_h: int = 1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,  # Exp 5b YAML uses 0.3
        metal_head_hidden: int = 64,
        bert_unfreeze_last_n: int = 2,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.bert_model_name = bert_model_name
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.text_max_seq_len = text_max_seq_len

        self.gaussian = GaussianDistance(gaussian_dmin, gaussian_dmax, gaussian_step)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        self.models: List[MultiTaskFiLM] = []
        for seed in seeds:
            ckpt = self.ckpt_dir / f"seed_{seed}" / "best_model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            model = MultiTaskFiLM(
                fusion_hidden_dim=fusion_hidden_dim,
                fusion_dropout=fusion_dropout,
                metal_head_hidden=metal_head_hidden,
                orig_atom_fea_len=NUM_ATOM_FEATURES,
                atom_fea_len=atom_fea_len,
                nbr_fea_len=self.gaussian.dim,
                n_conv=n_conv,
                cgcnn_h_fea_len=cgcnn_h_fea_len,
                cgcnn_n_h=cgcnn_n_h,
                bert_model_name=bert_model_name,
                bert_unfreeze_last_n=bert_unfreeze_last_n,
            )
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("seed_%d missing keys: %d (%s...)",
                               seed, len(missing), missing[:3])
            if unexpected:
                logger.warning("seed_%d unexpected keys: %d (%s...)",
                               seed, len(unexpected), unexpected[:3])
            model.to(self.device).eval()
            self.models.append(model)
        logger.info("Loaded %d ensemble members from %s on %s",
                    len(self.models), self.ckpt_dir, self.device)

    def _encode(self, cif_path: str, text: str):
        structure = Structure.from_file(cif_path)
        atom_fea_np, nbr_idx_np, nbr_fea_np, _ = build_crystal_graph(
            structure, radius=self.radius, max_neighbors=self.max_neighbors,
            gaussian=self.gaussian,
        )
        atom_fea = torch.tensor(atom_fea_np, dtype=torch.float32, device=self.device)
        nbr_fea = torch.tensor(nbr_fea_np, dtype=torch.float32, device=self.device)
        nbr_idx = torch.tensor(nbr_idx_np, dtype=torch.long, device=self.device)
        crystal_atom_idx = torch.zeros(atom_fea.shape[0], dtype=torch.long,
                                       device=self.device)

        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.text_max_seq_len, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        return atom_fea, nbr_fea, nbr_idx, crystal_atom_idx, input_ids, attention_mask

    @torch.inference_mode()
    def predict(self, cif_path: str, text: str) -> dict:
        """Return ensemble {band_gap_mean, band_gap_std, is_metal_prob}."""
        encoded = self._encode(cif_path, text)
        bg_preds = []
        metal_probs = []
        for model in self.models:
            bg, metal_logit = model(*encoded)
            bg_preds.append(bg.squeeze().detach().float().cpu().item())
            metal_probs.append(
                torch.sigmoid(metal_logit).squeeze().detach().float().cpu().item()
            )
        bg_arr = np.asarray(bg_preds)
        return {
            "band_gap_mean": float(bg_arr.mean()),
            "band_gap_std": float(bg_arr.std()),
            "is_metal_prob": float(np.mean(metal_probs)),
            "band_gap_per_seed": bg_preds,
            "is_metal_prob_per_seed": metal_probs,
        }
