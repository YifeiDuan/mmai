#!/usr/bin/env python3
"""Visualize Exp 6 cross-attention: per-atom attention over per-token text.

For a few representative test crystals, plot a heatmap with rows = atoms
(labelled by element) and columns = (selected) text tokens. Saves PNGs
to results/figures/crossattn/<material_id>.png.
"""

from __future__ import annotations

import argparse
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pymatgen.core import Structure
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_split
from src.data.crystal_graph import GaussianDistance, NUM_ATOM_FEATURES
from src.data.multitask_fusion_dataset import MultiTaskFusionDataset, collate_multitask_fusion
from src.models.crossattn_fusion import CrossAttnFusion


SELECTED_MIDS_DEFAULT = [
    "mp-3731",      # LiNbO3 — wide-gap insulator (3.34 eV)
    "mp-1076800",   # BaCuO3 — metal (0 eV)
    "mp-1019544",   # BaZrO3 — moderate gap
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp6_crossattn.yaml")
    p.add_argument("--ckpt", default="results/exp6_crossattn/seed_42/best_model.pt")
    p.add_argument("--out-dir", default="results/figures/crossattn")
    p.add_argument("--mids", nargs="*", default=SELECTED_MIDS_DEFAULT)
    p.add_argument("--top-k-tokens", type=int, default=20,
                   help="Show only this many highest-attention tokens per atom group.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(PROJECT_ROOT / args.config))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test set + restrict to selected materials
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    test_df = get_split(df, "test")
    selected_mids = [m for m in args.mids if m in test_df["material_id"].values]
    if not selected_mids:
        # Fall back to first 3 test materials
        selected_mids = test_df["material_id"].head(3).tolist()
        print(f"[viz] Default mids not in test split, falling back to: {selected_mids}")
    selected_df = test_df[test_df["material_id"].isin(selected_mids)].reset_index(drop=True)

    gcfg = cfg["graph"]
    gauss = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])
    tcfg = cfg["text"]
    xcfg = cfg["crossattn"]
    ccfg = cfg["cgcnn"]
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    ds = MultiTaskFusionDataset(
        selected_df, target_col=cfg["target"], radius=gcfg["radius"],
        max_neighbors=gcfg["max_neighbors"], gaussian=gauss,
        model_name=tcfg["model_name"], max_seq_len=tcfg["max_seq_len"],
        tokenizer=tokenizer,
    )
    loader = DataLoader(ds, batch_size=len(ds), collate_fn=collate_multitask_fusion)
    batch = next(iter(loader))
    (atom_fea, nbr_fea, nbr_idx, crys_idx,
     input_ids, attention_mask, target_bg, target_metal) = [b.to(device) for b in batch]

    model = CrossAttnFusion(
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gauss.dim, n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"], cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        bert_unfreeze_last_n=cfg["train"]["bert_unfreeze_last_n"],
        attn_dim=xcfg["attn_dim"], n_heads=xcfg["n_heads"],
        dropout=xcfg["dropout"], head_hidden_dim=xcfg["head_hidden_dim"],
        metal_head_hidden=xcfg["metal_head_hidden"],
    ).to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / args.ckpt, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        bg_pred, metal_logit, attn = model(
            atom_fea, nbr_fea, nbr_idx, crys_idx,
            input_ids, attention_mask, return_attention=True,
        )
    weights = attn["attn_weights"].cpu().numpy()    # (B, max_atoms, T)
    atom_mask = attn["atom_mask"].cpu().numpy()     # (B, max_atoms)
    token_mask = attn["token_mask"].cpu().numpy()   # (B, T)

    for b, mid in enumerate(selected_df["material_id"]):
        n_atoms = int(atom_mask[b].sum())
        n_tokens = int(token_mask[b].sum())
        W = weights[b, :n_atoms, :n_tokens]            # (n_atoms, n_tokens)

        # Atom labels via pymatgen
        cif_path = PROJECT_ROOT / selected_df.iloc[b]["cif_path"]
        struct = Structure.from_file(cif_path)
        atom_labels = [str(site.specie) for site in struct.sites][:n_atoms]
        # If lengths differ, fallback labels
        if len(atom_labels) != n_atoms:
            atom_labels = [f"atom{i}" for i in range(n_atoms)]

        # Token strings
        token_ids = input_ids[b, :n_tokens].cpu().tolist()
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

        # Pick top-K most-attended tokens (across atoms) for readability
        avg_attn_per_token = W.mean(axis=0)
        top_idx = np.argsort(avg_attn_per_token)[-args.top_k_tokens:][::-1]
        W_sub = W[:, top_idx]
        token_sub = [token_strs[i] for i in top_idx]

        true_bg = float(selected_df.iloc[b]["band_gap"])
        is_metal = bool(selected_df.iloc[b]["is_metal"])
        pred_bg = float(bg_pred[b].cpu().item())
        formula = selected_df.iloc[b]["formula"]

        fig, ax = plt.subplots(figsize=(max(8, args.top_k_tokens * 0.4), max(4, n_atoms * 0.3)))
        im = ax.imshow(W_sub, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(token_sub)))
        ax.set_xticklabels(token_sub, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(np.arange(n_atoms))
        ax.set_yticklabels(atom_labels, fontsize=8)
        ax.set_title(f"{mid}  {formula}  band_gap_true={true_bg:.2f} (metal={is_metal})  pred={pred_bg:.2f}\n"
                     f"top-{args.top_k_tokens} tokens (head-averaged cross-attention)",
                     fontsize=10)
        ax.set_xlabel("text tokens")
        ax.set_ylabel("crystal atoms")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout()
        out_path = out_dir / f"{mid}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"Done — {len(selected_df)} heatmaps under {out_dir}/")


if __name__ == "__main__":
    main()
