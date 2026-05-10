#!/usr/bin/env python3
"""Dimensionality ablation for Exp 3 (late fusion: concat / gated / FiLM).

Replaces the SciBERT CLS-token embedding with random Gaussian noise of the
same dimensionality (768) to test whether fusion performance gains come from
semantic text content or merely from the extra hidden dimensions it introduces.

The CGCNN encoder is always real and loaded from the Exp 1 checkpoint.
BERT is constructed (so projection layers have identical shapes) but never
called in forward; all BERT parameters are frozen.  Architecture, loss,
optimizer, and early-stopping logic are identical to run_exp3_fusion.py.

Usage:
    python scripts/run_dim_ablation_exp3.py                     # concat (default)
    python scripts/run_dim_ablation_exp3.py --fusion gated
    python scripts/run_dim_ablation_exp3.py --fusion film
    python scripts/run_dim_ablation_exp3.py --text-embed-std 0.5

Results saved to: results/exp_dim_ablation/exp3_rand_text/<fusion_type>/
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_split
from src.data.crystal_graph import GaussianDistance, NUM_ATOM_FEATURES
from src.data.fusion_dataset import FusionDataset, collate_fusion
from src.models.fusion import LateFusionConcat, LateFusionGated, LateFusionFiLM
from src.evaluation.metrics import regression_metrics


# ---------------------------------------------------------------------------
# Ablation model variants
# The mixin overrides encode() so that the BERT CLS token is replaced by
# i.i.d. Gaussian noise. All three fusion strategies share the same override.
# ---------------------------------------------------------------------------

class _RandTextEncodeMixin:
    """Replace BERT CLS output with Gaussian noise of matching dimension (768).

    self._text_embed_std controls the noise scale (set after construction).
    self.bert_out_dim (768) and self.cgcnn.get_embedding() are inherited from
    _DualEncoder and remain unchanged.
    """

    def encode(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
               input_ids, attention_mask):
        h_struct = self.cgcnn.get_embedding(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        # Same shape as a real SciBERT CLS token: (B, 768)
        h_text = (torch.randn(input_ids.shape[0], self.bert_out_dim, device=atom_fea.device)
                  * self._text_embed_std)
        return h_struct, h_text


class RandTextConcat(_RandTextEncodeMixin, LateFusionConcat):
    """Concat fusion with random text embeddings."""


class RandTextGated(_RandTextEncodeMixin, LateFusionGated):
    """Gated fusion with random text embeddings."""


class RandTextFiLM(_RandTextEncodeMixin, LateFusionFiLM):
    """FiLM fusion with random text embeddings."""


_ABLATION_MODELS = {
    "concat": RandTextConcat,
    "gated":  RandTextGated,
    "film":   RandTextFiLM,
}


# ---------------------------------------------------------------------------
# Helpers (mostly identical to Exp 3)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dimensionality ablation for Exp 3 late-fusion models"
    )
    p.add_argument("--config", default="configs/exp3_fusion.yaml")
    p.add_argument("--fusion", default="concat", choices=["concat", "gated", "film"],
                   help="Fusion strategy: concat, gated, or film.")
    p.add_argument("--text-embed-std", type=float, default=1.0,
                   help="Std of pseudo text embeddings (default 1.0 ≈ LayerNorm scale)")
    return p.parse_args()


def load_pretrained_cgcnn(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.cgcnn.load_state_dict(state, strict=False)
    print(f"  [CGCNN ckpt] loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:
        atom_fea       = atom_fea.to(device)
        nbr_fea        = nbr_fea.to(device)
        nbr_idx        = nbr_idx.to(device)
        crys_idx       = crys_idx.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target         = target.to(device)

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx,
                     input_ids, attention_mask).squeeze(-1)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.shape[0]
        n          += target.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:
        atom_fea       = atom_fea.to(device)
        nbr_fea        = nbr_fea.to(device)
        nbr_idx        = nbr_idx.to(device)
        crys_idx       = crys_idx.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx,
                     input_ids, attention_mask).squeeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(target.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    fusion_type = args.fusion
    out_dir = PROJECT_ROOT / "results" / "exp_dim_ablation" / "exp3_rand_text" / fusion_type
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[DimAblation-Exp3-{fusion_type}] device={device} "
        f"text_embed_std={args.text_embed_std}"
    )

    df       = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df   = get_split(df, "val")
    test_df  = get_split(df, "test")
    print(f"[DimAblation-Exp3-{fusion_type}] Train={len(train_df)}, Val={len(val_df)}, "
          f"Test={len(test_df)}")

    gcfg     = cfg["graph"]
    gaussian = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])

    from transformers import AutoTokenizer
    tcfg      = cfg["text"]
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    make = lambda d: FusionDataset(
        d, target_col=cfg["target"],
        radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
        gaussian=gaussian, model_name=tcfg["model_name"],
        max_seq_len=tcfg["max_seq_len"], tokenizer=tokenizer,
    )
    train_set, val_set, test_set = make(train_df), make(val_df), make(test_df)

    bs           = cfg["train"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, collate_fn=collate_fusion, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_set,   batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)

    ccfg  = cfg["cgcnn"]
    fcfg  = cfg["fusion"]
    print(f"[DimAblation-Exp3-{fusion_type}] Building ablation model: {fusion_type}")
    model = _ABLATION_MODELS[fusion_type](
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gaussian.dim,
        n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"],
        cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        freeze_bert=cfg["train"]["freeze_bert"],
        bert_unfreeze_last_n=cfg["train"]["bert_unfreeze_last_n"],
        fusion_hidden_dim=fcfg["hidden_dim"],
        fusion_dropout=fcfg["dropout"],
    ).to(device)

    # Noise scale for the mixin's encode()
    model._text_embed_std = args.text_embed_std

    # Load pretrained CGCNN (structure encoder is real); skip BERT entirely
    cgcnn_ckpt = cfg["train"].get("cgcnn_ckpt")
    if cgcnn_ckpt:
        ckpt_path = PROJECT_ROOT / cgcnn_ckpt
        if ckpt_path.exists():
            print(f"[DimAblation-Exp3-{fusion_type}] Loading pretrained CGCNN from {cgcnn_ckpt}")
            load_pretrained_cgcnn(model, ckpt_path, device)
    print("[DimAblation-Exp3] BERT skipped (not called in forward)")

    # Freeze all BERT parameters — they receive no gradients (encode() is overridden)
    for p in model.bert.parameters():
        p.requires_grad = False
    print("[DimAblation-Exp3] BERT fully frozen")

    # Freeze CGCNN if configured
    if cfg["train"]["freeze_cgcnn"]:
        model.freeze_cgcnn()
        print(f"[DimAblation-Exp3-{fusion_type}] CGCNN encoder frozen.")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DimAblation-Exp3-{fusion_type}] Total params: {n_total:,}, "
          f"Trainable: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    best_val_mae     = float("inf")
    patience_counter = 0
    patience         = cfg["train"]["patience"]
    history          = []

    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_true, val_pred = evaluate(model, val_loader, device)
        val_m = regression_metrics(val_true, val_pred)

        history.append({
            "epoch":     epoch,
            "train_loss": round(train_loss, 6),
            "val_mae":    round(val_m["mae"], 4),
            "val_rmse":   round(val_m["rmse"], 4),
            "val_r2":     round(val_m["r2"], 4),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"val MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} R2={val_m['r2']:.4f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae     = val_m["mae"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"[DimAblation-Exp3-{fusion_type}] Training done in {elapsed:.1f}s")

    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_true, test_pred = evaluate(model, test_loader, device)
    test_m = regression_metrics(test_true, test_pred)

    print(f"\n[DimAblation-Exp3-{fusion_type}] Test Results:")
    print(f"  MAE  = {test_m['mae']:.4f} eV")
    print(f"  RMSE = {test_m['rmse']:.4f} eV")
    print(f"  R2   = {test_m['r2']:.4f}")

    results = {
        "experiment":      f"exp3_dim_ablation_rand_text_{fusion_type}",
        "base_experiment": f"exp3_fusion_{fusion_type}",
        "ablation":        "rand_text",
        "text_embed_std":  args.text_embed_std,
        "fusion_type":     fusion_type,
        "target":          cfg["target"],
        "n_train":         len(train_set),
        "n_val":           len(val_set),
        "n_test":          len(test_set),
        "n_params_total":      n_total,
        "n_params_trainable":  n_trainable,
        "freeze_cgcnn":    cfg["train"]["freeze_cgcnn"],
        "training_time_s": round(elapsed, 1),
        "best_val_mae":    round(best_val_mae, 4),
        "test":            test_m,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    pred_df = test_set.df[["material_id", "formula", cfg["target"]]].copy()
    pred_df["predicted"] = test_pred
    pred_df["error"]     = test_pred - test_true
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"[DimAblation-Exp3-{fusion_type}] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
