#!/usr/bin/env python3
"""
Exp 4 Stage 2: Downstream regression using aligned encoders.

Usage:
    python scripts/run_exp4_regress.py                     # concat mode (default)
    python scripts/run_exp4_regress.py --mode sum          # sum aligned embeddings
    python scripts/run_exp4_regress.py --mode film         # FiLM on aligned space
    python scripts/run_exp4_regress.py --freeze-encoders   # freeze aligned encoders
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
from src.models.alignment import CLIPAlignmentModel, AlignedRegressor
from src.evaluation.metrics import regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Exp 4 Stage 2: Regression")
    p.add_argument("--config", default="configs/exp4_alignment.yaml")
    p.add_argument("--mode", default="concat", choices=["concat", "sum", "film"])
    p.add_argument("--freeze-encoders", action="store_true",
                   help="Freeze aligned encoders, train regression head only.")
    return p.parse_args()


def train_one_epoch(encoder, regressor, loader, optimizer, criterion, device, freeze_enc):
    if freeze_enc:
        encoder.eval()
    else:
        encoder.train()
    regressor.train()

    total_loss = 0.0
    n = 0
    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:

        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        if freeze_enc:
            with torch.no_grad():
                z_s = encoder.encode_structure(atom_fea, nbr_fea, nbr_idx, crys_idx)
                z_t = encoder.encode_text(input_ids, attention_mask)
        else:
            z_s = encoder.encode_structure(atom_fea, nbr_fea, nbr_idx, crys_idx)
            z_t = encoder.encode_text(input_ids, attention_mask)

        pred = regressor(z_s, z_t).squeeze(-1)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.shape[0]
        n += target.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(encoder, regressor, loader, device):
    encoder.eval()
    regressor.eval()
    all_pred, all_true = [], []

    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:

        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        z_s = encoder.encode_structure(atom_fea, nbr_fea, nbr_idx, crys_idx)
        z_t = encoder.encode_text(input_ids, attention_mask)
        pred = regressor(z_s, z_t).squeeze(-1)

        all_pred.append(pred.cpu().numpy())
        all_true.append(target.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mode = args.mode
    freeze_enc = args.freeze_encoders
    tag = f"{mode}_{'frozen' if freeze_enc else 'finetune'}"
    out_dir = PROJECT_ROOT / cfg["output_dir"] / f"stage2_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["alignment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp4-reg-{tag}] Device: {device}")

    # Data
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")
    print(f"[Exp4-reg-{tag}] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    gcfg = cfg["graph"]
    gaussian = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])

    from transformers import AutoTokenizer
    tcfg = cfg["text"]
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    train_set = FusionDataset(train_df, target_col=cfg["target"],
                               radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                               gaussian=gaussian, model_name=tcfg["model_name"],
                               max_seq_len=tcfg["max_seq_len"], tokenizer=tokenizer)
    val_set = FusionDataset(val_df, target_col=cfg["target"],
                             radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                             gaussian=gaussian, model_name=tcfg["model_name"],
                             max_seq_len=tcfg["max_seq_len"], tokenizer=tokenizer)
    test_set = FusionDataset(test_df, target_col=cfg["target"],
                              radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                              gaussian=gaussian, model_name=tcfg["model_name"],
                              max_seq_len=tcfg["max_seq_len"], tokenizer=tokenizer)

    rcfg = cfg["regression"]
    bs = rcfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, collate_fn=collate_fusion, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)

    # Load aligned encoder
    ccfg = cfg["cgcnn"]
    acfg = cfg["alignment"]
    encoder = CLIPAlignmentModel(
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gaussian.dim,
        n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"],
        cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        bert_unfreeze_last_n=acfg["bert_unfreeze_last_n"],
        proj_dim=acfg["proj_dim"],
    ).to(device)

    align_ckpt = PROJECT_ROOT / cfg["output_dir"] / "stage1" / "best_model.pt"
    if align_ckpt.exists():
        print(f"[Exp4-reg-{tag}] Loading aligned encoder from {align_ckpt}")
        encoder.load_state_dict(torch.load(align_ckpt, map_location=device, weights_only=True))
    else:
        print(f"[WARN] Aligned checkpoint not found at {align_ckpt}, using random init!")

    if freeze_enc:
        for p in encoder.parameters():
            p.requires_grad = False
        print(f"[Exp4-reg-{tag}] Encoders frozen.")

    # Regression head
    regressor = AlignedRegressor(
        proj_dim=acfg["proj_dim"],
        hidden_dim=rcfg["hidden_dim"],
        dropout=rcfg["dropout"],
        mode=mode,
    ).to(device)

    # Count params
    enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    reg_trainable = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    print(f"[Exp4-reg-{tag}] Encoder trainable: {enc_trainable:,}, Head trainable: {reg_trainable:,}")

    # Optimizer
    all_params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + \
                 list(regressor.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=rcfg["lr"], weight_decay=rcfg["weight_decay"])
    criterion = nn.MSELoss()

    # Training
    best_val_mae = float("inf")
    patience_counter = 0
    patience = rcfg["patience"]
    history = []

    t0 = time.time()
    for epoch in range(1, rcfg["epochs"] + 1):
        train_loss = train_one_epoch(encoder, regressor, train_loader, optimizer,
                                      criterion, device, freeze_enc)

        val_true, val_pred = evaluate(encoder, regressor, val_loader, device)
        val_m = regression_metrics(val_true, val_pred)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_mae": round(val_m["mae"], 4),
            "val_rmse": round(val_m["rmse"], 4),
            "val_r2": round(val_m["r2"], 4),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"val MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} R2={val_m['r2']:.4f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            patience_counter = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "regressor": regressor.state_dict(),
            }, out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"[Exp4-reg-{tag}] Training done in {elapsed:.1f}s")

    # Test
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    regressor.load_state_dict(ckpt["regressor"])
    test_true, test_pred = evaluate(encoder, regressor, test_loader, device)
    test_m = regression_metrics(test_true, test_pred)

    print(f"\n[Exp4-reg-{tag}] Test Results:")
    print(f"  MAE  = {test_m['mae']:.4f} eV")
    print(f"  RMSE = {test_m['rmse']:.4f} eV")
    print(f"  R2   = {test_m['r2']:.4f}")

    # Save
    results = {
        "experiment": f"exp4_regression_{tag}",
        "mode": mode,
        "freeze_encoders": freeze_enc,
        "target": cfg["target"],
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "encoder_trainable": enc_trainable,
        "head_trainable": reg_trainable,
        "training_time_s": round(elapsed, 1),
        "best_val_mae": round(best_val_mae, 4),
        "test": test_m,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    pred_df = test_set.df[["material_id", "formula", cfg["target"]]].copy()
    pred_df["predicted"] = test_pred
    pred_df["error"] = test_pred - test_true
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"[Exp4-reg-{tag}] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
