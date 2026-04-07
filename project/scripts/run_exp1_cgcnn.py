#!/usr/bin/env python3
"""
Exp 1: CGCNN Baseline — Structure-only band gap prediction.

Usage:
    python scripts/run_exp1_cgcnn.py
    python scripts/run_exp1_cgcnn.py --config configs/exp1_cgcnn.yaml
"""

from __future__ import annotations

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
from src.data.crystal_graph import (
    CrystalGraphDataset,
    GaussianDistance,
    NUM_ATOM_FEATURES,
    collate_crystal_graphs,
)
from src.models.cgcnn import CGCNN
from src.evaluation.metrics import regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Exp 1: CGCNN baseline")
    p.add_argument("--config", default="configs/exp1_cgcnn.yaml")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for atom_fea, nbr_fea, nbr_idx, crys_idx, target in loader:
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        target = target.to(device)

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx).squeeze(-1)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.shape[0]
        n += target.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_true = []
    for atom_fea, nbr_fea, nbr_idx, crys_idx, target in loader:
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx).squeeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(target.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp1] Device: {device}")

    # Data
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]))
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")
    print(f"[Exp1] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    gcfg = cfg["graph"]
    gaussian = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])

    train_set = CrystalGraphDataset(train_df, target_col=cfg["target"],
                                     radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                                     gaussian=gaussian)
    val_set = CrystalGraphDataset(val_df, target_col=cfg["target"],
                                   radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                                   gaussian=gaussian)
    test_set = CrystalGraphDataset(test_df, target_col=cfg["target"],
                                    radius=gcfg["radius"], max_neighbors=gcfg["max_neighbors"],
                                    gaussian=gaussian)

    bs = cfg["train"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, collate_fn=collate_crystal_graphs, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bs, shuffle=False, collate_fn=collate_crystal_graphs, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, collate_fn=collate_crystal_graphs, num_workers=0)

    # Model
    mcfg = cfg["model"]
    model = CGCNN(
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=mcfg["atom_fea_len"],
        nbr_fea_len=gaussian.dim,
        n_conv=mcfg["n_conv"],
        h_fea_len=mcfg["h_fea_len"],
        n_h=mcfg["n_h"],
        output_dim=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Exp1] CGCNN params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["train"]["scheduler_patience"],
        factor=cfg["train"]["scheduler_factor"])
    criterion = nn.MSELoss()

    # Training loop
    best_val_mae = float("inf")
    patience_counter = 0
    patience = cfg["train"]["patience"]
    history = []

    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_true, val_pred = evaluate(model, val_loader, device)
        val_m = regression_metrics(val_true, val_pred)
        scheduler.step(val_m["mae"])

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_mae": round(val_m["mae"], 4),
            "val_rmse": round(val_m["rmse"], 4),
            "val_r2": round(val_m["r2"], 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"val MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} R2={val_m['r2']:.4f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"[Exp1] Training done in {elapsed:.1f}s")

    # Test evaluation
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_true, test_pred = evaluate(model, test_loader, device)
    test_m = regression_metrics(test_true, test_pred)

    print(f"\n[Exp1] Test Results:")
    print(f"  MAE  = {test_m['mae']:.4f} eV")
    print(f"  RMSE = {test_m['rmse']:.4f} eV")
    print(f"  R2   = {test_m['r2']:.4f}")

    # Save results
    results = {
        "experiment": "exp1_cgcnn",
        "target": cfg["target"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_params": n_params,
        "training_time_s": round(elapsed, 1),
        "best_val_mae": round(best_val_mae, 4),
        "test": test_m,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # Save predictions for error analysis
    pred_df = test_df[["material_id", "formula", cfg["target"]]].copy()
    pred_df["predicted"] = test_pred
    pred_df["error"] = test_pred - test_true
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"[Exp1] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
