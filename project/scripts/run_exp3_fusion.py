#!/usr/bin/env python3
"""
Exp 3: Late Fusion — CGCNN + SciBERT multimodal band gap prediction.

Usage:
    python scripts/run_exp3_fusion.py                          # concat (default)
    python scripts/run_exp3_fusion.py --fusion gated           # gated fusion
    python scripts/run_exp3_fusion.py --fusion film            # FiLM fusion
    python scripts/run_exp3_fusion.py --no-pretrained          # no pretrained encoders
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
from src.models.fusion import build_fusion_model
from src.evaluation.metrics import regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Exp 3: Late Fusion")
    p.add_argument("--config", default="configs/exp3_fusion.yaml")
    p.add_argument("--fusion", default="concat", choices=["concat", "gated", "film"],
                   help="Fusion strategy: concat, gated, or film.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not load pretrained Exp1/Exp2 encoder weights.")
    return p.parse_args()


def load_pretrained_cgcnn(model, ckpt_path, device):
    """Load Exp 1 CGCNN weights into the fusion model's cgcnn sub-module."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Map keys: Exp 1 saves CGCNN directly; fusion model has cgcnn.xxx
    cgcnn_state = {}
    for k, v in state.items():
        cgcnn_state[k] = v
    missing, unexpected = model.cgcnn.load_state_dict(cgcnn_state, strict=False)
    print(f"  [CGCNN ckpt] loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def load_pretrained_bert(model, ckpt_path, device):
    """Load Exp 2 SciBERT weights into the fusion model's bert sub-module."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Exp 2 saves SciBERTRegressor with keys: bert.xxx and head.xxx
    bert_state = {}
    for k, v in state.items():
        if k.startswith("bert."):
            bert_state[k[len("bert."):]] = v
    missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
    print(f"  [BERT ckpt] loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
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

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx,
                     input_ids, attention_mask).squeeze(-1)
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
    all_pred, all_true = [], []
    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        pred = model(atom_fea, nbr_fea, nbr_idx, crys_idx,
                     input_ids, attention_mask).squeeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(target.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    fusion_type = args.fusion
    out_dir = PROJECT_ROOT / cfg["output_dir"] / fusion_type
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp3-{fusion_type}] Device: {device}")

    # Data (text_ok only)
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")
    print(f"[Exp3-{fusion_type}] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

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

    bs = cfg["train"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, collate_fn=collate_fusion, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)

    # Model
    ccfg = cfg["cgcnn"]
    fcfg = cfg["fusion"]
    print(f"[Exp3-{fusion_type}] Building fusion model: {fusion_type}")
    model = build_fusion_model(
        fusion_type=fusion_type,
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

    # Load pretrained encoder weights
    if not args.no_pretrained:
        cgcnn_ckpt = cfg["train"].get("cgcnn_ckpt")
        bert_ckpt = cfg["train"].get("bert_ckpt")
        if cgcnn_ckpt:
            ckpt_path = PROJECT_ROOT / cgcnn_ckpt
            if ckpt_path.exists():
                print(f"[Exp3-{fusion_type}] Loading pretrained CGCNN from {cgcnn_ckpt}")
                load_pretrained_cgcnn(model, ckpt_path, device)
        if bert_ckpt:
            ckpt_path = PROJECT_ROOT / bert_ckpt
            if ckpt_path.exists():
                print(f"[Exp3-{fusion_type}] Loading pretrained SciBERT from {bert_ckpt}")
                load_pretrained_bert(model, ckpt_path, device)

    # Freeze CGCNN if configured
    if cfg["train"]["freeze_cgcnn"]:
        model.freeze_cgcnn()
        print(f"[Exp3-{fusion_type}] CGCNN encoder frozen.")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Exp3-{fusion_type}] Total params: {n_total:,}, Trainable: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    # Training
    best_val_mae = float("inf")
    patience_counter = 0
    patience = cfg["train"]["patience"]
    history = []

    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_true, val_pred = evaluate(model, val_loader, device)
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
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"[Exp3-{fusion_type}] Training done in {elapsed:.1f}s")

    # Test
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_true, test_pred = evaluate(model, test_loader, device)
    test_m = regression_metrics(test_true, test_pred)

    print(f"\n[Exp3-{fusion_type}] Test Results:")
    print(f"  MAE  = {test_m['mae']:.4f} eV")
    print(f"  RMSE = {test_m['rmse']:.4f} eV")
    print(f"  R2   = {test_m['r2']:.4f}")

    # Save
    results = {
        "experiment": f"exp3_fusion_{fusion_type}",
        "fusion_type": fusion_type,
        "target": cfg["target"],
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params_total": n_total,
        "n_params_trainable": n_trainable,
        "pretrained_encoders": not args.no_pretrained,
        "freeze_cgcnn": cfg["train"]["freeze_cgcnn"],
        "freeze_bert": cfg["train"]["freeze_bert"],
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

    print(f"[Exp3-{fusion_type}] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
