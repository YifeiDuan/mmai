#!/usr/bin/env python3
"""
Exp 2: SciBERT Baseline — Text-only band gap prediction.

Usage:
    python scripts/run_exp2_scibert.py
    python scripts/run_exp2_scibert.py --freeze   # frozen encoder, train MLP head only
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
from src.data.text_dataset import TextRegressionDataset
from src.models.scibert_regressor import SciBERTRegressor
from src.evaluation.metrics import regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Exp 2: SciBERT baseline")
    p.add_argument("--config", default="configs/exp2_scibert.yaml")
    p.add_argument("--freeze", action="store_true",
                   help="Override config: freeze entire BERT, train MLP head only.")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device).squeeze(-1)

        pred = model(input_ids, attention_mask).squeeze(-1)
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
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].squeeze(-1)

        pred = model(input_ids, attention_mask).squeeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(target.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    freeze_bert = args.freeze or cfg["train"]["freeze_bert"]
    tag = "frozen" if freeze_bert else "finetune"
    out_dir = PROJECT_ROOT / cfg["output_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp2-{tag}] Device: {device}")

    # Data — text_ok only
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")
    print(f"[Exp2-{tag}] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    tcfg = cfg["text"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    train_set = TextRegressionDataset(train_df, target_col=cfg["target"],
                                       model_name=tcfg["model_name"],
                                       max_seq_len=tcfg["max_seq_len"],
                                       tokenizer=tokenizer)
    val_set = TextRegressionDataset(val_df, target_col=cfg["target"],
                                     model_name=tcfg["model_name"],
                                     max_seq_len=tcfg["max_seq_len"],
                                     tokenizer=tokenizer)
    test_set = TextRegressionDataset(test_df, target_col=cfg["target"],
                                      model_name=tcfg["model_name"],
                                      max_seq_len=tcfg["max_seq_len"],
                                      tokenizer=tokenizer)

    bs = cfg["train"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0)

    # Model
    mcfg = cfg["model"]
    model = SciBERTRegressor(
        model_name=tcfg["model_name"],
        hidden_dim=mcfg["hidden_dim"],
        dropout=mcfg["dropout"],
        freeze_bert=freeze_bert,
        unfreeze_last_n_layers=cfg["train"]["unfreeze_last_n_layers"],
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Exp2-{tag}] Total params: {n_params_total:,}, Trainable: {n_params_trainable:,}")

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
    print(f"[Exp2-{tag}] Training done in {elapsed:.1f}s")

    # Test
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_true, test_pred = evaluate(model, test_loader, device)
    test_m = regression_metrics(test_true, test_pred)

    print(f"\n[Exp2-{tag}] Test Results:")
    print(f"  MAE  = {test_m['mae']:.4f} eV")
    print(f"  RMSE = {test_m['rmse']:.4f} eV")
    print(f"  R2   = {test_m['r2']:.4f}")

    # Save
    results = {
        "experiment": f"exp2_scibert_{tag}",
        "target": cfg["target"],
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_params_total": n_params_total,
        "n_params_trainable": n_params_trainable,
        "freeze_bert": freeze_bert,
        "training_time_s": round(elapsed, 1),
        "best_val_mae": round(best_val_mae, 4),
        "test": test_m,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # Save predictions
    pred_df = test_set.df[["material_id", "formula", cfg["target"]]].copy()
    pred_df["predicted"] = test_pred
    pred_df["error"] = test_pred - test_true
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"[Exp2-{tag}] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
