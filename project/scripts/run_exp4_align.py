#!/usr/bin/env python3
"""
Exp 4 Stage 1: Contrastive alignment pretraining (CLIP-style).

Usage:
    python scripts/run_exp4_align.py
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_split
from src.data.crystal_graph import GaussianDistance, NUM_ATOM_FEATURES
from src.data.fusion_dataset import FusionDataset, collate_fusion
from src.models.alignment import (
    CLIPAlignmentModel, JointAlignmentModel,
    info_nce_loss, soft_info_nce_loss, compute_alignment_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description="Exp 4 Stage 1: Alignment")
    p.add_argument("--config", default="configs/exp4_alignment.yaml")
    return p.parse_args()


def load_pretrained_cgcnn(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.cgcnn.load_state_dict(state, strict=False)
    print(f"  [CGCNN ckpt] loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def load_pretrained_bert(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    bert_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
    print(f"  [BERT ckpt] loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def train_one_epoch(model, loader, optimizer, device, joint=False, alpha=0.5,
                    soft_contrastive=False, soft_margin=0.5):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    mse_fn = torch.nn.MSELoss()

    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, target) in loader:

        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        if joint:
            z_s, z_t, temperature, pred = model.forward_joint(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attention_mask)
        else:
            z_s, z_t, temperature = model(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attention_mask)

        # Contrastive loss
        if soft_contrastive:
            l_contrast = soft_info_nce_loss(z_s, z_t, target, temperature, soft_margin)
        else:
            l_contrast = info_nce_loss(z_s, z_t, temperature)

        # Joint: add regression loss
        if joint:
            l_reg = mse_fn(pred, target)
            loss = alpha * l_contrast + (1 - alpha) * l_reg
        else:
            loss = l_contrast

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics = compute_alignment_metrics(z_s, z_t)

        total_loss += loss.item()
        total_acc += metrics["mean_retrieval_acc"]
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_z_s, all_z_t = [], []
    for (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attention_mask, _target) in loader:

        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crys_idx = crys_idx.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        z_s, z_t, _ = model(atom_fea, nbr_fea, nbr_idx, crys_idx,
                             input_ids, attention_mask)
        all_z_s.append(z_s.cpu())
        all_z_t.append(z_t.cpu())

    z_s = torch.cat(all_z_s, dim=0)
    z_t = torch.cat(all_z_t, dim=0)

    # Compute metrics on full set
    metrics = compute_alignment_metrics(z_s, z_t)

    # Also compute loss
    temperature = model.temperature.detach().cpu()
    loss = info_nce_loss(z_s, z_t, temperature).item()
    metrics["loss"] = loss
    return metrics, z_s.numpy(), z_t.numpy()


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = PROJECT_ROOT / cfg["output_dir"] / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    acfg = cfg["alignment"]
    seed = acfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp4-align] Device: {device}")

    # Data
    df = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    print(f"[Exp4-align] Train={len(train_df)}, Val={len(val_df)}")

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

    bs = acfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, collate_fn=collate_fusion,
        num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bs, shuffle=False, collate_fn=collate_fusion, num_workers=0)

    # Model
    ccfg = cfg["cgcnn"]
    joint = acfg.get("joint", False)
    alpha = acfg.get("alpha", 0.5)
    soft_contrastive = acfg.get("soft_contrastive", False)
    soft_margin = acfg.get("soft_margin", 0.5)

    encoder_kwargs = dict(
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gaussian.dim,
        n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"],
        cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        bert_unfreeze_last_n=acfg["bert_unfreeze_last_n"],
        proj_dim=acfg["proj_dim"],
        temperature_init=acfg["temperature_init"],
    )

    if joint:
        model = JointAlignmentModel(**encoder_kwargs).to(device)
        print(f"[Exp4-align] Mode: JOINT (alpha={alpha}, soft={soft_contrastive})")
    else:
        model = CLIPAlignmentModel(**encoder_kwargs).to(device)
        print(f"[Exp4-align] Mode: pure contrastive")

    # Load pretrained encoders
    cgcnn_ckpt = acfg.get("cgcnn_ckpt")
    if cgcnn_ckpt:
        ckpt_path = PROJECT_ROOT / cgcnn_ckpt
        if ckpt_path.exists():
            print(f"[Exp4-align] Loading pretrained CGCNN")
            load_pretrained_cgcnn(model, ckpt_path, device)
    bert_ckpt = acfg.get("bert_ckpt")
    if bert_ckpt:
        ckpt_path = PROJECT_ROOT / bert_ckpt
        if ckpt_path.exists():
            print(f"[Exp4-align] Loading pretrained SciBERT")
            load_pretrained_bert(model, ckpt_path, device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Exp4-align] Total: {n_total:,}, Trainable: {n_trainable:,}")

    # Optimizer: separate LR for encoders vs projection/regression heads
    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["proj_struct", "proj_text", "log_temperature", "reg_head"]):
            head_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": acfg["lr"]},
        {"params": encoder_params, "lr": acfg["lr"] * acfg["encoder_lr_mult"]},
    ], weight_decay=acfg["weight_decay"])

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    patience = acfg["patience"]
    history = []

    t0 = time.time()
    for epoch in range(1, acfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            joint=joint, alpha=alpha,
            soft_contrastive=soft_contrastive, soft_margin=soft_margin)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        temp = model.temperature.item()
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_s2t_acc": round(val_metrics["s2t_acc"], 4),
            "val_t2s_acc": round(val_metrics["t2s_acc"], 4),
            "val_acc": round(val_metrics["mean_retrieval_acc"], 4),
            "val_pos_sim": round(val_metrics["mean_pos_sim"], 4),
            "temperature": round(temp, 4),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.3f} acc={train_acc:.3f} | "
                  f"val_acc={val_metrics['mean_retrieval_acc']:.3f} "
                  f"pos_sim={val_metrics['mean_pos_sim']:.3f} "
                  f"temp={temp:.4f}")

        if val_metrics["mean_retrieval_acc"] > best_val_acc:
            best_val_acc = val_metrics["mean_retrieval_acc"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"[Exp4-align] Training done in {elapsed:.1f}s")
    print(f"[Exp4-align] Best val retrieval acc: {best_val_acc:.4f}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    val_metrics, val_z_s, val_z_t = evaluate(model, val_loader, device)

    # Save
    results = {
        "experiment": "exp4_alignment_stage1",
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_params_total": n_total,
        "n_params_trainable": n_trainable,
        "training_time_s": round(elapsed, 1),
        "best_val_retrieval_acc": round(best_val_acc, 4),
        "final_val_metrics": {k: round(v, 4) for k, v in val_metrics.items()},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # Save embeddings for t-SNE visualization
    np.savez(out_dir / "val_embeddings.npz", z_s=val_z_s, z_t=val_z_t)

    print(f"[Exp4-align] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
