#!/usr/bin/env python3
"""Dimensionality ablation for Exp 6 (cross-attention token fusion).

Replaces the SciBERT last_hidden_state token sequence with random Gaussian
noise of the same shape (B, seq_len, 768) to test whether the cross-attention
mechanism derives value from semantic token content or merely from having
per-token key/value vectors to attend over.

Unlike Exp 3 (which uses only the CLS token), Exp 6 feeds the full token
sequence as K/V in multi-head cross-attention against per-atom CGCNN queries.
The random tokens preserve the same attention-mask structure, so the cross-
attention computation is numerically identical except that K/V carry no
linguistic meaning.

CGCNN is always real and loaded from the Exp 1 checkpoint.
BERT is constructed (so kv_proj has identical input dim) but never called;
all BERT parameters are frozen.  Architecture, loss, optimizer, scheduler,
and early-stopping logic are identical to run_exp6_crossattn.py.

Usage:
    python scripts/run_dim_ablation_exp6.py --no-wandb
    python scripts/run_dim_ablation_exp6.py --seed 0 --no-wandb
    python scripts/run_dim_ablation_exp6.py --smoke --no-wandb
    python scripts/run_dim_ablation_exp6.py --text-embed-std 0.5 --no-wandb

Results saved to: results/exp_dim_ablation/exp6_rand_text/seed_<seed>/
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset, get_split
from src.data.crystal_graph import GaussianDistance, NUM_ATOM_FEATURES
from src.data.multitask_fusion_dataset import MultiTaskFusionDataset, collate_multitask_fusion
from src.models.crossattn_fusion import CrossAttnFusion, gather_atoms_per_crystal
from src.models.multitask_fusion import multitask_loss
from src.evaluation.metrics import regression_metrics


# ---------------------------------------------------------------------------
# Ablation model
# ---------------------------------------------------------------------------

class RandTextCrossAttn(CrossAttnFusion):
    """CrossAttnFusion with SciBERT token sequence replaced by Gaussian noise.

    The full token sequence (B, T, 768) used as K/V is replaced by i.i.d.
    N(0, text_embed_std^2) noise of the same shape, preserving the attention
    mask so padding positions are still masked.  The q_proj and kv_proj layers
    maintain identical parameter counts; only the input to kv_proj changes.
    """

    def __init__(self, text_embed_std: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_embed_std = text_embed_std

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        # 1. CGCNN node embeddings: (N_total, atom_fea_len) — always real
        node_emb = self.cgcnn_trunk.get_node_embedding(atom_fea, nbr_fea, nbr_idx)

        # 2. Random token sequence: same shape as BERT last_hidden_state (B, T, 768)
        bsz, seq_len = input_ids.shape
        token_emb = (torch.randn(bsz, seq_len, self.bert_dim, device=atom_fea.device)
                     * self.text_embed_std)

        # 3. Project to common attention dim
        Q_unpadded = self.q_proj(node_emb)   # (N_total, attn_dim)
        KV         = self.kv_proj(token_emb) # (B, T, attn_dim)

        # 4. Pad atoms-per-crystal into (B, max_atoms, attn_dim)
        batch_size = int(crystal_atom_idx.max().item()) + 1
        Q_padded, atom_mask = gather_atoms_per_crystal(Q_unpadded, crystal_atom_idx, batch_size)
        key_padding_mask = ~attention_mask.bool()  # mask STILL applied — same structure as real

        # 5. Cross-attention: each atom queries the (random) text token vectors
        attn_out, attn_weights = self.attn(
            Q_padded, KV, KV,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        attn_out = self.attn_norm(attn_out + Q_padded)

        # 6. Mean pool over valid atoms per crystal → (B, attn_dim)
        atom_mask_f = atom_mask.unsqueeze(-1).float()
        pooled = (attn_out * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1)

        # 7. Heads
        bandgap_pred = self.bandgap_head(pooled).squeeze(-1)
        metal_logit  = self.metal_head(pooled).squeeze(-1)

        if return_attention:
            return bandgap_pred, metal_logit, {
                "attn_weights": attn_weights,
                "atom_mask":    atom_mask,
                "token_mask":   attention_mask.bool(),
            }
        return bandgap_pred, metal_logit


# ---------------------------------------------------------------------------
# Helpers (identical to Exp 6)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dimensionality ablation for Exp 6 cross-attention fusion"
    )
    p.add_argument("--config", default="configs/exp6_crossattn.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--text-embed-std", type=float, default=1.0,
                   help="Std of pseudo text token embeddings (default 1.0 ≈ LayerNorm scale)")
    return p.parse_args()


def cosine_warmup_lr_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def load_pretrained_cgcnn(model: RandTextCrossAttn, cfg: dict, device: torch.device):
    """Load CGCNN trunk from exp1 checkpoint; skip BERT (not called in forward)."""
    train_cfg  = cfg["train"]
    cgcnn_ckpt = PROJECT_ROOT / train_cfg["cgcnn_ckpt"]

    if cgcnn_ckpt.exists():
        state = torch.load(cgcnn_ckpt, map_location=device, weights_only=True)
        miss, unexp = model.cgcnn_trunk.load_state_dict(state, strict=False)
        print(f"[ckpt] CGCNN trunk loaded — missing {len(miss)}, unexpected {len(unexp)}")
    else:
        print(f"[ckpt] CGCNN ckpt not found at {cgcnn_ckpt} — starting from scratch")
    print("[ckpt] BERT skipped (text ablated — not called in forward)")


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    grad_clip, alpha_metal, huber_delta, use_amp,
                    log_step_fn=None, epoch_idx: int = 0):
    model.train()
    n_total, l_total, l_reg_total, l_metal_total = 0, 0.0, 0.0, 0.0
    for batch in loader:
        (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attn_mask, target_bg, target_metal) = [
            b.to(device, non_blocking=True) for b in batch
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            bandgap_pred, metal_logit = model(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attn_mask,
            )
            total_loss, l_reg, l_metal = multitask_loss(
                bandgap_pred, metal_logit, target_bg, target_metal,
                alpha_metal=alpha_metal, huber_delta=huber_delta,
            )

        scaler.scale(total_loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bsz            = target_bg.shape[0]
        n_total        += bsz
        l_total        += total_loss.item() * bsz
        l_reg_total    += l_reg.item() * bsz
        l_metal_total  += l_metal.item() * bsz

        if log_step_fn is not None:
            log_step_fn({
                "train/step_loss_total": total_loss.item(),
                "train/step_loss_reg":   l_reg.item(),
                "train/step_loss_metal": l_metal.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch_idx,
            })

    return l_total / max(n_total, 1), l_reg_total / max(n_total, 1), l_metal_total / max(n_total, 1)


@torch.no_grad()
def evaluate(model, loader, device, alpha_metal, huber_delta, use_amp):
    model.eval()
    bg_preds, bg_trues, metal_preds, metal_trues = [], [], [], []
    n_total, l_total, l_reg_total, l_metal_total = 0, 0.0, 0.0, 0.0
    for batch in loader:
        (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attn_mask, target_bg, target_metal) = [
            b.to(device, non_blocking=True) for b in batch
        ]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            bandgap_pred, metal_logit = model(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attn_mask,
            )
            total_loss, l_reg, l_metal = multitask_loss(
                bandgap_pred, metal_logit, target_bg, target_metal,
                alpha_metal=alpha_metal, huber_delta=huber_delta,
            )
        bg_preds.append(bandgap_pred.float().cpu().numpy())
        bg_trues.append(target_bg.cpu().numpy())
        metal_preds.append(torch.sigmoid(metal_logit).float().cpu().numpy())
        metal_trues.append(target_metal.cpu().numpy())
        bsz           = target_bg.shape[0]
        n_total       += bsz
        l_total       += total_loss.item() * bsz
        l_reg_total   += l_reg.item() * bsz
        l_metal_total += l_metal.item() * bsz

    bg_preds    = np.concatenate(bg_preds)
    bg_trues    = np.concatenate(bg_trues)
    metal_preds = np.concatenate(metal_preds)
    metal_trues = np.concatenate(metal_trues)
    reg       = regression_metrics(bg_trues, bg_preds)
    metal_acc = float(((metal_preds > 0.5).astype(int) == metal_trues.astype(int)).mean())
    return {
        "loss_total":  l_total / max(n_total, 1),
        "loss_reg":    l_reg_total / max(n_total, 1),
        "loss_metal":  l_metal_total / max(n_total, 1),
        "mae": reg["mae"], "rmse": reg["rmse"], "r2": reg["r2"],
        "metal_acc": metal_acc,
    }, bg_preds, bg_trues, metal_preds, metal_trues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = PROJECT_ROOT / "results" / "exp_dim_ablation" / "exp6_rand_text" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    print(
        f"[DimAblation-Exp6] seed={seed} device={device} amp={use_amp} "
        f"smoke={args.smoke} text_embed_std={args.text_embed_std}"
    )

    use_wandb = cfg.get("wandb", {}).get("enabled", False) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="mmai-dim-ablation",
            entity=cfg["wandb"].get("entity"),
            name=f"exp6_rand_text_seed{seed}{'_smoke' if args.smoke else ''}",
            config={
                **cfg,
                "seed": seed,
                "smoke": args.smoke,
                "ablation": "rand_text",
                "text_embed_std": args.text_embed_std,
            },
            dir=str(out_dir),
        )
        log_step_fn = lambda d: wandb.log(d)
    else:
        log_step_fn = None

    df       = load_dataset(str(PROJECT_ROOT / cfg["dataset_path"]), text_ok_only=True)
    train_df = get_split(df, "train")
    val_df   = get_split(df, "val")
    test_df  = get_split(df, "test")
    if args.smoke:
        train_df = train_df.head(8).reset_index(drop=True)
        val_df   = val_df.head(4).reset_index(drop=True)
        test_df  = test_df.head(4).reset_index(drop=True)
    print(f"[DimAblation-Exp6] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    gcfg  = cfg["graph"]
    gauss = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])
    tcfg  = cfg["text"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    def make_set(d):
        return MultiTaskFusionDataset(
            d, target_col=cfg["target"], radius=gcfg["radius"],
            max_neighbors=gcfg["max_neighbors"], gaussian=gauss,
            model_name=tcfg["model_name"], max_seq_len=tcfg["max_seq_len"],
            tokenizer=tokenizer,
        )

    train_set, val_set, test_set = make_set(train_df), make_set(val_df), make_set(test_df)
    bs = 4 if args.smoke else cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              collate_fn=collate_multitask_fusion, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False,
                              collate_fn=collate_multitask_fusion, num_workers=0,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False,
                              collate_fn=collate_multitask_fusion, num_workers=0,
                              pin_memory=device.type == "cuda")

    ccfg = cfg["cgcnn"]
    xcfg = cfg["crossattn"]
    model = RandTextCrossAttn(
        text_embed_std=args.text_embed_std,
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gauss.dim,
        n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"],
        cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        bert_unfreeze_last_n=cfg["train"]["bert_unfreeze_last_n"],
        attn_dim=xcfg["attn_dim"],
        n_heads=xcfg["n_heads"],
        dropout=xcfg["dropout"],
        head_hidden_dim=xcfg["head_hidden_dim"],
        metal_head_hidden=xcfg["metal_head_hidden"],
    ).to(device)

    # Load pretrained CGCNN; BERT is skipped
    load_pretrained_cgcnn(model, cfg, device)

    # Freeze all BERT parameters — they receive no gradients (forward() overridden)
    for p in model.bert.parameters():
        p.requires_grad = False
    print("[DimAblation-Exp6] BERT fully frozen")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DimAblation-Exp6] params total={n_total:,}, trainable={n_train:,}")

    epochs = 1 if args.smoke else cfg["train"]["epochs"]
    optim  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps     = epochs * steps_per_epoch
    sched_cfg       = cfg["train"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=cosine_warmup_lr_lambda(total_steps, sched_cfg["warmup_ratio"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_mae     = float("inf")
    patience         = cfg["train"]["patience"]
    patience_counter = 0
    history          = []
    t0               = time.time()

    for ep in range(1, epochs + 1):
        train_total, train_reg, train_metal = train_one_epoch(
            model, train_loader, optim, scheduler, scaler, device,
            grad_clip=cfg["train"]["grad_clip_norm"],
            alpha_metal=cfg["loss"]["alpha_metal"],
            huber_delta=cfg["loss"]["huber_delta"],
            use_amp=use_amp,
            log_step_fn=log_step_fn,
            epoch_idx=ep,
        )
        val_m, *_ = evaluate(
            model, val_loader, device,
            alpha_metal=cfg["loss"]["alpha_metal"],
            huber_delta=cfg["loss"]["huber_delta"],
            use_amp=use_amp,
        )
        rec = {
            "epoch":             ep,
            "train_loss_total":  round(train_total, 6),
            "train_loss_reg":    round(train_reg, 6),
            "train_loss_metal":  round(train_metal, 6),
            "val_loss_total":    round(val_m["loss_total"], 6),
            "val_mae":           round(val_m["mae"], 4),
            "val_rmse":          round(val_m["rmse"], 4),
            "val_r2":            round(val_m["r2"], 4),
            "val_metal_acc":     round(val_m["metal_acc"], 4),
            "lr":                optim.param_groups[0]["lr"],
        }
        history.append(rec)
        if use_wandb:
            import wandb
            wandb.log({**{f"epoch/{k}": v for k, v in rec.items()}, "epoch": ep})
        if ep == 1 or ep % 5 == 0:
            print(f"  ep {ep:3d} | tr_loss={train_total:.4f} | "
                  f"val MAE={val_m['mae']:.4f} R2={val_m['r2']:.4f} "
                  f"metal_acc={val_m['metal_acc']:.3f}")
        if val_m["mae"] < best_val_mae:
            best_val_mae     = val_m["mae"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop @ epoch {ep}")
                break

    elapsed = time.time() - t0
    print(f"[DimAblation-Exp6] training done in {elapsed:.1f}s, best val MAE={best_val_mae:.4f}")

    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_m, test_bg_pred, test_bg_true, test_metal_pred, test_metal_true = evaluate(
        model, test_loader, device,
        alpha_metal=cfg["loss"]["alpha_metal"],
        huber_delta=cfg["loss"]["huber_delta"],
        use_amp=use_amp,
    )
    print(
        f"\n[DimAblation-Exp6] TEST seed={seed}: "
        f"MAE={test_m['mae']:.4f} RMSE={test_m['rmse']:.4f} "
        f"R2={test_m['r2']:.4f} metal_acc={test_m['metal_acc']:.3f}"
    )

    results = {
        "experiment":      "exp6_dim_ablation_rand_text",
        "base_experiment": "exp6_crossattn",
        "ablation":        "rand_text",
        "text_embed_std":  args.text_embed_std,
        "seed":            seed,
        "smoke":           args.smoke,
        "n_train":         len(train_set),
        "n_val":           len(val_set),
        "n_test":          len(test_set),
        "n_params_total":     n_total,
        "n_params_trainable": n_train,
        "training_time_s": round(elapsed, 1),
        "best_val_mae":    round(best_val_mae, 4),
        "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_m.items()},
        "config_summary": {
            "lr":          cfg["train"]["lr"],
            "batch_size":  bs,
            "epochs_max":  epochs,
            "patience":    patience,
            "huber_delta": cfg["loss"]["huber_delta"],
            "alpha_metal": cfg["loss"]["alpha_metal"],
            "attn_dim":    cfg["crossattn"]["attn_dim"],
            "n_heads":     cfg["crossattn"]["n_heads"],
            "dropout":     cfg["crossattn"]["dropout"],
            "amp":         use_amp,
            "grad_clip":   cfg["train"]["grad_clip_norm"],
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pred_df = test_set.df[["material_id", "formula", cfg["target"], "is_metal"]].copy()
    pred_df["bandgap_pred"] = test_bg_pred
    pred_df["bandgap_err"]  = test_bg_pred - test_bg_true
    pred_df["metal_prob"]   = test_metal_pred
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"[DimAblation-Exp6] saved → {out_dir}/")

    if use_wandb:
        import wandb
        wandb.summary["test_mae"]       = test_m["mae"]
        wandb.summary["test_r2"]        = test_m["r2"]
        wandb.summary["test_metal_acc"] = test_m["metal_acc"]
        wandb.finish()


if __name__ == "__main__":
    main()
