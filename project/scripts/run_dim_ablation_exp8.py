#!/usr/bin/env python3
"""Dimensionality ablation for Exp 8 (3-modality fusion).

Replaces real SciBERT text embeddings and/or ResNet18 image embeddings with
random Gaussian noise of identical dimensionality to isolate whether model
performance gains come from semantic content or from the additional capacity
introduced by the extra hidden dimensions.

Ablation modes (CLI flags, default: both ablated):
  --ablate-text      SciBERT CLS token → N(0, text_embed_std)    [dim 768]
  --no-ablate-text   Use real SciBERT embeddings
  --ablate-image     ResNet18 projection → N(0, image_embed_std)  [dim  64]
  --no-ablate-image  Use real ResNet18 embeddings
  --text-embed-std   Std of pseudo text noise  (default 1.0)
  --image-embed-std  Std of pseudo image noise (default 1.0)

Structure (CGCNN) is always real. Architecture, loss, optimizer, and
scheduler are identical to run_exp8_three_modality.py so results are
directly comparable.

Results are saved to:
  results/exp_dim_ablation/<mode>/seed_<seed>/
where <mode> encodes which modalities are ablated, e.g. rand_text_rand_image.
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
from src.data.three_modality_dataset import ThreeModalityFusionDataset, collate_three_modality
from src.models.three_modality_fusion import ThreeModalityFusion
from src.models.multitask_fusion import multitask_loss
from src.evaluation.metrics import regression_metrics


# ---------------------------------------------------------------------------
# Dimensionality-ablation model
# ---------------------------------------------------------------------------

class DimAblationFusion(ThreeModalityFusion):
    """ThreeModalityFusion with text/image encoders optionally bypassed by noise.

    When ablating a modality, the corresponding encoder is not called and its
    output is replaced by an i.i.d. Gaussian tensor with the same shape.
    Projection layers that receive the random embeddings (proj_text, film_gen
    for text; image_encoder.proj for image) retain identical dimensions and
    parameter counts to the full model, so this is a pure content ablation.

    Random text embedding: N(0, text_embed_std^2), shape (B, 768)
    Random image embedding: N(0, image_embed_std^2), shape (B, image_dim)

    Value ranges: std=1.0 is chosen to match the approximate scale of
    layer-normalised BERT hidden states and projected ResNet18 features.
    Adjust via --text-embed-std / --image-embed-std if needed.
    """

    def __init__(
        self,
        ablate_text: bool = True,
        ablate_image: bool = True,
        text_embed_std: float = 1.0,
        image_embed_std: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ablate_text = ablate_text
        self.ablate_image = ablate_image
        self.text_embed_std = text_embed_std
        self.image_embed_std = image_embed_std

    def forward(
        self,
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
        input_ids, attention_mask,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Structure branch: always real CGCNN
        h_s = self.cgcnn.get_embedding(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        bsz = input_ids.shape[0]
        dev = atom_fea.device

        # Text branch: real BERT CLS token or fresh i.i.d. Gaussian noise
        if self.ablate_text:
            # Same dimensionality as SciBERT hidden size (768) but semantically empty
            h_t = torch.randn(bsz, self.bert_out_dim, device=dev) * self.text_embed_std
        else:
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            h_t = bert_out.last_hidden_state[:, 0, :]  # CLS token

        # FiLM: h_t (real or noise) modulates projected structure features
        h_s_proj = self.proj_struct(h_s)
        h_t_proj = self.proj_text(h_t)

        film_params = self.film_gen(h_t)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma + 1.0          # centre around identity transform
        h_modulated = gamma * h_s_proj + beta
        h_modulated = self.mod_norm(h_modulated) + h_s_proj  # residual

        # Image branch: real ResNet18 projection or fresh i.i.d. Gaussian noise
        if self.ablate_image:
            # Same dimensionality as image_encoder output (64) but semantically empty
            h_img = torch.randn(bsz, self.image_dim, device=dev) * self.image_embed_std
        else:
            h_img = self.image_encoder(images)

        h_fused = torch.cat([h_modulated, h_t_proj, h_img], dim=1)
        bandgap_pred = self.head(h_fused).squeeze(-1)
        metal_logit  = self.metal_head(h_fused).squeeze(-1)
        return bandgap_pred, metal_logit


# ---------------------------------------------------------------------------
# Helpers (identical to Exp 8)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dimensionality ablation for Exp 8 three-modality fusion"
    )
    p.add_argument("--config", default="configs/exp8_three_modality.yaml",
                   help="YAML config to inherit (same as Exp 8)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")

    # Ablation toggles
    p.add_argument("--ablate-text",  dest="ablate_text",
                   action="store_true", default=True,
                   help="Replace SciBERT embeddings with Gaussian noise (default: True)")
    p.add_argument("--no-ablate-text", dest="ablate_text",
                   action="store_false",
                   help="Use real SciBERT embeddings")
    p.add_argument("--ablate-image", dest="ablate_image",
                   action="store_true", default=True,
                   help="Replace ResNet18 embeddings with Gaussian noise (default: True)")
    p.add_argument("--no-ablate-image", dest="ablate_image",
                   action="store_false",
                   help="Use real ResNet18 embeddings")

    # Noise magnitude
    p.add_argument("--text-embed-std", type=float, default=1.0,
                   help="Std of pseudo text embeddings (default 1.0 ≈ LayerNorm scale)")
    p.add_argument("--image-embed-std", type=float, default=1.0,
                   help="Std of pseudo image embeddings (default 1.0)")

    return p.parse_args()


def ablation_mode_label(ablate_text: bool, ablate_image: bool) -> str:
    """Human-readable tag for the output subdirectory."""
    parts = []
    if ablate_text:
        parts.append("rand_text")
    if ablate_image:
        parts.append("rand_image")
    if not parts:
        parts.append("no_ablation")
    return "_".join(parts)


def cosine_warmup_lr_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def load_pretrained(model: DimAblationFusion, cfg: dict, device: torch.device,
                    load_bert: bool = True):
    """Load CGCNN from exp1 and (optionally) BERT from exp2 checkpoints."""
    train_cfg = cfg["train"]
    cgcnn_ckpt = PROJECT_ROOT / train_cfg["cgcnn_ckpt"]
    bert_ckpt  = PROJECT_ROOT / train_cfg["bert_ckpt"]

    if cgcnn_ckpt.exists():
        state = torch.load(cgcnn_ckpt, map_location=device, weights_only=True)
        miss, unexp = model.cgcnn.load_state_dict(state, strict=False)
        print(f"[ckpt] CGCNN loaded — missing {len(miss)}, unexpected {len(unexp)}")
    if load_bert and bert_ckpt.exists():
        state = torch.load(bert_ckpt, map_location=device, weights_only=True)
        bert_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
        miss, unexp = model.bert.load_state_dict(bert_state, strict=False)
        print(f"[ckpt] BERT loaded — missing {len(miss)}, unexpected {len(unexp)}")
    elif not load_bert:
        print("[ckpt] BERT skipped (text ablated — BERT not called in forward)")


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    grad_clip, alpha_metal, huber_delta, use_amp,
                    log_step_fn=None, epoch_idx: int = 0):
    model.train()
    n_total, l_total, l_reg_total, l_metal_total = 0, 0.0, 0.0, 0.0
    for batch in loader:
        (atom_fea, nbr_fea, nbr_idx, crys_idx,
         input_ids, attn_mask, target_bg, target_metal,
         images) = [b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            bandgap_pred, metal_logit = model(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attn_mask, images,
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

        bsz = target_bg.shape[0]
        n_total      += bsz
        l_total      += total_loss.item() * bsz
        l_reg_total  += l_reg.item() * bsz
        l_metal_total += l_metal.item() * bsz

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
         input_ids, attn_mask, target_bg, target_metal,
         images) = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            bandgap_pred, metal_logit = model(
                atom_fea, nbr_fea, nbr_idx, crys_idx, input_ids, attn_mask, images,
            )
            total_loss, l_reg, l_metal = multitask_loss(
                bandgap_pred, metal_logit, target_bg, target_metal,
                alpha_metal=alpha_metal, huber_delta=huber_delta,
            )
        bg_preds.append(bandgap_pred.float().cpu().numpy())
        bg_trues.append(target_bg.cpu().numpy())
        metal_preds.append(torch.sigmoid(metal_logit).float().cpu().numpy())
        metal_trues.append(target_metal.cpu().numpy())
        bsz = target_bg.shape[0]
        n_total       += bsz
        l_total       += total_loss.item() * bsz
        l_reg_total   += l_reg.item() * bsz
        l_metal_total += l_metal.item() * bsz

    bg_preds    = np.concatenate(bg_preds)
    bg_trues    = np.concatenate(bg_trues)
    metal_preds = np.concatenate(metal_preds)
    metal_trues = np.concatenate(metal_trues)
    reg = regression_metrics(bg_trues, bg_preds)
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
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mode = ablation_mode_label(args.ablate_text, args.ablate_image)
    out_dir = PROJECT_ROOT / "results" / "exp_dim_ablation" / mode / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    print(
        f"[DimAblation] mode={mode} seed={seed} device={device} amp={use_amp} smoke={args.smoke}\n"
        f"              ablate_text={args.ablate_text} (std={args.text_embed_std})"
        f"  ablate_image={args.ablate_image} (std={args.image_embed_std})"
    )

    use_wandb = cfg.get("wandb", {}).get("enabled", False) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="mmai-dim-ablation",
            entity=cfg["wandb"].get("entity"),
            name=f"{mode}_seed{seed}{'_smoke' if args.smoke else ''}",
            config={
                **cfg,
                "seed": seed,
                "smoke": args.smoke,
                "ablate_text": args.ablate_text,
                "ablate_image": args.ablate_image,
                "text_embed_std": args.text_embed_std,
                "image_embed_std": args.image_embed_std,
                "ablation_mode": mode,
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
    print(f"[DimAblation] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    gcfg  = cfg["graph"]
    gauss = GaussianDistance(gcfg["gaussian_dmin"], gcfg["gaussian_dmax"], gcfg["gaussian_step"])
    tcfg  = cfg["text"]
    icfg  = cfg["image"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])

    ccfg = cfg["cgcnn"]
    fcfg = cfg["fusion"]
    model = DimAblationFusion(
        # Ablation flags
        ablate_text=args.ablate_text,
        ablate_image=args.ablate_image,
        text_embed_std=args.text_embed_std,
        image_embed_std=args.image_embed_std,
        # Architecture (identical to Exp 8)
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        atom_fea_len=ccfg["atom_fea_len"],
        nbr_fea_len=gauss.dim,
        n_conv=ccfg["n_conv"],
        cgcnn_h_fea_len=ccfg["h_fea_len"],
        cgcnn_n_h=ccfg["n_h"],
        bert_model_name=tcfg["model_name"],
        bert_unfreeze_last_n=cfg["train"]["bert_unfreeze_last_n"],
        fusion_hidden_dim=fcfg["hidden_dim"],
        fusion_dropout=fcfg["dropout"],
        metal_head_hidden=fcfg["metal_head_hidden"],
        image_dim=icfg["out_dim"],
        freeze_image_trunk=icfg["freeze_trunk"],
    ).to(device)

    # Load pre-trained CGCNN; skip BERT load when ablating text (not called in forward)
    load_pretrained(model, cfg, device, load_bert=not args.ablate_text)

    # Freeze encoders that are bypassed so their parameters don't enter the optimizer
    if args.ablate_text:
        for p in model.bert.parameters():
            p.requires_grad = False
        print("[DimAblation] BERT fully frozen (not called in forward)")
    if args.ablate_image:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        print("[DimAblation] ImageEncoder fully frozen (not called in forward)")

    image_preprocess = model.image_encoder.preprocess

    def make_set(d):
        return ThreeModalityFusionDataset(
            d, target_col=cfg["target"], radius=gcfg["radius"],
            max_neighbors=gcfg["max_neighbors"], gaussian=gauss,
            model_name=tcfg["model_name"], max_seq_len=tcfg["max_seq_len"],
            tokenizer=tokenizer,
            image_dir=icfg["dir"],
            image_preprocess=image_preprocess,
            project_root=PROJECT_ROOT,
        )

    train_set, val_set, test_set = make_set(train_df), make_set(val_df), make_set(test_df)
    bs = 4 if args.smoke else cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              collate_fn=collate_three_modality, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False,
                              collate_fn=collate_three_modality, num_workers=0,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False,
                              collate_fn=collate_three_modality, num_workers=0,
                              pin_memory=device.type == "cuda")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DimAblation] params total={n_total:,}, trainable={n_train:,}")

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

    best_val_mae    = float("inf")
    patience        = cfg["train"]["patience"]
    patience_counter = 0
    history = []
    t0 = time.time()

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
            "epoch": ep,
            "train_loss_total": round(train_total, 6),
            "train_loss_reg":   round(train_reg, 6),
            "train_loss_metal": round(train_metal, 6),
            "val_loss_total":   round(val_m["loss_total"], 6),
            "val_mae":          round(val_m["mae"], 4),
            "val_rmse":         round(val_m["rmse"], 4),
            "val_r2":           round(val_m["r2"], 4),
            "val_metal_acc":    round(val_m["metal_acc"], 4),
            "lr": optim.param_groups[0]["lr"],
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
            best_val_mae = val_m["mae"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop @ epoch {ep}")
                break

    elapsed = time.time() - t0
    print(f"[DimAblation] training done in {elapsed:.1f}s, best val MAE={best_val_mae:.4f}")

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
        f"\n[DimAblation] TEST seed={seed} mode={mode}: "
        f"MAE={test_m['mae']:.4f} RMSE={test_m['rmse']:.4f} "
        f"R2={test_m['r2']:.4f} metal_acc={test_m['metal_acc']:.3f}"
    )

    results = {
        "experiment": "exp_dim_ablation",
        "base_experiment": "exp8_three_modality",
        "ablation_mode": mode,
        "ablate_text": args.ablate_text,
        "ablate_image": args.ablate_image,
        "text_embed_std": args.text_embed_std,
        "image_embed_std": args.image_embed_std,
        "seed": seed,
        "smoke": args.smoke,
        "n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set),
        "n_params_total": n_total, "n_params_trainable": n_train,
        "training_time_s": round(elapsed, 1),
        "best_val_mae": round(best_val_mae, 4),
        "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_m.items()},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pred_df = test_set.df[["material_id", "formula", cfg["target"], "is_metal"]].copy()
    pred_df["bandgap_pred"] = test_bg_pred
    pred_df["bandgap_err"]  = test_bg_pred - test_bg_true
    pred_df["metal_prob"]   = test_metal_pred
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"[DimAblation] saved → {out_dir}/")

    if use_wandb:
        import wandb
        wandb.summary["test_mae"]       = test_m["mae"]
        wandb.summary["test_r2"]        = test_m["r2"]
        wandb.summary["test_metal_acc"] = test_m["metal_acc"]
        wandb.finish()


if __name__ == "__main__":
    main()
