#!/usr/bin/env python3
"""Build the modality-ablation table for the poster.

Computes 5-seed ensemble test MAE / R^2 for:
  1. exp1_cgcnn_multiseed         — structure-only
  2. exp2_scibert_multiseed/finetune — text-only
  3. exp3_fusion_multiseed/film   — structure + text (fusion, no multitask)
  4. exp5b_multitask_film_nowarm  — struct + text + multitask aux head (no warm)
  5. exp5_multitask_film          — struct + text + multitask + FiLM warm-start
  6. exp6_crossattn               — struct + text via cross-attention
  7. exp7_albef                   — struct + text + multitask + ALBEF init (Phase B)
  8. exp8_three_modality          — struct + text + image + multitask (Phase C)

Outputs:
  - results/ablation_table.json
  - results/ablation_table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ensemble import average_predictions
from src.evaluation.metrics import regression_metrics


# Each entry: name, glob pattern relative to project root, modalities label
ENTRIES = [
    ("Exp 1 CGCNN (struct only)",
     "results/exp1_cgcnn_multiseed/seed_*/test_predictions.csv",
     "struct", "exp1_struct"),
    ("Exp 2 SciBERT FT (text only)",
     "results/exp2_scibert_multiseed/seed_*/finetune/test_predictions.csv",
     "text", "exp2_text"),
    ("Exp 3 FiLM (struct + text)",
     "results/exp3_fusion_multiseed/seed_*/film/test_predictions.csv",
     "struct+text", "exp3_film"),
    ("Exp 5 MT-FiLM (struct + text + MT, FiLM warm)",
     "results/exp5_multitask_film/seed_*/test_predictions.csv",
     "struct+text+MT (warm)", "exp5_mt"),
    ("Exp 5b MT-FiLM (struct + text + MT, no warm)",
     "results/exp5b_multitask_film_nowarm/seed_*/test_predictions.csv",
     "struct+text+MT", "exp5b_mt_nowarm"),
    ("Exp 6 CrossAttn (struct + text via cross-attn)",
     "results/exp6_crossattn/seed_*/test_predictions.csv",
     "struct+text+MT (cross-attn)", "exp6_cross"),
    ("Exp 7 ALBEF (align then fuse)",
     "results/exp7_albef/seed_*/test_predictions.csv",
     "struct+text+MT (align init)", "exp7_albef"),
    ("Exp 8 3-modality (struct + text + image)",
     "results/exp8_three_modality/seed_*/test_predictions.csv",
     "struct+text+image+MT", "exp8_3mod"),
]

# These two columns vary across files: exp1 uses "band_gap" + "predicted",
# all multitask scripts use "band_gap" + "bandgap_pred", original exp2/3 also use
# "predicted". Detect which column is present.
PRED_CANDIDATES = ["bandgap_pred", "predicted"]


def get_pred_col(df: pd.DataFrame) -> str:
    for c in PRED_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"None of {PRED_CANDIDATES} in df.columns: {list(df.columns)}")


def ensemble_for_glob(pattern: str):
    paths = sorted((PROJECT_ROOT).glob(pattern))
    if not paths:
        return None
    preds, true = [], None
    seed_maes = []
    for p in paths:
        df = pd.read_csv(p)
        col = get_pred_col(df)
        preds.append(df[col].values)
        if true is None:
            true = df["band_gap"].values
        seed_maes.append(float(np.mean(np.abs(df[col].values - true))))
    if true is None:
        return None
    ens = average_predictions(preds)
    m = regression_metrics(true, ens)
    return {
        "n_seeds": len(preds),
        "n_test": len(true),
        "single_seed_mae_mean": round(float(np.mean(seed_maes)), 4),
        "single_seed_mae_std": round(float(np.std(seed_maes)), 4),
        "single_seed_mae_min": round(float(np.min(seed_maes)), 4),
        "ensemble_mae": round(m["mae"], 4),
        "ensemble_rmse": round(m["rmse"], 4),
        "ensemble_r2": round(m["r2"], 4),
    }


def main():
    rows = []
    for name, pat, mods, key in ENTRIES:
        result = ensemble_for_glob(pat)
        if result is None:
            print(f"  [skip] {name}: no files matching {pat}")
            continue
        result["name"] = name
        result["modalities"] = mods
        result["key"] = key
        rows.append(result)
        print(f"  {name}: ens MAE={result['ensemble_mae']:.4f}, R²={result['ensemble_r2']:.4f}, n_seeds={result['n_seeds']}")

    if not rows:
        print("Nothing to report yet.")
        return

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "ablation_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Markdown table
    lines = [
        "| # | Method | Modalities | n seeds | Single-seed MAE (mean ± std) | Best seed MAE | **Ensemble MAE** | Ensemble R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['modalities']} | {r['n_seeds']} | "
            f"{r['single_seed_mae_mean']:.4f} ± {r['single_seed_mae_std']:.4f} | "
            f"{r['single_seed_mae_min']:.4f} | "
            f"**{r['ensemble_mae']:.4f}** | {r['ensemble_r2']:.4f} |"
        )
    md = "\n".join(lines) + "\n"
    with open(out_dir / "ablation_table.md", "w") as f:
        f.write("# Modality Ablation (5-seed ensemble) on band_gap test set\n\n")
        f.write(md)
    print(f"\nSaved → {out_dir/'ablation_table.json'}")
    print(f"Saved → {out_dir/'ablation_table.md'}")
    print()
    print(md)


if __name__ == "__main__":
    main()
