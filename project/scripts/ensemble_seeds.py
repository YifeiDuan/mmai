#!/usr/bin/env python3
"""Ensemble multi-seed predictions for any experiment that writes
results/<expname>/seed_*/test_predictions.csv with columns
[material_id, formula, band_gap, bandgap_pred, ...].

Computes:
  - Per-seed test MAE / RMSE / R^2
  - Cross-seed mean ± std of single-seed MAEs
  - Ensemble MAE / RMSE / R^2 (mean of bandgap_pred across seeds, then metrics)

Usage:
    python scripts/ensemble_seeds.py results/exp5_multitask_film
    python scripts/ensemble_seeds.py results/exp5_multitask_film results/exp6_crossattn ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ensemble import average_predictions
from src.evaluation.metrics import regression_metrics


def ensemble_one_experiment(exp_dir: Path) -> dict:
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if not seed_dirs:
        return {"error": f"no seed_* under {exp_dir}"}

    preds, true = [], None
    seed_maes = {}
    metal_accs = {}
    for sd in seed_dirs:
        pred_csv = sd / "test_predictions.csv"
        if not pred_csv.exists():
            continue
        df = pd.read_csv(pred_csv)
        preds.append(df["bandgap_pred"].values)
        if true is None:
            true = df["band_gap"].values
        seed_label = sd.name.replace("seed_", "")
        seed_maes[seed_label] = float(np.mean(np.abs(df["bandgap_pred"].values - true)))
        if "metal_prob" in df.columns and "is_metal" in df.columns:
            preds_metal = (df["metal_prob"].values > 0.5).astype(int)
            metal_accs[seed_label] = float((preds_metal == df["is_metal"].astype(int).values).mean())

    if not preds:
        return {"error": "no test_predictions.csv files found"}

    ens = average_predictions(preds)
    ens_metrics = regression_metrics(true, ens)
    single_maes = list(seed_maes.values())

    return {
        "experiment": exp_dir.name,
        "n_seeds": len(preds),
        "n_test": len(true),
        "single_seed_test_mae": seed_maes,
        "single_seed_metal_acc": metal_accs,
        "single_seed_mae_mean": round(float(np.mean(single_maes)), 4),
        "single_seed_mae_std": round(float(np.std(single_maes)), 4),
        "ensemble_mae": round(ens_metrics["mae"], 4),
        "ensemble_rmse": round(ens_metrics["rmse"], 4),
        "ensemble_r2": round(ens_metrics["r2"], 4),
        "improvement_vs_seed_mean_pct": round(
            100.0 * (np.mean(single_maes) - ens_metrics["mae"]) / np.mean(single_maes), 2,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="One or more results/exp_*/ directories")
    parser.add_argument("--output", default=None, help="If set, dump JSON to this path")
    parser.add_argument("--baseline-mae", type=float, default=0.5094,
                        help="SoTA baseline MAE for comparison (default: 0.5094 from project README)")
    args = parser.parse_args()

    all_results = []
    for d in args.exp_dirs:
        path = Path(d)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        result = ensemble_one_experiment(path)
        all_results.append(result)
        print()
        print(f"=== {result.get('experiment', d)} ===")
        if "error" in result:
            print(f"  {result['error']}")
            continue
        print(f"  n_seeds: {result['n_seeds']}, n_test: {result['n_test']}")
        print(f"  Single-seed MAE per seed: " +
              ", ".join(f"{s}={v:.4f}" for s, v in result['single_seed_test_mae'].items()))
        if result['single_seed_metal_acc']:
            print(f"  Single-seed metal_acc:    " +
                  ", ".join(f"{s}={v:.3f}" for s, v in result['single_seed_metal_acc'].items()))
        print(f"  Single-seed mean ± std:  {result['single_seed_mae_mean']:.4f} ± {result['single_seed_mae_std']:.4f}")
        print(f"  Ensemble MAE:            {result['ensemble_mae']:.4f}  (R²={result['ensemble_r2']:.4f}, RMSE={result['ensemble_rmse']:.4f})")
        print(f"  Variance reduction (ens vs seed-mean): {result['improvement_vs_seed_mean_pct']:.1f}% lower")
        delta = args.baseline_mae - result["ensemble_mae"]
        rel = 100.0 * delta / args.baseline_mae
        sign = "+" if delta > 0 else ""
        print(f"  vs SoTA baseline {args.baseline_mae:.4f}: ensemble Δ = {sign}{delta:.4f} ({sign}{rel:.1f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON dumped → {args.output}")


if __name__ == "__main__":
    main()
