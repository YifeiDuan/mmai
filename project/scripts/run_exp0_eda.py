#!/usr/bin/env python3
"""
Exp 0: Exploratory Data Analysis for ABO3 Perovskite Dataset.

Usage:
    python scripts/run_exp0_eda.py                          # all plots
    python scripts/run_exp0_eda.py --skip-tsne              # skip slow t-SNE
    python scripts/run_exp0_eda.py --config configs/exp0_eda.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.visualization.eda import (
    plot_band_gap_distribution,
    plot_formation_energy_distribution,
    plot_element_frequency,
    plot_spacegroup_pie,
    plot_wordcloud,
    plot_text_length_distribution,
    plot_text_embedding_tsne,
    save_summary_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 0: EDA")
    p.add_argument(
        "--config",
        type=str,
        default="configs/exp0_eda.yaml",
        help="Path to EDA config YAML.",
    )
    p.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE plot.")
    p.add_argument("--skip-wordcloud", action="store_true", help="Skip word cloud.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_path = PROJECT_ROOT / cfg["dataset_path"]
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[EDA] Loading dataset from {dataset_path}")
    df = load_dataset(str(dataset_path))
    print(f"[EDA] Loaded {len(df)} samples")

    # 0. Summary stats
    save_summary_stats(df, out_dir)

    # 1. Band gap
    print("[EDA] Plotting band gap distribution ...")
    plot_band_gap_distribution(df, out_dir)

    # 2. Formation energy
    print("[EDA] Plotting formation energy distribution ...")
    plot_formation_energy_distribution(df, out_dir)

    # 3. Element frequency
    print("[EDA] Plotting element frequency ...")
    plot_element_frequency(df, out_dir)

    # 4. Space group
    print("[EDA] Plotting spacegroup distribution ...")
    plot_spacegroup_pie(df, out_dir)

    # 5. Word cloud
    if not args.skip_wordcloud:
        print("[EDA] Generating word cloud ...")
        plot_wordcloud(df, out_dir)

    # 6. Text length
    print("[EDA] Plotting text length distribution ...")
    plot_text_length_distribution(df, out_dir)

    # 7. t-SNE
    if not args.skip_tsne:
        emb_cfg = cfg.get("embedding", {})
        print("[EDA] Computing SciBERT embeddings + t-SNE ...")
        plot_text_embedding_tsne(
            df,
            out_dir,
            model_name=emb_cfg.get("model_name", "allenai/scibert_scivocab_uncased"),
            max_seq_len=emb_cfg.get("max_seq_len", 256),
            method=emb_cfg.get("method", "tsne"),
            perplexity=emb_cfg.get("perplexity", 30),
            band_gap_threshold=emb_cfg.get("band_gap_threshold", 2.0),
        )

    print(f"\n[EDA] All figures saved to {out_dir}/")
    print("[EDA] Done.")


if __name__ == "__main__":
    main()
