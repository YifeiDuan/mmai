"""Generate poster figures from an inverse design run.

Reads ``results/inverse_design/target_*/all_candidates.csv`` and writes:
- convergence.png   — per-target subplots: candidates as scatter, best-so-far as line
- top_candidates.png — top-3 per target, annotated with predictions + flags
- vocab_usage.png   — element frequency over iterations (LLM exploration heatmap)

Usage:
    python scripts/visualize_inverse_design.py [--results-dir results/inverse_design]
                                                [--out-dir results/figures/inverse_design]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("viz")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/inverse_design")
    p.add_argument("--out-dir", default="results/figures/inverse_design")
    return p.parse_args()


def load_runs(results_dir: Path):
    """Return dict {target_bg: candidates_df}."""
    runs = {}
    for sub in sorted(results_dir.glob("target_*")):
        if not sub.is_dir():
            continue
        try:
            tbg = float(sub.name.replace("target_", ""))
        except ValueError:
            continue
        csv = sub / "all_candidates.csv"
        if not csv.exists():
            log.warning("Missing %s, skipping", csv)
            continue
        df = pd.read_csv(csv)
        runs[tbg] = df
    return runs


def plot_convergence(runs, out_path: Path):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (tbg, df) in zip(axes, sorted(runs.items())):
        df_ok = df[~df["failed"].astype(bool)].copy() if "failed" in df.columns else df
        # Scatter: every candidate
        ax.scatter(df_ok["iter"], df_ok["error_to_target"],
                   c="lightsteelblue", alpha=0.7, s=40, label="candidates")
        # Highlight high-confidence
        if "low_confidence" in df_ok.columns:
            hi = df_ok[~df_ok["low_confidence"].astype(bool)]
            ax.scatter(hi["iter"], hi["error_to_target"],
                       c="steelblue", alpha=0.95, s=55, label="high confidence",
                       edgecolor="white", linewidth=0.5)
        # Best-so-far line
        best_per_iter = df_ok.groupby("iter")["error_to_target"].min()
        running_best = best_per_iter.cummin()
        ax.plot(running_best.index, running_best.values, color="crimson",
                marker="o", lw=2, label="best so far")
        ax.axhline(0.0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"target = {tbg:.1f} eV", fontsize=12)
        ax.set_xlabel("iteration")
        ax.set_ylabel("|predicted − target| (eV)")
        ax.set_xticks(sorted(df_ok["iter"].unique()))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Inverse design convergence (Exp 5b ensemble oracle)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("wrote %s", out_path)


def plot_top_candidates(runs, out_path: Path, k: int = 3):
    sorted_runs = sorted(runs.items())
    n = len(sorted_runs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (tbg, df) in zip(axes, sorted_runs):
        df_ok = df[~df["failed"].astype(bool)].copy() if "failed" in df.columns else df
        # Sort by |error|, prefer high-confidence and charge-balanced
        sort_cols = ["error_to_target"]
        if "low_confidence" in df_ok.columns:
            sort_cols.append("low_confidence")
        if "charge_warning" in df_ok.columns:
            sort_cols.append("charge_warning")
        top = df_ok.sort_values(sort_cols).head(k).reset_index(drop=True)

        labels = [f"{r['A']}{r['B']}O3\n(base {r['base_mid']})" for _, r in top.iterrows()]
        means = top["pred_bg_mean"].values
        stds = top["pred_bg_std"].values
        ax.errorbar(range(len(labels)), means, yerr=stds,
                    fmt="o", color="steelblue", capsize=5, lw=2, ms=10,
                    label="prediction ± std")
        ax.axhline(tbg, color="crimson", ls="--", lw=1.5, label=f"target {tbg:.1f} eV")
        # Annotate flags
        for i, (_, r) in enumerate(top.iterrows()):
            flags = []
            if r.get("charge_warning", False):
                flags.append("charge?")
            if r.get("low_confidence", False):
                flags.append("low conf")
            if flags:
                ax.text(i, means[i] + stds[i] + 0.1, "·".join(flags),
                        ha="center", fontsize=8, color="darkorange")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("predicted band gap (eV)")
        ax.set_title(f"target = {tbg:.1f} eV — top {k}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Top-k inverse design candidates per target", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("wrote %s", out_path)


def plot_vocab_usage(runs, out_path: Path):
    """Frequency heatmap of which A/B elements were chosen at each iteration."""
    sorted_runs = sorted(runs.items())
    n = len(sorted_runs)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 8.0), sharex=False)
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, (tbg, df) in enumerate(sorted_runs):
        for row, axis in enumerate(["A", "B"]):
            ax = axes[row, col]
            counts = df.groupby(["iter", axis]).size().unstack(axis, fill_value=0)
            # Show only elements actually used
            counts = counts.loc[:, (counts.sum(axis=0) > 0)]
            counts = counts.sort_index(axis=1)
            im = ax.imshow(counts.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(counts.shape[1]))
            ax.set_xticklabels(counts.columns, rotation=60, fontsize=8)
            ax.set_yticks(range(counts.shape[0]))
            ax.set_yticklabels(counts.index, fontsize=9)
            ax.set_xlabel(f"{axis}-site element")
            ax.set_ylabel("iter")
            ax.set_title(f"target {tbg:.1f} eV — {axis}-site usage", fontsize=11)
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("count", fontsize=8)
    fig.suptitle("LLM proposer vocab exploration (per iteration)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("wrote %s", out_path)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    if not runs:
        log.error("No runs found in %s", results_dir)
        return
    log.info("Loaded %d runs from %s", len(runs), results_dir)

    plot_convergence(runs, out_dir / "convergence.png")
    plot_top_candidates(runs, out_dir / "top_candidates.png")
    plot_vocab_usage(runs, out_dir / "vocab_usage.png")
    log.info("done")


if __name__ == "__main__":
    main()
