"""Exp 0: Exploratory Data Analysis plotting functions.

Each function takes a DataFrame and an output directory, saves one figure,
and returns the matplotlib Figure for optional interactive use.
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (8, 5)


# ------------------------------------------------------------------
# 1. Band-gap distribution
# ------------------------------------------------------------------
def plot_band_gap_distribution(
    df: pd.DataFrame, out_dir: Path
) -> plt.Figure:
    """Histogram of band_gap, colored by is_metal."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    metals = df[df["is_metal"] == True]["band_gap"]
    nonmetals = df[df["is_metal"] == False]["band_gap"]

    ax.hist(nonmetals, bins=40, alpha=0.7, label=f"Non-metal ({len(nonmetals)})", color="#4C72B0")
    ax.hist(metals, bins=10, alpha=0.7, label=f"Metal ({len(metals)})", color="#DD8452")

    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title("Band Gap Distribution of ABO$_3$ Perovskites")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "band_gap_distribution.pdf", dpi=150)
    fig.savefig(out_dir / "band_gap_distribution.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 2. Formation energy distribution
# ------------------------------------------------------------------
def plot_formation_energy_distribution(
    df: pd.DataFrame, out_dir: Path
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["formation_energy_per_atom"], bins=40, color="#55A868", edgecolor="white")
    ax.set_xlabel("Formation Energy (eV/atom)")
    ax.set_ylabel("Count")
    ax.set_title("Formation Energy Distribution of ABO$_3$ Perovskites")
    fig.tight_layout()
    fig.savefig(out_dir / "formation_energy_distribution.pdf", dpi=150)
    fig.savefig(out_dir / "formation_energy_distribution.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 3. A-site / B-site element frequency
# ------------------------------------------------------------------
def plot_element_frequency(
    df: pd.DataFrame, out_dir: Path
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, site, color in zip(
        axes, ["a_site", "b_site"], ["#4C72B0", "#C44E52"]
    ):
        counts = df[site].value_counts().head(20)
        counts.plot.barh(ax=ax, color=color, edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title(f"Top-20 {site.replace('_', ' ').title()} Elements")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / "element_frequency.pdf", dpi=150)
    fig.savefig(out_dir / "element_frequency.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 4. Space-group distribution (pie)
# ------------------------------------------------------------------
def plot_spacegroup_pie(
    df: pd.DataFrame, out_dir: Path
) -> plt.Figure:
    counts = df["spacegroup_symbol"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 10},
    )
    ax.set_title("Space Group Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "spacegroup_pie.pdf", dpi=150)
    fig.savefig(out_dir / "spacegroup_pie.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 5. Word cloud from robocrys text
# ------------------------------------------------------------------
def plot_wordcloud(
    df: pd.DataFrame, out_dir: Path
) -> Optional[plt.Figure]:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[WARN] wordcloud not installed, skipping. pip install wordcloud")
        return None

    texts = df.loc[df["text_ok"] == True, "robocrys_text"]
    corpus = " ".join(texts.dropna().astype(str))

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        max_words=120,
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Robocrystallographer Descriptions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "wordcloud.pdf", dpi=150)
    fig.savefig(out_dir / "wordcloud.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 6. Text length (token count) distribution
# ------------------------------------------------------------------
def plot_text_length_distribution(
    df: pd.DataFrame, out_dir: Path
) -> plt.Figure:
    texts = df.loc[df["text_ok"] == True, "robocrys_text"].dropna()
    token_counts = texts.apply(lambda t: len(str(t).split()))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(token_counts, bins=40, color="#8172B2", edgecolor="white")
    ax.axvline(token_counts.median(), color="red", ls="--", label=f"Median={token_counts.median():.0f}")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Robocrys Text Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "text_length_distribution.pdf", dpi=150)
    fig.savefig(out_dir / "text_length_distribution.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 7. t-SNE / UMAP of SciBERT [CLS] embeddings
# ------------------------------------------------------------------
def plot_text_embedding_tsne(
    df: pd.DataFrame,
    out_dir: Path,
    model_name: str = "allenai/scibert_scivocab_uncased",
    max_seq_len: int = 256,
    method: str = "tsne",
    perplexity: int = 30,
    band_gap_threshold: float = 2.0,
) -> Optional[plt.Figure]:
    """Compute SciBERT [CLS] embeddings and plot 2-D projection.

    This function requires torch and transformers.  It runs on CPU by
    default (fast enough for ~500 samples).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("[WARN] torch/transformers not installed, skipping t-SNE plot.")
        return None

    sub = df[df["text_ok"] == True].copy()
    texts = sub["robocrys_text"].fillna("").astype(str).tolist()
    gaps = sub["band_gap"].values

    print(f"[t-SNE] Encoding {len(texts)} texts with {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls)

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"[t-SNE] Embedding shape: {embeddings.shape}")

    # Dimensionality reduction
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            print("[WARN] umap not installed, falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    coords = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=gaps,
        cmap="coolwarm",
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Band Gap (eV)")
    method_label = method.upper()
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    ax.set_title(f"SciBERT [CLS] Embeddings ({method_label}) — Colored by Band Gap")
    fig.tight_layout()
    fig.savefig(out_dir / f"text_embedding_{method}.pdf", dpi=150)
    fig.savefig(out_dir / f"text_embedding_{method}.png", dpi=150)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# 8. Summary statistics table (saved as CSV)
# ------------------------------------------------------------------
def save_summary_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Save a CSV of key summary statistics."""
    stats = {
        "total_samples": len(df),
        "text_ok_samples": int(df["text_ok"].sum()) if "text_ok" in df.columns else "N/A",
        "text_failed": int((~df["text_ok"]).sum()) if "text_ok" in df.columns else "N/A",
        "n_metals": int(df["is_metal"].sum()),
        "n_nonmetals": int((~df["is_metal"]).sum()),
        "unique_a_site": df["a_site"].nunique(),
        "unique_b_site": df["b_site"].nunique(),
        "unique_spacegroups": df["spacegroup_symbol"].nunique(),
        "band_gap_mean": round(df["band_gap"].mean(), 4),
        "band_gap_std": round(df["band_gap"].std(), 4),
        "band_gap_min": round(df["band_gap"].min(), 4),
        "band_gap_max": round(df["band_gap"].max(), 4),
        "form_energy_mean": round(df["formation_energy_per_atom"].mean(), 4),
        "form_energy_std": round(df["formation_energy_per_atom"].std(), 4),
        "train_samples": int((df["split"] == "train").sum()),
        "val_samples": int((df["split"] == "val").sum()),
        "test_samples": int((df["split"] == "test").sum()),
    }
    stats_df = pd.DataFrame(list(stats.items()), columns=["metric", "value"])
    stats_df.to_csv(out_dir / "summary_stats.csv", index=False)
    print(f"[INFO] Summary stats saved to {out_dir / 'summary_stats.csv'}")
    return stats_df
