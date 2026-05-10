#!/usr/bin/env python3
"""Render every CIF in the ABO3 dataset to a 224×224 ball-stick PNG.

Why two views? A single 2D projection of an octahedrally-coordinated
perovskite often hides B-O coordination geometry that distinguishes
metals from insulators. Rendering xy + xz views and tiling into one
PNG gives the vision encoder a near-3D signal at the cost of one
extra trivial concat.

Output: data/processed/multimodal_v1/images/<material_id>.png

Runs on CPU. Skips already-rendered files. ~10 ms per crystal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from ase.visualize.plot import plot_atoms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset


def render_one(cif_path: Path, out_path: Path, size: int = 224) -> bool:
    """Render a single CIF as a 2-view ball-stick PNG."""
    try:
        atoms = ase_read(str(cif_path))
    except Exception as e:
        print(f"[skip] {cif_path.name}: ase_read failed: {e}")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(size / 50, size / 100), dpi=50)
    plot_atoms(atoms, axes[0], rotation="0x,0y,0z",
               radii=0.45, show_unit_cell=2)
    plot_atoms(atoms, axes[1], rotation="-90x,0y,0z",
               radii=0.45, show_unit_cell=2)
    for a in axes:
        a.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(out_path, dpi=50, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/multimodal_v1/dataset.parquet")
    p.add_argument("--out-dir", default="data/processed/multimodal_v1/images")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--force", action="store_true",
                   help="Re-render even if PNG exists.")
    args = p.parse_args()

    df = load_dataset(str(PROJECT_ROOT / args.data))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(df)} crystals to {out_dir}/")
    n_done, n_skip, n_fail = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mid = row["material_id"]
        cif = PROJECT_ROOT / row["cif_path"]
        out = out_dir / f"{mid}.png"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        if not cif.exists():
            print(f"[skip] {mid}: cif missing at {cif}")
            n_fail += 1
            continue
        if render_one(cif, out, size=args.size):
            n_done += 1
        else:
            n_fail += 1
    print(f"Done: rendered={n_done} skipped(existing)={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
