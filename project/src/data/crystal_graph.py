"""Crystal graph construction from CIF files using pymatgen."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pymatgen.core import Structure


# Atom feature vector: one-hot encoding over common elements in perovskites
# Covers all likely A-site, B-site elements + O
ELEM_LIST = [
    "H", "Li", "Be", "B", "C", "N", "O", "F",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At",
    "Th", "U",
]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEM_LIST)}
NUM_ATOM_FEATURES = len(ELEM_LIST)


def atom_feature(element_symbol: str) -> np.ndarray:
    """One-hot encoding for an element."""
    vec = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    idx = ELEM_TO_IDX.get(element_symbol, None)
    if idx is not None:
        vec[idx] = 1.0
    return vec


class GaussianDistance:
    """Expand distances with Gaussian basis functions."""

    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, step: float = 0.2):
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.centers = np.arange(dmin, dmax + 1e-8, step)
        self.width = step  # sigma = step

    @property
    def dim(self) -> int:
        return len(self.centers)

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """distances: (N,) -> (N, dim)"""
        return np.exp(
            -((distances[:, None] - self.centers[None, :]) ** 2)
            / (2 * self.width ** 2)
        ).astype(np.float32)


def build_crystal_graph(
    structure: Structure,
    radius: float = 8.0,
    max_neighbors: int = 12,
    gaussian: Optional[GaussianDistance] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a crystal graph from a pymatgen Structure.

    Returns:
        atom_fea: (n_atoms, n_atom_features)
        nbr_idx:  (n_atoms, max_neighbors) - neighbor indices
        nbr_fea:  (n_atoms, max_neighbors, n_bond_features) - Gaussian-expanded distances
        nbr_dist: (n_atoms, max_neighbors) - raw distances
    """
    if gaussian is None:
        gaussian = GaussianDistance()

    n_atoms = len(structure)

    # Atom features
    atom_fea = np.array(
        [atom_feature(str(site.specie)) for site in structure],
        dtype=np.float32,
    )

    # Neighbor list
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    # Each entry: list of (site, distance, index, image)

    nbr_idx = np.zeros((n_atoms, max_neighbors), dtype=np.int64)
    nbr_dist = np.zeros((n_atoms, max_neighbors), dtype=np.float32)

    for i, site_nbrs in enumerate(all_nbrs):
        # Sort by distance
        sorted_nbrs = sorted(site_nbrs, key=lambda x: x[1])
        if len(sorted_nbrs) < max_neighbors:
            # Pad by repeating nearest neighbor
            while len(sorted_nbrs) < max_neighbors:
                sorted_nbrs.append(sorted_nbrs[-1] if sorted_nbrs else (None, 0.0, i, None))

        for j in range(max_neighbors):
            nbr_idx[i, j] = sorted_nbrs[j][2]
            nbr_dist[i, j] = sorted_nbrs[j][1]

    nbr_fea = gaussian.expand(nbr_dist.reshape(-1)).reshape(
        n_atoms, max_neighbors, -1
    )

    return atom_fea, nbr_idx, nbr_fea, nbr_dist


class CrystalGraphDataset(Dataset):
    """PyTorch dataset for crystal graph data.

    Loads CIF files on-the-fly and builds crystal graphs.
    """

    def __init__(
        self,
        df,
        target_col: str = "band_gap",
        radius: float = 8.0,
        max_neighbors: int = 12,
        gaussian: Optional[GaussianDistance] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.gaussian = gaussian or GaussianDistance()
        self._cache: Dict[int, Tuple] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        cif_path = row["cif_path"]
        target = float(row[self.target_col])

        structure = Structure.from_file(cif_path)
        atom_fea, nbr_idx, nbr_fea, _ = build_crystal_graph(
            structure,
            radius=self.radius,
            max_neighbors=self.max_neighbors,
            gaussian=self.gaussian,
        )

        atom_fea = torch.tensor(atom_fea, dtype=torch.float32)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx, dtype=torch.long)
        target_t = torch.tensor([target], dtype=torch.float32)

        result = (atom_fea, nbr_fea, nbr_idx, target_t)
        self._cache[idx] = result
        return result


def collate_crystal_graphs(batch):
    """Custom collate: concatenate variable-size graphs with batch index."""
    atom_feas, nbr_feas, nbr_idxs, targets = zip(*batch)

    # Offset neighbor indices per graph
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    crystal_atom_idx = []  # maps each atom to its crystal index in batch

    offset = 0
    for i, (af, nf, ni) in enumerate(zip(atom_feas, nbr_feas, nbr_idxs)):
        n = af.shape[0]
        batch_atom_fea.append(af)
        batch_nbr_fea.append(nf)
        batch_nbr_idx.append(ni + offset)
        crystal_atom_idx.extend([i] * n)
        offset += n

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_idx, dim=0),
        torch.tensor(crystal_atom_idx, dtype=torch.long),
        torch.cat(targets, dim=0),
    )
