"""Inverse design main loop: random init iter 0, then LLM-proposed iters 1..n.

The agent maintains a history of (iter, candidates, predictions) records, asks
the proposer for new candidates conditioned on that history, and evaluates
each candidate by mutating a base CIF + text and running the forward oracle.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from src.inverse_design.proposer import Candidate
from src.inverse_design.substitute import substitute_cif, substitute_text

logger = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    iter: int
    A: str
    B: str
    base_mid: str
    pred_bg_mean: float
    pred_bg_std: float
    is_metal_prob: float
    error_to_target: float
    cif_path: str
    text: str
    charge_warning: bool = False
    low_confidence: bool = False
    failed: bool = False
    fail_reason: str = ""


def _charge_balance_warning(a: str, b: str) -> bool:
    """Heuristic: ABO3 needs A^n + B^m + 3*(-2) = 0, i.e. n + m = 6.

    Uses a coarse oxidation-state lookup table that covers the elements seen
    in the dataset. Returns True if no (A, B) charge pair sums to 6 — i.e.
    the substitution is unlikely to be charge-balanced and the prediction
    should be flagged in downstream CSVs. False positives are tolerable
    because we're labelling, not gating.
    """
    common_states = {
        # Group I, II
        "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
        "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
        # Transition / post-transition (most common Wyckoff oxidation states)
        "Sc": [3], "Y": [3], "La": [3], "Ti": [3, 4], "Zr": [4], "Hf": [4],
        "V": [3, 4, 5], "Nb": [3, 4, 5], "Ta": [4, 5],
        "Cr": [3, 4, 6], "Mo": [3, 4, 5, 6], "W": [4, 5, 6],
        "Mn": [2, 3, 4], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3],
        "Cu": [1, 2], "Zn": [2], "Cd": [2],
        "Al": [3], "Ga": [3], "In": [3],
        "Si": [4], "Ge": [4], "Sn": [2, 4], "Pb": [2, 4],
        "P": [3, 5], "As": [3, 5], "Sb": [3, 5], "Bi": [3, 5],
        "Ru": [3, 4], "Rh": [3], "Ir": [3, 4], "Pt": [2, 4], "Pd": [2, 4],
        "Ce": [3, 4], "Pr": [3, 4], "Nd": [3], "Sm": [2, 3], "Eu": [2, 3],
        "Gd": [3], "Tb": [3, 4], "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3],
        "Yb": [2, 3], "Lu": [3], "Th": [4], "U": [4, 5, 6],
        "Tc": [4, 7], "Re": [4, 6, 7],
    }
    sa = common_states.get(a)
    sb = common_states.get(b)
    if not sa or not sb:
        return True  # unknown pair → flag conservatively
    for na in sa:
        for nb in sb:
            if na + nb == 6:
                return False
    return True


class InverseDesignAgent:
    """LLM-in-the-loop iterative inverse design."""

    def __init__(
        self,
        oracle,
        proposer,
        base_df: pd.DataFrame,
        target_bg: float,
        a_vocab: Sequence[str],
        b_vocab: Sequence[str],
        n_iter: int = 5,
        n_per_iter: int = 5,
        workdir: Optional[Path] = None,
        seed: int = 0,
        std_threshold: float = 0.5,
    ):
        self.oracle = oracle
        self.proposer = proposer
        # Keep only rows with a valid CIF + text — those are what we can mutate.
        df = base_df.copy()
        for col in ("material_id", "a_site", "b_site", "cif_path", "robocrys_text"):
            if col not in df.columns:
                raise ValueError(f"base_df missing required column: {col}")
        df = df[df["robocrys_text"].notna()].reset_index(drop=True)
        self.base_df = df
        self.base_index = df.set_index("material_id")
        self.target_bg = float(target_bg)
        self.a_vocab = list(a_vocab)
        self.b_vocab = list(b_vocab)
        self.n_iter = int(n_iter)
        self.n_per_iter = int(n_per_iter)
        self.workdir = Path(workdir) if workdir is not None else Path(".")
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "cifs").mkdir(exist_ok=True)
        self.seed = int(seed)
        self.std_threshold = float(std_threshold)
        self._rng = random.Random(seed)

    def _random_init(self, n: int) -> List[Candidate]:
        bases = self.base_df["material_id"].tolist()
        return [
            (
                self._rng.choice(self.a_vocab),
                self._rng.choice(self.b_vocab),
                self._rng.choice(bases),
            )
            for _ in range(n)
        ]

    def _evaluate(self, it: int, cand_idx: int, cand: Candidate) -> CandidateRecord:
        a, b, base_mid = cand
        if base_mid not in self.base_index.index:
            logger.warning("Unknown base mid %s; sampling random base", base_mid)
            base_mid = self._rng.choice(self.base_df["material_id"].tolist())
        base = self.base_index.loc[base_mid]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        base_a = str(base["a_site"])
        base_b = str(base["b_site"])
        base_cif = str(base["cif_path"])
        base_text = str(base["robocrys_text"])

        out_cif = self.workdir / "cifs" / f"iter{it:02d}_cand{cand_idx:02d}.cif"
        try:
            substitute_cif(base_cif, base_a, base_b, a, b, str(out_cif))
            new_text = substitute_text(base_text, base_a, base_b, a, b)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Substitution failed for %s/%s -> %s/%s", base_a, base_b, a, b)
            return CandidateRecord(
                iter=it, A=a, B=b, base_mid=base_mid,
                pred_bg_mean=float("nan"), pred_bg_std=float("nan"),
                is_metal_prob=float("nan"), error_to_target=float("inf"),
                cif_path=str(out_cif), text="",
                failed=True, fail_reason=f"substitute: {type(exc).__name__}: {exc}",
            )

        try:
            pred = self.oracle.predict(str(out_cif), new_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Oracle prediction failed")
            return CandidateRecord(
                iter=it, A=a, B=b, base_mid=base_mid,
                pred_bg_mean=float("nan"), pred_bg_std=float("nan"),
                is_metal_prob=float("nan"), error_to_target=float("inf"),
                cif_path=str(out_cif), text=new_text,
                failed=True, fail_reason=f"oracle: {type(exc).__name__}: {exc}",
            )

        mean = float(pred["band_gap_mean"])
        std = float(pred["band_gap_std"])
        metal = float(pred["is_metal_prob"])
        return CandidateRecord(
            iter=it, A=a, B=b, base_mid=base_mid,
            pred_bg_mean=mean, pred_bg_std=std, is_metal_prob=metal,
            error_to_target=abs(mean - self.target_bg),
            cif_path=str(out_cif), text=new_text,
            charge_warning=_charge_balance_warning(a, b),
            low_confidence=std > self.std_threshold,
        )

    def run(self) -> List[CandidateRecord]:
        all_records: List[CandidateRecord] = []
        history: List[dict] = []

        for it in range(self.n_iter):
            if it == 0:
                cands = self._random_init(self.n_per_iter)
                source = "random"
            else:
                cands = self.proposer.propose(
                    target_bg=self.target_bg, history=history,
                    a_vocab=self.a_vocab, b_vocab=self.b_vocab,
                    base_set=self.base_df["material_id"].tolist(),
                    n_proposals=self.n_per_iter,
                )
                # Pad with random if proposer returned too few
                if len(cands) < self.n_per_iter:
                    cands = list(cands) + self._random_init(self.n_per_iter - len(cands))
                cands = cands[: self.n_per_iter]
                source = "llm"

            iter_records: List[CandidateRecord] = []
            for j, c in enumerate(cands):
                rec = self._evaluate(it, j, c)
                iter_records.append(rec)

            iter_history = {
                "iter": it,
                "source": source,
                "candidates": [
                    {
                        "A": r.A, "B": r.B, "base_mid": r.base_mid,
                        "pred_bg_mean": r.pred_bg_mean,
                        "pred_bg_std": r.pred_bg_std,
                        "is_metal_prob": r.is_metal_prob,
                        "error_to_target": r.error_to_target,
                    }
                    for r in iter_records
                ],
            }
            history.append(iter_history)
            all_records.extend(iter_records)

            best = min(iter_records, key=lambda r: r.error_to_target)
            logger.info(
                "iter %d (%s): best |err|=%.3f (A=%s B=%s base=%s)",
                it, source, best.error_to_target, best.A, best.B, best.base_mid,
            )

        return all_records


def records_to_dataframe(records: Iterable[CandidateRecord]) -> pd.DataFrame:
    """Tabular view of agent records, ready for CSV/JSON dump."""
    return pd.DataFrame([asdict(r) for r in records])
