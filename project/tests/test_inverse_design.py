"""Tests for the inverse design agent (substitute, oracle, proposer, agent)."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


# -------------------- substitute.py --------------------

class TestSubstituteCif:
    def _sample_cif(self) -> str:
        path = PROJECT_ROOT / "data/processed/multimodal_v1/cifs/mp-3731.cif"  # LiNbO3
        if not path.exists():
            pytest.skip(f"Sample CIF missing: {path}")
        return str(path)

    def test_idempotent_when_base_equals_new(self, tmp_path):
        from src.inverse_design.substitute import substitute_cif
        from pymatgen.core import Structure

        out = tmp_path / "out.cif"
        substitute_cif(self._sample_cif(), "Li", "Nb", "Li", "Nb", str(out))
        s_orig = Structure.from_file(self._sample_cif())
        s_new = Structure.from_file(out)
        # Same composition and same site count
        assert s_orig.composition.reduced_formula == s_new.composition.reduced_formula
        assert len(s_orig) == len(s_new)

    def test_swap_li_to_sr_changes_composition(self, tmp_path):
        from src.inverse_design.substitute import substitute_cif
        from pymatgen.core import Structure

        out = tmp_path / "swap.cif"
        substitute_cif(self._sample_cif(), "Li", "Nb", "Sr", "Ti", str(out))
        s_new = Structure.from_file(out)
        species = set(str(sp) for sp in s_new.composition.elements)
        assert "Li" not in species and "Nb" not in species
        assert "Sr" in species and "Ti" in species
        assert "O" in species  # O preserved

    def test_invalid_element_raises(self, tmp_path):
        from src.inverse_design.substitute import substitute_cif

        out = tmp_path / "bad.cif"
        with pytest.raises(Exception):  # pymatgen Element raises ValueError
            substitute_cif(self._sample_cif(), "Li", "Nb", "Xx", "Ti", str(out))

    def test_output_is_cif_readable(self, tmp_path):
        from src.inverse_design.substitute import substitute_cif
        from pymatgen.core import Structure

        out = tmp_path / "ba_zr.cif"
        substitute_cif(self._sample_cif(), "Li", "Nb", "Ba", "Zr", str(out))
        # pymatgen must be able to read it back
        s = Structure.from_file(out)
        assert len(s) > 0
        assert "Ba" in {str(sp) for sp in s.composition.elements}


class TestSubstituteText:
    def test_idempotent_when_base_equals_new(self):
        from src.inverse_design.substitute import substitute_text
        text = "Li(1) is bonded in a 6-coordinate geometry to one O(1)"
        assert substitute_text(text, "Li", "Nb", "Li", "Nb") == text

    def test_simple_element_replaced_with_word_boundary(self):
        from src.inverse_design.substitute import substitute_text
        text = "Li is the A-site cation."
        out = substitute_text(text, "Li", "Nb", "Sr", "Ti")
        assert "Sr is the A-site cation." == out
        assert "Li" not in out

    def test_element_with_paren_index_replaced(self):
        from src.inverse_design.substitute import substitute_text
        text = "There are four inequivalent Li sites. Li(1) is bonded to O(1)."
        out = substitute_text(text, "Li", "Nb", "Sr", "Ti")
        assert "Sr(1)" in out
        assert "Sr sites" in out
        assert "Li" not in out
        assert "O(1)" in out  # O preserved

    def test_does_not_corrupt_o_words(self):
        from src.inverse_design.substitute import substitute_text
        text = "Octahedra and tetrahedral coordination around O(1)."
        out = substitute_text(text, "Li", "Nb", "Sr", "Ti")
        assert "Octahedra" in out  # NOT corrupted
        assert "tetrahedral" in out
        assert "O(1)" in out

    def test_does_not_replace_substring_of_longer_symbol(self):
        # 'Co' must not match the 'co' inside 'coordinate' (different case),
        # and \b prevents matching inside larger word tokens regardless.
        from src.inverse_design.substitute import substitute_text
        text = "Co(1) and Nb(2) are in a coordinate environment."
        out = substitute_text(text, "Co", "Nb", "Ni", "Ti")
        assert "coordinate environment" in out  # NOT mangled
        assert "Ni(1)" in out
        assert "Ti(2)" in out

    def test_refuses_to_substitute_oxygen(self):
        from src.inverse_design.substitute import substitute_text
        text = "Foo bar"
        with pytest.raises(ValueError):
            substitute_text(text, "O", "Nb", "Sr", "Ti")
        with pytest.raises(ValueError):
            substitute_text(text, "Li", "Nb", "O", "Ti")


# -------------------- proposer.py (parser only — no GPU) --------------------

class TestProposerParser:
    def test_parses_valid_json_block(self):
        from src.inverse_design.proposer import parse_proposals
        raw = (
            'Some preamble.\n'
            '[{"A": "Sr", "B": "Ti", "base": "mp-3731"}, '
            ' {"A": "Ba", "B": "Zr", "base": "mp-19051"}]\n'
            'Rationale: blah blah.'
        )
        a_vocab = {"Sr", "Ba", "Li"}
        b_vocab = {"Ti", "Zr", "Nb"}
        base_set = {"mp-3731", "mp-19051"}
        out = parse_proposals(raw, a_vocab=a_vocab, b_vocab=b_vocab, base_set=base_set)
        assert len(out) == 2
        assert out[0] == ("Sr", "Ti", "mp-3731")
        assert out[1] == ("Ba", "Zr", "mp-19051")

    def test_returns_empty_on_malformed_json(self):
        from src.inverse_design.proposer import parse_proposals
        out = parse_proposals(
            "no json here at all",
            a_vocab={"Sr"}, b_vocab={"Ti"}, base_set={"mp-1"},
        )
        assert out == []

    def test_filters_out_of_vocab_elements(self):
        from src.inverse_design.proposer import parse_proposals
        raw = (
            '[{"A": "Sr", "B": "Ti", "base": "mp-1"}, '
            ' {"A": "Xx", "B": "Ti", "base": "mp-1"}, '
            ' {"A": "Sr", "B": "Yy", "base": "mp-1"}, '
            ' {"A": "Sr", "B": "Ti", "base": "mp-NOT-IN-VOCAB"}]'
        )
        out = parse_proposals(
            raw,
            a_vocab={"Sr"}, b_vocab={"Ti"}, base_set={"mp-1"},
        )
        assert len(out) == 1
        assert out[0] == ("Sr", "Ti", "mp-1")

    def test_random_fallback_returns_n_candidates(self):
        from src.inverse_design.proposer import random_fallback_propose
        a_vocab = ["Sr", "Ba", "Ca", "Li"]
        b_vocab = ["Ti", "Zr", "Nb"]
        base_set = ["mp-1", "mp-2"]
        import random
        random.seed(0)
        cands = random_fallback_propose(n=5, a_vocab=a_vocab, b_vocab=b_vocab,
                                        base_set=base_set)
        assert len(cands) == 5
        for a, b, mid in cands:
            assert a in a_vocab
            assert b in b_vocab
            assert mid in base_set


# -------------------- agent.py (with stub oracle/proposer) --------------------

class TestAgent:
    def test_records_carry_iter_and_error(self, tmp_path, monkeypatch):
        """Agent loop should populate CandidateRecord across iter 0..n-1."""
        import pandas as pd

        from src.inverse_design.agent import (
            CandidateRecord, InverseDesignAgent,
        )

        # Build a minimal base_df with 3 valid entries.
        cif_p = PROJECT_ROOT / "data/processed/multimodal_v1/cifs/mp-3731.cif"
        if not cif_p.exists():
            pytest.skip("Sample CIF missing")
        df = pd.DataFrame([
            {"material_id": "mp-3731", "a_site": "Li", "b_site": "Nb",
             "cif_path": str(cif_p), "robocrys_text": "Li(1) bonded to O(1).",
             "band_gap": 3.336},
        ])
        a_vocab = ["Sr", "Ba"]
        b_vocab = ["Ti", "Zr"]

        class StubOracle:
            def predict(self, cif_path, text):
                return {"band_gap_mean": 2.0, "band_gap_std": 0.1,
                        "is_metal_prob": 0.05}

        class StubProposer:
            def propose(self, target_bg, history, a_vocab, b_vocab, base_set,
                        n_proposals):
                return [("Sr", "Ti", "mp-3731")] * n_proposals

        agent = InverseDesignAgent(
            oracle=StubOracle(), proposer=StubProposer(),
            base_df=df, target_bg=2.0,
            n_iter=2, n_per_iter=2,
            workdir=tmp_path,
            a_vocab=a_vocab, b_vocab=b_vocab,
            seed=0,
        )
        records = agent.run()

        assert len(records) == 2 * 2  # n_iter * n_per_iter
        assert all(isinstance(r, CandidateRecord) for r in records)
        iters = sorted({r.iter for r in records})
        assert iters == [0, 1]
        # error_to_target = |pred_mean - target|
        for r in records:
            assert r.error_to_target == pytest.approx(abs(r.pred_bg_mean - 2.0))


# -------------------- Slow / GPU tests (skipped unless --runslow) --------------------

@pytest.mark.slow
def test_oracle_loads_and_predicts():
    """Smoke: load 1 ckpt, predict on a known training material, sanity-check error."""
    from src.inverse_design.oracle import EnsembleForwardOracle
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Needs GPU")

    ckpt_dir = PROJECT_ROOT / "results/exp5b_multitask_film_nowarm"
    if not ckpt_dir.exists():
        pytest.skip(f"Missing ckpt dir: {ckpt_dir}")

    import pandas as pd
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/multimodal_v1/dataset.parquet")
    df = df[df["text_ok"] == True].reset_index(drop=True)
    row = df.iloc[0]

    oracle = EnsembleForwardOracle(str(ckpt_dir), device="cuda")
    out = oracle.predict(row["cif_path"], row["robocrys_text"])
    assert "band_gap_mean" in out
    assert "band_gap_std" in out
    assert "is_metal_prob" in out
    err = abs(out["band_gap_mean"] - float(row["band_gap"]))
    # 5-seed ensemble train MAE ≈ 0.49, so first sample (likely train) should be tight
    assert err < 1.5, f"Unexpectedly large oracle error {err:.3f} on a training sample"
