"""Tests for Exp 6: Cross-attention token fusion (per-atom Q × per-token K/V)."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.data.crystal_graph import NUM_ATOM_FEATURES


# -------------------- Helper-function unit tests --------------------

def test_gather_atoms_per_crystal_shapes_and_values():
    """gather_atoms_per_crystal: pad concatenated atoms back into per-crystal tensors."""
    from src.models.crossattn_fusion import gather_atoms_per_crystal

    # 3 crystals with atom counts [2, 4, 1]
    node_emb = torch.tensor([
        [1.0, 2.0],   # crystal 0, atom 0
        [3.0, 4.0],   # crystal 0, atom 1
        [5.0, 6.0],   # crystal 1, atom 0
        [7.0, 8.0],   # crystal 1, atom 1
        [9.0, 10.0],  # crystal 1, atom 2
        [11.0, 12.0], # crystal 1, atom 3
        [13.0, 14.0], # crystal 2, atom 0
    ])
    crystal_atom_idx = torch.tensor([0, 0, 1, 1, 1, 1, 2], dtype=torch.long)

    out, mask = gather_atoms_per_crystal(node_emb, crystal_atom_idx, batch_size=3)
    assert out.shape == (3, 4, 2)  # max atoms = 4
    assert mask.shape == (3, 4)
    # Crystal 0: 2 valid atoms
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[0, 1], torch.tensor([3.0, 4.0]))
    assert mask[0].tolist() == [True, True, False, False]
    # Crystal 1: 4 valid atoms
    assert mask[1].tolist() == [True, True, True, True]
    assert torch.allclose(out[1, 3], torch.tensor([11.0, 12.0]))
    # Crystal 2: 1 valid atom, padded
    assert mask[2].tolist() == [True, False, False, False]


# -------------------- CGCNN.get_node_embedding test --------------------

def test_cgcnn_get_node_embedding_returns_per_atom():
    """New CGCNN.get_node_embedding(...) returns (N_total, atom_fea_len) per-atom embeddings."""
    from src.models.cgcnn import CGCNN

    torch.manual_seed(0)
    model = CGCNN(orig_atom_fea_len=NUM_ATOM_FEATURES, atom_fea_len=64,
                  nbr_fea_len=41, n_conv=3)
    model.eval()
    N_total, M = 10, 12
    atom_fea = torch.randn(N_total, NUM_ATOM_FEATURES)
    nbr_fea = torch.randn(N_total, M, 41)
    nbr_idx = torch.zeros(N_total, M, dtype=torch.long)

    with torch.no_grad():
        node_emb = model.get_node_embedding(atom_fea, nbr_fea, nbr_idx)
    assert node_emb.shape == (N_total, 64), f"got {node_emb.shape}"


# -------------------- CrossAttnFusion model tests --------------------

def test_crossattn_metal_head_zero_init():
    """CrossAttnFusion final layers (bandgap + metal heads) should be zero-init for stability."""
    from src.models.crossattn_fusion import CrossAttnFusion

    model = CrossAttnFusion(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    assert torch.all(model.metal_head[-1].weight == 0)
    assert torch.all(model.metal_head[-1].bias == 0)
    assert torch.all(model.bandgap_head[-1].weight == 0)
    assert torch.all(model.bandgap_head[-1].bias == 0)


def test_crossattn_loads_pretrained_cgcnn_state():
    """CGCNN warm-start ckpt should load into CrossAttnFusion.cgcnn_trunk via strict=False."""
    from src.models.cgcnn import CGCNN
    from src.models.crossattn_fusion import CrossAttnFusion

    cgcnn = CGCNN(orig_atom_fea_len=NUM_ATOM_FEATURES)
    crossattn = CrossAttnFusion(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    miss, unexp = crossattn.cgcnn_trunk.load_state_dict(cgcnn.state_dict(), strict=False)
    # CGCNN has (embedding, convs, conv_to_fc, hidden, fc_out); cgcnn_trunk reuses all.
    assert len(miss) == 0, f"expected no missing keys, got: {miss}"
    assert len(unexp) == 0, f"expected no unexpected keys, got: {unexp}"


# -------------------- Forward shape test (slow: BERT instantiation) --------------------

@pytest.mark.slow
def test_crossattn_forward_returns_two_tensors():
    """Forward returns (B,) bandgap_pred and (B,) metal_logit."""
    from src.models.crossattn_fusion import CrossAttnFusion

    torch.manual_seed(0)
    model = CrossAttnFusion(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    model.eval()

    # Synthetic batch=2, atoms-per-crystal=[3, 5]
    atom_fea = torch.randn(8, NUM_ATOM_FEATURES)
    nbr_fea = torch.randn(8, 12, 41)
    nbr_idx = torch.zeros(8, 12, dtype=torch.long)
    crystal_atom_idx = torch.tensor([0]*3 + [1]*5, dtype=torch.long)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    with torch.no_grad():
        bandgap_pred, metal_logit = model(
            atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
            input_ids, attention_mask,
        )
    assert bandgap_pred.shape == (2,), f"bandgap shape: {bandgap_pred.shape}"
    assert metal_logit.shape == (2,), f"metal shape: {metal_logit.shape}"
