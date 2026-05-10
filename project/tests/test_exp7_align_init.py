"""Tests for Exp 7: ALBEF-style align-before-fuse — load CGCNN+BERT from
CLIPAlignmentModel checkpoint via the new --align-init flag."""

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


def test_align_state_dict_keys_compatible_with_multitaskfilm():
    """CLIPAlignmentModel state-dict's cgcnn.* + bert.* prefixes must match
    MultiTaskFiLM submodule names so strict=False loading populates everything
    expected (only proj_*/film_gen/heads remain randomly initialised)."""
    from src.models.alignment import CLIPAlignmentModel
    from src.models.multitask_fusion import MultiTaskFiLM

    align = CLIPAlignmentModel(orig_atom_fea_len=NUM_ATOM_FEATURES,
                                bert_unfreeze_last_n=2)
    mt = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES,
                       bert_unfreeze_last_n=2)

    align_state = align.state_dict()
    cgcnn_state = {k[len("cgcnn."):]: v for k, v in align_state.items()
                   if k.startswith("cgcnn.")}
    bert_state = {k[len("bert."):]: v for k, v in align_state.items()
                  if k.startswith("bert.")}

    miss_c, unexp_c = mt.cgcnn.load_state_dict(cgcnn_state, strict=False)
    miss_b, unexp_b = mt.bert.load_state_dict(bert_state, strict=False)

    # Loading should be lossless: every CGCNN key in MultiTaskFiLM is in align ckpt
    assert len(miss_c) == 0, f"CGCNN missing keys: {miss_c}"
    assert len(unexp_c) == 0, f"CGCNN unexpected keys: {unexp_c}"
    assert len(miss_b) == 0, f"BERT missing keys: {miss_b}"
    assert len(unexp_b) == 0, f"BERT unexpected keys: {unexp_b}"


def test_align_init_shapes_match():
    """Sanity: CGCNN inside MultiTaskFiLM has same parameter shapes as
    CGCNN inside CLIPAlignmentModel — so weights transfer 1:1."""
    from src.models.alignment import CLIPAlignmentModel
    from src.models.multitask_fusion import MultiTaskFiLM

    align = CLIPAlignmentModel(orig_atom_fea_len=NUM_ATOM_FEATURES,
                                atom_fea_len=64, n_conv=3,
                                cgcnn_h_fea_len=128, cgcnn_n_h=1)
    mt = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES,
                       atom_fea_len=64, n_conv=3,
                       cgcnn_h_fea_len=128, cgcnn_n_h=1)

    align_cgcnn_state = align.cgcnn.state_dict()
    mt_cgcnn_state = mt.cgcnn.state_dict()
    for k in mt_cgcnn_state:
        assert k in align_cgcnn_state, f"{k} not in alignment CGCNN"
        assert mt_cgcnn_state[k].shape == align_cgcnn_state[k].shape, \
            f"{k}: shape mismatch {mt_cgcnn_state[k].shape} vs {align_cgcnn_state[k].shape}"
