"""Tests for Exp 8: 3-modality fusion (CGCNN + SciBERT + ResNet18 image)."""

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


# -------------------- Image encoder --------------------

def test_image_encoder_output_dim_and_zero_init():
    from src.models.image_encoder import FrozenResNet18Encoder

    enc = FrozenResNet18Encoder(out_dim=64, freeze_trunk=True)
    # Final projection layer zero-init
    assert torch.all(enc.proj[-1].weight == 0)
    assert torch.all(enc.proj[-1].bias == 0)
    # Trunk frozen
    for p in enc.trunk.parameters():
        assert not p.requires_grad
    # Forward shape
    x = torch.randn(2, 3, 224, 224)
    enc.eval()
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 64)
    # zero-init → output is zero before training
    assert torch.allclose(out, torch.zeros_like(out))


# -------------------- Three-modality model --------------------

@pytest.mark.slow
def test_three_modality_forward_shapes():
    from src.models.three_modality_fusion import ThreeModalityFusion

    torch.manual_seed(0)
    model = ThreeModalityFusion(
        orig_atom_fea_len=NUM_ATOM_FEATURES,
        bert_unfreeze_last_n=2,
        image_dim=64,
    )
    model.eval()

    atom_fea = torch.randn(8, NUM_ATOM_FEATURES)
    nbr_fea = torch.randn(8, 12, 41)
    nbr_idx = torch.zeros(8, 12, dtype=torch.long)
    crystal_atom_idx = torch.tensor([0]*4 + [1]*4, dtype=torch.long)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        bg, ml = model(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
                       input_ids, attention_mask, images)
    assert bg.shape == (2,)
    assert ml.shape == (2,)


def test_three_modality_loads_multitask_film_state_dict_with_head_filter():
    """Loading a MultiTaskFiLM checkpoint into ThreeModalityFusion is expected
    to need head/metal_head filtering because the new heads consume an
    additional image_dim of input. After filtering, all encoder weights
    (cgcnn.*, bert.*, proj_*, film_gen.*, mod_norm.*) load cleanly."""
    from src.models.multitask_fusion import MultiTaskFiLM
    from src.models.three_modality_fusion import ThreeModalityFusion

    mt = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    tm = ThreeModalityFusion(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2,
                              image_dim=64)
    src_state = {k: v for k, v in mt.state_dict().items()
                 if not k.startswith(("head.", "metal_head."))}
    miss, unexp = tm.load_state_dict(src_state, strict=False)
    # All missing should be the new image branch + (newly-shaped) heads
    expected_prefixes = ("image_encoder.", "head.", "metal_head.")
    bad_missing = [k for k in miss if not k.startswith(expected_prefixes)]
    assert not bad_missing, f"unexpected missing keys: {bad_missing}"
    assert not unexp, f"unexpected: {unexp}"


# -------------------- Dataset (uses pre-rendered PNGs) --------------------

@pytest.fixture(scope="module")
def small_df():
    from src.data.loader import load_dataset
    df = load_dataset(
        str(PROJECT_ROOT / "data/processed/multimodal_v1/dataset.parquet"),
        text_ok_only=True,
    )
    return df.head(2).reset_index(drop=True)


@pytest.mark.slow
def test_three_modality_dataset_returns_8_tuple(small_df):
    """If render_cifs_to_png.py has been run, dataset returns 8-tuple per item."""
    from src.data.three_modality_dataset import ThreeModalityFusionDataset

    img_dir = PROJECT_ROOT / "data/processed/multimodal_v1/images"
    if not img_dir.exists() or not (img_dir / f"{small_df.iloc[0]['material_id']}.png").exists():
        pytest.skip("Pre-rendered images not yet available")

    ds = ThreeModalityFusionDataset(small_df, target_col="band_gap",
                                     project_root=PROJECT_ROOT)
    item = ds[0]
    assert len(item) == 8
    image_t = item[7]
    assert image_t.dim() == 3 and image_t.shape[0] == 3


@pytest.mark.slow
def test_collate_three_modality_returns_9_tuple(small_df):
    from src.data.three_modality_dataset import ThreeModalityFusionDataset, collate_three_modality

    img_dir = PROJECT_ROOT / "data/processed/multimodal_v1/images"
    if not img_dir.exists() or not (img_dir / f"{small_df.iloc[0]['material_id']}.png").exists():
        pytest.skip("Pre-rendered images not yet available")

    ds = ThreeModalityFusionDataset(small_df, target_col="band_gap",
                                     project_root=PROJECT_ROOT)
    out = collate_three_modality([ds[0], ds[1]])
    assert len(out) == 9
    images = out[8]
    assert images.shape[0] == 2 and images.shape[1] == 3
