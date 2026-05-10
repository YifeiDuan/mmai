"""Tests for Exp 5: Multi-task FiLM (band_gap regression + is_metal classification)."""

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
from src.data.loader import load_dataset


# -------------------- Fixtures --------------------

@pytest.fixture(scope="module")
def small_df():
    """Two text_ok=True rows from canonical dataset."""
    df = load_dataset(
        str(PROJECT_ROOT / "data/processed/multimodal_v1/dataset.parquet"),
        text_ok_only=True,
    )
    return df.head(2).reset_index(drop=True)


# -------------------- Pure-tensor unit tests --------------------

def test_multitask_loss_perfect_prediction_is_near_zero():
    """Huber + 0.3*BCE should be ~0 when both predictions match targets."""
    from src.models.multitask_fusion import multitask_loss

    bandgap_pred = torch.tensor([1.0, 2.0, 3.0])
    target_bg = torch.tensor([1.0, 2.0, 3.0])
    metal_logit = torch.tensor([10.0, -10.0, 10.0])  # confident & correct
    target_metal = torch.tensor([1.0, 0.0, 1.0])

    total, l_reg, l_metal = multitask_loss(
        bandgap_pred, metal_logit, target_bg, target_metal,
        alpha_metal=0.3, huber_delta=0.5,
    )
    assert l_reg.item() < 1e-6, f"Expected near-zero Huber loss, got {l_reg.item()}"
    assert l_metal.item() < 1e-3, f"Expected near-zero BCE loss, got {l_metal.item()}"
    assert torch.isclose(total, l_reg + 0.3 * l_metal)


def test_multitask_loss_huber_caps_outliers():
    """Huber ≈ |x| once error > delta; ensures we get linear scaling for outliers."""
    from src.models.multitask_fusion import multitask_loss

    # error of 4 eV with delta=0.5 → Huber = 0.5 * 0.5 + (4-0.5)*0.5 = 1.875
    bandgap_pred = torch.tensor([4.0])
    target_bg = torch.tensor([0.0])
    metal_logit = torch.tensor([0.0])
    target_metal = torch.tensor([0.0])

    _, l_reg, _ = multitask_loss(
        bandgap_pred, metal_logit, target_bg, target_metal,
        alpha_metal=0.0, huber_delta=0.5,
    )
    # MSE would give 16.0; Huber should give 1.875
    assert 1.5 < l_reg.item() < 2.0, f"Huber outlier loss = {l_reg.item()}, expected ~1.875"


def test_ensemble_average_predictions():
    """Average of [0,2,4] and [2,4,6] = [1,3,5]."""
    from src.evaluation.ensemble import average_predictions
    import numpy as np

    p1 = np.array([0.0, 2.0, 4.0])
    p2 = np.array([2.0, 4.0, 6.0])
    avg = average_predictions([p1, p2])
    assert np.allclose(avg, [1.0, 3.0, 5.0])


def test_ensemble_handles_torch_and_numpy():
    """average_predictions accepts both torch tensors and numpy arrays."""
    from src.evaluation.ensemble import average_predictions
    import numpy as np

    p1 = torch.tensor([1.0, 2.0])
    p2 = np.array([3.0, 4.0])
    avg = average_predictions([p1, p2])
    assert isinstance(avg, np.ndarray)
    assert np.allclose(avg, [2.0, 3.0])


# -------------------- Model structural tests --------------------

def test_multitask_film_metal_head_zero_init():
    """Zero-init final layer of metal_head → starts neutral (sigmoid 0.5)."""
    from src.models.multitask_fusion import MultiTaskFiLM

    model = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    last = model.metal_head[-1]
    assert torch.all(last.weight == 0), "metal_head final weight should be zero-init"
    assert torch.all(last.bias == 0), "metal_head final bias should be zero-init"


def test_multitask_film_loads_film_state_dict():
    """LateFusionFiLM checkpoint should load into MultiTaskFiLM with strict=False."""
    from src.models.fusion import LateFusionFiLM
    from src.models.multitask_fusion import MultiTaskFiLM

    film = LateFusionFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES)
    mt = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES)
    missing, unexpected = mt.load_state_dict(film.state_dict(), strict=False)
    assert all(k.startswith("metal_head") for k in missing), \
        f"Only metal_head.* should be missing, got: {missing}"
    assert len(unexpected) == 0, f"No unexpected keys, got: {unexpected}"


# -------------------- Dataset tests --------------------

def test_multitask_dataset_returns_7_tuple(small_df):
    from src.data.multitask_fusion_dataset import MultiTaskFusionDataset

    ds = MultiTaskFusionDataset(small_df, target_col="band_gap")
    item = ds[0]
    assert len(item) == 7, f"Expected 7 elements, got {len(item)}: {[type(x) for x in item]}"
    atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask, target_t, is_metal_t = item
    assert atom_fea.dtype == torch.float32
    assert is_metal_t.dtype == torch.float32
    assert is_metal_t.shape == torch.Size([1])
    assert is_metal_t.item() in (0.0, 1.0), f"is_metal must be 0 or 1, got {is_metal_t.item()}"


def test_multitask_collate_8_tuple(small_df):
    from src.data.multitask_fusion_dataset import MultiTaskFusionDataset, collate_multitask_fusion

    ds = MultiTaskFusionDataset(small_df, target_col="band_gap")
    batch = [ds[0], ds[1]]
    out = collate_multitask_fusion(batch)
    assert len(out) == 8, f"Expected 8-tuple, got {len(out)}"
    *_, targets, is_metal = out
    assert is_metal.shape == (2,)
    assert targets.shape == (2,)
    assert is_metal.dtype == torch.float32


# -------------------- Forward shape test (slow: instantiates SciBERT) --------------------

@pytest.mark.slow
def test_multitask_film_forward_returns_two_tensors():
    """Real forward: (B,) bandgap_pred and (B,) metal_logit."""
    from src.models.multitask_fusion import MultiTaskFiLM

    torch.manual_seed(0)
    model = MultiTaskFiLM(orig_atom_fea_len=NUM_ATOM_FEATURES, bert_unfreeze_last_n=2)
    model.eval()

    # Synthetic batch=2, total atoms=10 (5 per crystal)
    atom_fea = torch.randn(10, NUM_ATOM_FEATURES)
    nbr_fea = torch.randn(10, 12, 41)
    nbr_idx = torch.zeros(10, 12, dtype=torch.long)
    crystal_atom_idx = torch.tensor([0]*5 + [1]*5, dtype=torch.long)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    with torch.no_grad():
        bandgap_pred, metal_logit = model(
            atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
            input_ids, attention_mask,
        )
    assert bandgap_pred.shape == (2,), f"bandgap shape: {bandgap_pred.shape}"
    assert metal_logit.shape == (2,), f"metal_logit shape: {metal_logit.shape}"
