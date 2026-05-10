"""Prediction ensembling utilities."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def average_predictions(preds: Sequence[ArrayLike]) -> np.ndarray:
    """Element-wise mean of K prediction arrays.

    Each input may be a torch.Tensor or numpy array of identical shape.
    Returns a numpy array with the same shape.
    """
    if len(preds) == 0:
        raise ValueError("average_predictions: empty sequence")
    arrs = [_to_numpy(p) for p in preds]
    shape = arrs[0].shape
    for i, a in enumerate(arrs[1:], 1):
        if a.shape != shape:
            raise ValueError(f"average_predictions: shape mismatch at index {i}: {a.shape} vs {shape}")
    return np.mean(np.stack(arrs, axis=0), axis=0)
