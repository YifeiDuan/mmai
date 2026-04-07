"""Data loading utilities for the perovskite ABO3 dataset."""

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATASET_PATH = "data/processed/multimodal_v1/dataset.parquet"
DEFAULT_SPLIT_MAP_PATH = "data/interim/filtered_abo3/split_map.parquet"


def load_dataset(
    path: str = DEFAULT_DATASET_PATH,
    text_ok_only: bool = False,
) -> pd.DataFrame:
    """Load the main dataset parquet.

    Args:
        path: Path to dataset.parquet.
        text_ok_only: If True, keep only rows where text_ok == True.

    Returns:
        DataFrame with all columns from the processed dataset.
    """
    df = pd.read_parquet(path)
    if text_ok_only:
        df = df[df["text_ok"] == True].reset_index(drop=True)
    return df


def load_split_map(path: str = DEFAULT_SPLIT_MAP_PATH) -> pd.DataFrame:
    """Load the train/val/test split mapping."""
    return pd.read_parquet(path)


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter dataset by split name (train / val / test)."""
    return df[df["split"] == split].reset_index(drop=True)
