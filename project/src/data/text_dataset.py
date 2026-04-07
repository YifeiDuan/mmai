"""Text dataset for SciBERT-based property prediction."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextRegressionDataset(Dataset):
    """Tokenized text + regression target dataset."""

    def __init__(
        self,
        df,
        target_col: str = "band_gap",
        text_col: str = "robocrys_text",
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_seq_len: int = 256,
        tokenizer=None,
    ):
        # Only keep rows with valid text
        self.df = df[df["text_ok"] == True].reset_index(drop=True)
        self.target_col = target_col
        self.text_col = text_col
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        target = float(row[self.target_col])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor([target], dtype=torch.float32),
        }
