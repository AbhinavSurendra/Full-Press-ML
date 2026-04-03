"""PyTorch dataset utilities for possession-prefix sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceBatch:
    features: torch.Tensor
    labels: torch.Tensor


class PossessionSequenceDataset(Dataset):
    """Dataset for sequence models trained on possession prefixes."""

    def __init__(
        self,
        frame_table: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        max_len: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.max_len = max_len
        self.examples = []

        for _, group in frame_table.groupby("possession_id"):
            group = group.sort_values("frame_idx").head(max_len)
            sequence = group[feature_columns].to_numpy(dtype=np.float32)
            label = int(group[label_column].iloc[-1])
            self.examples.append((sequence, label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, label = self.examples[index]
        padded = np.zeros((self.max_len, len(self.feature_columns)), dtype=np.float32)
        padded[: len(sequence)] = sequence
        return torch.from_numpy(padded), torch.tensor(label, dtype=torch.long)

