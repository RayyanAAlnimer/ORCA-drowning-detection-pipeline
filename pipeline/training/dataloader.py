from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TemporalSequenceDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt samples found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        sample = torch.load(self.files[index], map_location="cpu")
        x = sample["X"].float()
        y = torch.tensor(sample["y"], dtype=torch.float32)
        return x, y
