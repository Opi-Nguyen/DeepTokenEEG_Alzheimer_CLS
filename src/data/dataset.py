# src/data/dataset.py
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IndexedEEGDataset(Dataset):
    """
    Hold base arrays (X,y,s,k) and index list to avoid copying.
    Returns: (x, y, s, k)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, s: np.ndarray, k: np.ndarray, indices: Sequence[int]):
        self.X = X
        self.y = y
        self.s = s
        self.k = k
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        sid = str(self.s[idx])
        kk = int(self.k[idx]) if self.k is not None else -1
        return x, y, sid, kk


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
