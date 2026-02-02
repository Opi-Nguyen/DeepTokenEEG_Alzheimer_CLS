# src/data/npz_dataloader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class BandCache:
    X: np.ndarray
    y: np.ndarray
    sid: np.ndarray          # subject id per segment (prefer k else s)
    sid_is_k: bool
    band_name: str
    npz_path: str
    json_path: Optional[str] = None


def load_band_npz(npz_path: str, band_name: str) -> BandCache:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    X_key = f"X_{band_name}"
    if X_key not in npz.files:
        raise KeyError(f"NPZ missing {X_key}. Available={npz.files}")

    X = np.asarray(npz[X_key])
    y = np.asarray(npz["y"]).astype(np.int64)

    sid_is_k = False
    if "k" in npz.files:
        sid = np.asarray(npz["k"])
        sid_is_k = True
    else:
        sid = np.asarray(npz["s"])

    # normalize sid type (keep ints if possible, else strings)
    if sid.dtype.kind in ("U", "S", "O"):
        sid = sid.astype(str)
    else:
        sid = sid.astype(np.int64)

    # X float32
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    return BandCache(
        X=X,
        y=y,
        sid=sid,
        sid_is_k=sid_is_k,
        band_name=band_name,
        npz_path=npz_path,
    )


class EEGSegmentDataset(Dataset):
    """
    Dataset view by indices.
    Returns: (x, y, sid)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, sid: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.sid = sid
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = torch.from_numpy(self.X[idx])               # (C, T)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        # sid keep as python int/str for grouping later
        sid = self.sid[idx].item() if hasattr(self.sid[idx], "item") else self.sid[idx]
        return x, y, sid


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
