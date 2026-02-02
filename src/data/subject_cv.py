# src/data/subject_cv.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def _subject_labels_from_segments(y: np.ndarray, sid: np.ndarray) -> Dict[Any, int]:
    """
    label per subject = majority label from its segments.
    Assumes binary labels {0,1}.
    """
    mp: Dict[Any, list] = {}
    for lb, s in zip(y.tolist(), sid.tolist()):
        mp.setdefault(s, []).append(int(lb))

    out: Dict[Any, int] = {}
    for s, lbs in mp.items():
        # majority
        c0 = sum(1 for v in lbs if v == 0)
        c1 = len(lbs) - c0
        out[s] = 1 if c1 >= c0 else 0
    return out


def make_subject_folds(
    y: np.ndarray,
    sid: np.ndarray,
    n_folds: int = 5,
    seed: int = 45,
    stratify: bool = True,
) -> List[List[Any]]:
    """
    Return folds: list of list(subject_ids)
    Stratify at subject-level by label if stratify=True.
    """
    rng = np.random.default_rng(int(seed))
    sid = np.asarray(sid)
    y = np.asarray(y).astype(np.int64)

    subj_label = _subject_labels_from_segments(y, sid)
    subjects = list(subj_label.keys())

    folds: List[List[Any]] = [[] for _ in range(int(n_folds))]

    if stratify:
        # split subjects by class then round-robin assign
        cls0 = [s for s in subjects if subj_label[s] == 0]
        cls1 = [s for s in subjects if subj_label[s] == 1]
        rng.shuffle(cls0)
        rng.shuffle(cls1)

        i = 0
        for s in cls0:
            folds[i % n_folds].append(s)
            i += 1
        j = 0
        for s in cls1:
            folds[j % n_folds].append(s)
            j += 1
    else:
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            folds[i % n_folds].append(s)

    return folds


def split_fold_rotate_60202(
    folds: List[List[Any]],
    fold_idx: int,
) -> Tuple[set, set, set]:
    """
    For fold i:
      test = folds[i]
      val  = folds[(i+1)%K]
      train= others
    """
    K = len(folds)
    i = int(fold_idx) % K
    test = set(folds[i])
    val = set(folds[(i + 1) % K])
    train = set()
    for j in range(K):
        if j == i or j == (i + 1) % K:
            continue
        train.update(folds[j])
    return train, val, test


def indices_for_subjects(sid: np.ndarray, subject_set: set) -> np.ndarray:
    sid = np.asarray(sid)
    mask = np.isin(sid, np.array(list(subject_set), dtype=sid.dtype if sid.dtype != object else object))
    return np.where(mask)[0].astype(np.int64)
