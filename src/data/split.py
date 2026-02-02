# src/data/split.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def stratified_subject_split(
    subject_uids: List[str],
    subject_labels: List[int],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 45,
) -> Dict[str, List[str]]:
    """
    Split by SUBJECT (uids), stratified by label.
    Return dict: {train_uids, val_uids, test_uids}
    """
    rng = np.random.default_rng(int(seed))
    uids = np.asarray(subject_uids, dtype=object)
    labels = np.asarray(subject_labels, dtype=np.int32)

    train_uids: List[str] = []
    val_uids: List[str] = []
    test_uids: List[str] = []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)

        n = int(idx.size)
        n_tr = int(round(n * float(ratios[0])))
        n_va = int(round(n * float(ratios[1])))

        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)

        tr = idx[:n_tr]
        va = idx[n_tr:n_tr + n_va]
        te = idx[n_tr + n_va:]

        train_uids.extend(uids[tr].tolist())
        val_uids.extend(uids[va].tolist())
        test_uids.extend(uids[te].tolist())

    rng.shuffle(train_uids)
    rng.shuffle(val_uids)
    rng.shuffle(test_uids)

    return {"train_uids": train_uids, "val_uids": val_uids, "test_uids": test_uids}


def split_by_dataset_then_merge(
    subject_index: List[dict],
    ratios: Tuple[float, float, float],
    seed: int,
    allowed_labels: Tuple[int, ...],
) -> Dict[str, object]:
    """
    subject_index item format (from your sidecar json):
      single: {uid, label, key, ...}
      multi:  {uid, label, key, dataset, ...}

    Split per dataset (stratified), then merge.
    """
    # group subjects by dataset
    by_ds: Dict[str, List[dict]] = {}
    for it in subject_index:
        lb = int(it.get("label", -999))
        if lb not in allowed_labels:
            continue
        ds = str(it.get("dataset", it.get("dataset_name", it.get("dataset_id", "single"))))
        by_ds.setdefault(ds, []).append(it)

    merged = {"train_uids": [], "val_uids": [], "test_uids": []}
    per_dataset = {}

    for ds, items in sorted(by_ds.items(), key=lambda x: x[0]):
        uids = [str(x["uid"]) for x in items]
        lbs = [int(x["label"]) for x in items]
        sp = stratified_subject_split(uids, lbs, ratios=ratios, seed=seed)
        merged["train_uids"].extend(sp["train_uids"])
        merged["val_uids"].extend(sp["val_uids"])
        merged["test_uids"].extend(sp["test_uids"])

        per_dataset[ds] = {
            "n_subjects": len(items),
            "train": len(sp["train_uids"]),
            "val": len(sp["val_uids"]),
            "test": len(sp["test_uids"]),
            "label_counts": {
                "labels": {str(k): int(sum(1 for v in lbs if v == k)) for k in sorted(set(lbs))}
            }
        }

    return {"splits": merged, "per_dataset": per_dataset}
