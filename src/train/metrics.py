from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else float("nan")


def _rankdata_with_ties(a: np.ndarray) -> np.ndarray:
    # ranks start at 1; ties get average rank
    a = np.asarray(a, dtype=float)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    sorted_a = a[order]

    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=float)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _rankdata_with_ties(y_score)
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def confusion_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metrics: List[str]) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    out: Dict[str, float] = {}
    for m in metrics:
        if m == "acc":
            out["acc"] = accuracy(y_true, y_pred)
        elif m == "f1":
            out["f1"] = f1_binary(y_true, y_pred, pos_label=1)
        elif m == "auc":
            out["auc"] = roc_auc_binary(y_true, y_prob)
        else:
            raise ValueError(f"Unknown metric: {m}")
    return out
