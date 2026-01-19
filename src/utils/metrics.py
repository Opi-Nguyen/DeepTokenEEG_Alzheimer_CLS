import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_basic_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    accs, f1s, pres, recs = [], [], [], []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_basic_metrics(yt, yp)
        accs.append(m["accuracy"])
        f1s.append(m["f1_weighted"])
        pres.append(m["precision_weighted"])
        recs.append(m["recall_weighted"])

    return {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(pres)), "precision_std": float(np.std(pres)),
        "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
    }
