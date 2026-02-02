import numpy as np
import torch
from src.utils.metrics import bootstrap_metrics
from data.old.dataset_v1 import subject_average_probs

def predict_probs(model, loader, device):
    model.eval()
    probs, targets, sids = [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            targets.extend(y.numpy())
            sids.extend(sid.numpy())
    return np.asarray(probs), np.asarray(targets), np.asarray(sids)

def evaluate_with_thresholds(model, test_loader, device, seg_thr, subj_thr,
                             n_bootstraps=1000, seed=0):
    probs, targets, sids = predict_probs(model, test_loader, device)

    # segment-level
    seg_pred = (probs > seg_thr).astype(int)
    seg_report = bootstrap_metrics(targets, seg_pred, n_bootstraps=n_bootstraps, seed=seed)

    # subject-level
    y_true_subj, y_prob_subj = subject_average_probs(probs, targets, sids)
    subj_pred = (y_prob_subj > subj_thr).astype(int)
    subj_report = bootstrap_metrics(y_true_subj, subj_pred, n_bootstraps=n_bootstraps, seed=seed)

    return {
        "segment": {"threshold": float(seg_thr), **seg_report},
        "subject": {"threshold": float(subj_thr), **subj_report},
    }
