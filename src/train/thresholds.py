import numpy as np
import torch
from sklearn.metrics import f1_score
from src.data.dataset import subject_average_probs

def find_best_segment_threshold(model, val_loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y, _sid in val_loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())

    best_f1, best_thr = 0.0, 0.5
    all_probs = np.asarray(all_probs)
    all_targets = np.asarray(all_targets)

    for thr in np.arange(0.01, 1.0, 0.01):
        pred = (all_probs > thr).astype(int)
        f1 = f1_score(all_targets, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, float(best_f1)

def find_best_subject_threshold(model, val_loader, device):
    model.eval()
    probs, targets, sids = [], [], []
    with torch.no_grad():
        for x, y, sid in val_loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            targets.extend(y.numpy())
            sids.extend(sid.numpy())

    y_true_subj, y_prob_subj = subject_average_probs(probs, targets, sids)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.01, 1.0, 0.01):
        pred = (y_prob_subj > thr).astype(int)
        f1 = f1_score(y_true_subj, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, float(best_f1)
