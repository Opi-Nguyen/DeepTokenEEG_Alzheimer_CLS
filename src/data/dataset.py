import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def to_torch_dataset(X, y, s):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    s_t = torch.from_numpy(s).long()
    return TensorDataset(X_t, y_t, s_t)

def make_loaders(cache, batch_size, num_workers=0):
    train_ds = to_torch_dataset(cache["X_train"], cache["y_train"], cache["s_train"])
    val_ds   = to_torch_dataset(cache["X_val"], cache["y_val"], cache["s_val"])
    test_ds  = to_torch_dataset(cache["X_test"], cache["y_test"], cache["s_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def subject_average_probs(probs, targets, sids):
    """
    probs: (N,) probability for class=1
    targets: (N,)
    sids: (N,)
    returns: y_true_subj, y_prob_subj
    """
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    sids = np.asarray(sids)

    subj_probs = {}
    subj_label = {}
    for p, t, sid in zip(probs, targets, sids):
        if sid not in subj_probs:
            subj_probs[sid] = []
            subj_label[sid] = int(t)
        subj_probs[sid].append(float(p))

    s_list = list(subj_probs.keys())
    y_true = np.array([subj_label[s] for s in s_list], dtype=int)
    y_prob = np.array([np.mean(subj_probs[s]) for s in s_list], dtype=float)
    return y_true, y_prob
