import random

def class_stratified_split(ids, ratios, seed):
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]

def make_subject_splits(subject_data, ratios, seed):
    class0 = [pid for pid, d in subject_data.items() if d["label"] == 0]
    class1 = [pid for pid, d in subject_data.items() if d["label"] == 1]

    train0, val0, test0 = class_stratified_split(class0, ratios, seed)
    train1, val1, test1 = class_stratified_split(class1, ratios, seed)

    train_ids = sorted(train0 + train1)
    val_ids = sorted(val0 + val1)
    test_ids = sorted(test0 + test1)

    return {
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "counts": {
            "train": {"hc": len(train0), "ad": len(train1), "total": len(train_ids)},
            "val":   {"hc": len(val0),   "ad": len(val1),   "total": len(val_ids)},
            "test":  {"hc": len(test0),  "ad": len(test1),  "total": len(test_ids)},
        }
    }
