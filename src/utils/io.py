import os
import json
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_npz(path: str, **arrays):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def load_npz(path: str):
    return np.load(path, allow_pickle=True)
