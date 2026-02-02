# scripts/eval.py
import os
import re
import argparse
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.io import load_json, load_npz, save_json
from src.data.dataset import IndexedEEGDataset, make_loader
from src.models.model import Model
"""
python -m scripts.eval \
  --run_dir /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/outputs/runs/alpha_swt_resnet/single/ADFTD/alpha/model1/seed_45/blocks_1_dilations_2_2_2_2_2/ \
  --cfg_model configs/model1.yaml

"""
def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _best_thr_f1(y_true: np.ndarray, p: np.ndarray, thrs=None) -> dict:
    if thrs is None:
        thrs = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        pred = (p >= t).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(t)
    return {"thr": float(best_thr), "f1": float(best_f1)}


def _metrics_binary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def _collect_uid_probs(uids, probs_ad, labels):
    mp = {}
    for uid, p, y in zip(uids, probs_ad, labels):
        mp.setdefault(uid, {"ps": [], "ys": []})
        mp[uid]["ps"].append(float(p))
        mp[uid]["ys"].append(int(y))

    subj_uid, subj_p, subj_y = [], [], []
    for uid, v in mp.items():
        subj_uid.append(uid)
        subj_p.append(float(np.mean(v["ps"])))
        ys = v["ys"]
        subj_y.append(int(max(set(ys), key=ys.count)))
    return subj_uid, np.asarray(subj_p, dtype=np.float32), np.asarray(subj_y, dtype=np.int32)


def _build_uid2idx(s: np.ndarray) -> dict:
    uid2idx = {}
    for i, uid in enumerate(s.tolist()):
        uid2idx.setdefault(uid, []).append(i)
    return uid2idx


def _idx_from_uids(uids, uid2idx) -> np.ndarray:
    out = []
    for uid in uids:
        out.extend(uid2idx.get(uid, []))
    return np.asarray(out, dtype=np.int64)


def _extract_model_cfg(snapshot: dict) -> dict | None:
    if not isinstance(snapshot, dict):
        return None

    # direct
    m = snapshot.get("model")
    if isinstance(m, dict):
        if "d_model" in m and "resnet" in m:
            return m
        if "model" in m and isinstance(m["model"], dict):
            mm = m["model"]
            if "d_model" in mm and "resnet" in mm:
                return mm

    # common alternatives
    for key in ["cfg_model", "model_cfg", "config_model", "model_yaml", "cfg"]:
        v = snapshot.get(key)
        if isinstance(v, dict):
            if "model" in v and isinstance(v["model"], dict):
                mm = v["model"]
                if "d_model" in mm and "resnet" in mm:
                    return mm
            if "d_model" in v and "resnet" in v:
                return v

    # nested
    for key in ["config", "snapshot", "run_info"]:
        v = snapshot.get(key)
        if isinstance(v, dict):
            mm = _extract_model_cfg(v)
            if mm is not None:
                return mm

    return None


def _try_load_state_dict(best_path: str, device):
    # avoid warning if available
    try:
        obj = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(best_path, map_location=device)

    # some checkpoints save {"state_dict": ...}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    else:
        raise RuntimeError(f"Unsupported checkpoint format in {best_path}")

    # strip DataParallel prefix
    if len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    return sd


def _parse_blocks_and_dilations_from_run_dir(run_dir: str):
    """
    Expect folder part like: blocks_1_dilations_2_2_2_2_2
    Return (n_blocks:int|None, dilations:list[int]|None)
    """
    base = os.path.basename(os.path.normpath(run_dir))
    m = re.search(r"blocks_(\d+)_dilations_([0-9_]+)", base)
    if not m:
        return None, None
    n_blocks = int(m.group(1))
    dils = [int(x) for x in m.group(2).split("_") if x.strip() != ""]
    return n_blocks, dils


def _infer_n_blocks_from_state_dict(sd: dict) -> int | None:
    idxs = set()
    for k in sd.keys():
        m = re.match(r"res_blocks\.(\d+)\.", k)
        if m:
            idxs.add(int(m.group(1)))
    if not idxs:
        return None
    return max(idxs) + 1


def main(run_dir: str, cfg_model_fallback: str | None = None, batch_size: int = 64, num_workers: int = 2):
    run_dir = os.path.abspath(run_dir)

    splits_path = os.path.join(run_dir, "splits_subject.json")
    snapshot_path = os.path.join(run_dir, "config_snapshot.json")
    best_path = os.path.join(run_dir, "best.pth")

    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing: {splits_path}")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing: {best_path}")

    split_info = load_json(splits_path)

    snap = load_json(snapshot_path) if os.path.exists(snapshot_path) else {}
    model_cfg = _extract_model_cfg(snap)

    if model_cfg is None:
        if cfg_model_fallback is None:
            raise KeyError(
                f"config_snapshot.json missing model config and no --cfg_model fallback.\n"
                f"snapshot_path={snapshot_path}"
            )
        model_cfg = _load_yaml(cfg_model_fallback).get("model")
        if model_cfg is None:
            raise KeyError(f"Fallback yaml has no top-level 'model': {cfg_model_fallback}")

    # load checkpoint state_dict first (for robust n_blocks inference)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = _try_load_state_dict(best_path, device)

    # determine correct n_blocks / dilations for THIS run
    n_blocks_dir, dils_dir = _parse_blocks_and_dilations_from_run_dir(run_dir)
    n_blocks_sd = _infer_n_blocks_from_state_dict(state_dict)

    n_blocks = n_blocks_dir if n_blocks_dir is not None else (n_blocks_sd if n_blocks_sd is not None else int(model_cfg["resnet"]["n_blocks"]))
    dils = dils_dir if dils_dir is not None else list(model_cfg["resnet"]["dilations"])

    if len(dils) < n_blocks:
        dils = dils + [dils[-1]] * (n_blocks - len(dils))
    dils = dils[:n_blocks]

    # override model_cfg to match checkpoint
    model_cfg = dict(model_cfg)
    model_cfg["resnet"] = dict(model_cfg["resnet"])
    model_cfg["resnet"]["n_blocks"] = int(n_blocks)
    model_cfg["resnet"]["dilations"] = dils

    # ---- load cache ----
    cache_npz = split_info["cache"]["npz"]
    cache_json = split_info["cache"]["json"]
    band = split_info["cache"]["band"]
    labels_cfg = split_info.get("labels_cfg", {"HC": 0, "AD": 1})
    num_class = int(split_info.get("num_class", model_cfg.get("num_class", 2)))

    ad_index = int(labels_cfg.get("AD", 1))
    if ad_index < 0 or ad_index >= num_class:
        raise RuntimeError(f"Invalid AD index={ad_index} for num_class={num_class} labels_cfg={labels_cfg}")

    cache = load_npz(cache_npz)
    cache_dict = {k: cache[k] for k in cache.files}
    sidecar = load_json(cache_json)

    X_key = f"X_{band}"
    if X_key not in cache_dict:
        raise KeyError(f"NPZ missing {X_key}. keys={list(cache_dict.keys())}")

    X = cache_dict[X_key]
    y = cache_dict["y"].astype(np.int32)
    s = cache_dict["s"].astype(str)
    k = cache_dict.get("k", None)

    uid2idx = _build_uid2idx(s)
    val_uids = split_info["splits"]["val_uids"]
    test_uids = split_info["splits"]["test_uids"]

    val_idx = _idx_from_uids(val_uids, uid2idx)
    test_idx = _idx_from_uids(test_uids, uid2idx)

    if val_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(f"Empty val/test segments. val_idx={val_idx.size}, test_idx={test_idx.size}")

    # loaders
    val_ds = IndexedEEGDataset(X, y, s, k, val_idx)
    test_ds = IndexedEEGDataset(X, y, s, k, test_idx)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- build model consistent with checkpoint ----
    enc_in = len(sidecar.get("channels_19", [])) or 19
    model = Model(
        enc_in=int(enc_in),
        num_class=int(num_class),
        d_model=int(model_cfg["d_model"]),
        dropout=float(model_cfg["dropout"]),
        n_blocks=int(model_cfg["resnet"]["n_blocks"]),
        dilations=model_cfg["resnet"]["dilations"],
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def predict(loader):
        all_logits, all_y, all_s = [], [], []
        with torch.no_grad():
            for batch in loader:
                x, yy, ss, _kk = batch
                x = x.to(device, non_blocking=True)
                logits = model(x).detach().cpu().numpy()
                all_logits.append(logits)
                all_y.append(yy.numpy())
                all_s.extend(list(ss))

        logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, num_class), dtype=np.float32)
        yy = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int32)
        probs = _softmax_np(logits)
        p_ad = probs[:, ad_index] if probs.size else np.zeros((0,), dtype=np.float32)
        return yy.astype(np.int32), np.asarray(all_s, dtype=str), p_ad.astype(np.float32)

    # --- validation thresholds ---
    yv, sv, pv = predict(val_loader)
    seg_best = _best_thr_f1(yv, pv)
    seg_thr = seg_best["thr"]

    _subj_uid_v, subj_p_v, subj_y_v = _collect_uid_probs(sv.tolist(), pv.tolist(), yv.tolist())
    subj_best = _best_thr_f1(subj_y_v, subj_p_v)
    subj_thr = subj_best["thr"]

    # --- test eval ---
    yt, st, pt = predict(test_loader)

    seg_pred = (pt >= seg_thr).astype(np.int32)
    seg_metrics = {"threshold": float(seg_thr), "n_segments": int(yt.size), **_metrics_binary(yt, seg_pred)}

    _subj_uid_t, subj_p_t, subj_y_t = _collect_uid_probs(st.tolist(), pt.tolist(), yt.tolist())
    subj_pred = (subj_p_t >= subj_thr).astype(np.int32)
    subj_metrics = {"threshold": float(subj_thr), "n_subjects": int(subj_y_t.size), **_metrics_binary(subj_y_t, subj_pred)}

    report = {
        "run_dir": run_dir,
        "best_ckpt": best_path,
        "cache": {"npz": cache_npz, "json": cache_json, "band": band},
        "labels_cfg": labels_cfg,
        "model_used_in_eval": {
            "d_model": int(model_cfg["d_model"]),
            "dropout": float(model_cfg["dropout"]),
            "num_class": int(num_class),
            "n_blocks": int(model_cfg["resnet"]["n_blocks"]),
            "dilations": list(model_cfg["resnet"]["dilations"]),
            "enc_in": int(enc_in),
        },
        "segment": seg_metrics,
        "subject": subj_metrics,
        "threshold_search": {"segment_best_f1": seg_best, "subject_best_f1": subj_best},
        "sources": {
            "splits_subject_path": splits_path,
            "snapshot_path": snapshot_path if os.path.exists(snapshot_path) else None,
            "cfg_model_fallback": cfg_model_fallback,
            "n_blocks_from": "run_dir" if n_blocks_dir is not None else ("state_dict" if n_blocks_sd is not None else "model_cfg"),
        },
    }

    out_path = os.path.join(run_dir, "eval_report.json")
    save_json(out_path, report)
    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--cfg_model", default=None, help="Fallback if config_snapshot.json doesn't contain model config")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    main(args.run_dir, cfg_model_fallback=args.cfg_model, batch_size=args.batch_size, num_workers=args.num_workers)
