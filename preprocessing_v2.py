# preprocessing.py
# ------------------------------------------------------------
# Unified multiband preprocessing (19ch, bandpass 0.5–45Hz, resample 128Hz, SWT sym4 level=4,
# segment 1s overlap 0.5s, z-score per segment) across multiple datasets.
#
# OUTPUT:
# 1) SINGLE-DATASET CACHE:
#    {out_dir}/single/{dataset_name}/{band_name}/{band_name}.npz
#    - Each band file contains:
#        X_train_{band}, X_val_{band}, X_test_{band}
#        y_train, y_val, y_test
#        s_train, s_val, s_test
#    - A dataset-level meta JSON:
#        {out_dir}/single/{dataset_name}/{dataset_name}_meta.json
#
# 2) MULTI-DATASET CACHE:
#    Split 60/20/20 BY SUBJECT WITHIN EACH DATASET (stratified by AD/HC), then merged.
#    {out_dir}/multidataset/{band_name}/{band_name}.npz
#    - Each band file contains:
#        X_train_{band}, X_val_{band}, X_test_{band}
#        y_train, y_val, y_test
#        s_train, s_val, s_test
#    - Multi meta JSON:
#        {out_dir}/multidataset/multidataset_meta.json
#
# IMPORTANT CHANGE REQUESTED:
# - Now /.../DeepTokenEEG_Alzheimer_CLS/dataset contains ADFTD inside folder "ADFTD"
#   (e.g., datasets_root/ADFTD/ds004504 or datasets_root/ADFTD as a BIDS root).
# - This script auto-detects and scans ADFTD from datasets_root. No separate adftd_root required.
#
# RUN:
"""
 python preprocessing_v2.py \
   --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
   --out_dir /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/cache_unified \
   --fs_target 128 --seg_seconds 1.0 --overlap 0.5 --seed 42 --device cuda:0 --adfsu_fs_in 128
# ------------------------------------------------------------
"""
import os
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
import mne

from src.data.preprocessing import (
    read_filter_resample,
    swt_band_extract,
    segment_signal,
    zscore_per_segment,
)
from src.data.split import make_subject_splits  # kept (not required for new per-dataset split)
from src.utils.io import ensure_dir, save_npz, save_json

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import torch
except Exception:
    torch = None

try:
    import scipy.io as sio
except Exception:
    sio = None

mne.set_log_level("ERROR")

# --------------------------
# Constants
# --------------------------
CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"
]

CHANNELS_19_LEGACY = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

BANDS = ["fullband", "delta", "theta", "alpha", "beta", "gamma"]
BAND_PRESETS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 16.0),
    "beta":  (16.0, 32.0),
    "gamma": (32.0, 45.0),
    "fullband": (0.5, 45.0),
}
SWT_RECONSTRUCTION_MAP = {
    "delta": {"A4"},
    "theta": {"D4"},
    "alpha": {"D3"},
    "beta":  {"D2"},
    "gamma": {"D1"},
    "fullband": {"A0"},
}

# --------------------------
# Helpers
# --------------------------
def resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    d = (device or "cpu").strip().lower()
    if d.startswith("cuda"):
        return device if torch.cuda.is_available() else "cpu"
    return "cpu" if d == "cpu" else device


def _norm(x: str) -> str:
    x = ("" if x is None else str(x)).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _has_token(s: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", s) is not None


def label_from_text(text: str, other_map: Dict[str, int]) -> int:
    """
    Global mapping:
      AD -> 0
      HC -> 1
      Other diseases -> 2+
      Unknown -> -1
    Supports ADFTD short codes: A->AD, C->HC, F->Other
    """
    t = _norm(text)

    if t in {"a"}:
        return 0
    if t in {"c"}:
        return 1
    if t in {"f"}:
        if t not in other_map:
            other_map[t] = 2 if not other_map else max(other_map.values()) + 1
        return other_map[t]

    if ("alzheimer" in t) or ("alzheim" in t) or _has_token(t, "ad"):
        return 0

    if any(k in t for k in ["healthy", "control", "normal", "cn"]) or _has_token(t, "hc"):
        return 1

    if t in {"", "n/a", "na", "unknown", "unlabeled", "none", "nan"}:
        return -1

    if t not in other_map:
        other_map[t] = 2 if not other_map else max(other_map.values()) + 1
    return other_map[t]


def extract_digits_id(s: str) -> int:
    m = re.findall(r"\d+", str(s))
    return int(m[-1]) if m else 0


def find_first_ancestor_label(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    for p in reversed(parts):
        pl = p.lower()
        if "adsz" in pl:
            continue
        if pl == "ad" or "_ad" in pl or "alzheimer" in pl or re.search(r"\b1_ad\b", pl):
            return "AD"
        if "hc" in pl or "healthy" in pl or "control" in pl or "normal" in pl or "cn" in pl:
            return "HC"
        if "mci" in pl:
            return "MCI"
        if "ftd" in pl:
            return "FTD"
    return ""


def dataset_code(name: str) -> int:
    n = (name or "").lower()
    if "brainlat" in n:
        return 1
    if "aud" in n or "auditory" in n:
        return 2
    if "adfsu" in n:
        return 3
    if "apava" in n:
        return 4
    if "ds004504" in n or "adftd" in n:
        return 5
    return 9


def load_participants_map(root: str) -> Dict[str, str]:
    """
    participants.tsv -> {sub-xxx : label_text}
    """
    tsv = os.path.join(root, "participants.tsv")
    if not os.path.exists(tsv):
        return {}

    if pd is None:
        with open(tsv, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if not lines:
            return {}
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]

        pid_idx = 0
        for i, c in enumerate(header):
            if c.strip().lower() in {"participant_id", "subject", "sub_id", "id"}:
                pid_idx = i
                break

        label_idx = 1 if len(header) > 1 else 0
        for i, c in enumerate(header):
            cl = c.strip().lower()
            if any(k in cl for k in ["diagnos", "group", "status", "condition", "dx", "phenotype", "disease"]):
                label_idx = i
                break

        out = {}
        for r in rows:
            if len(r) <= max(pid_idx, label_idx):
                continue
            out[str(r[pid_idx])] = str(r[label_idx])
        return out

    df = pd.read_csv(tsv, sep="\t")
    pid_col = None
    for c in df.columns:
        if c.lower() in {"participant_id", "subject", "sub_id", "id"}:
            pid_col = c
            break
    if pid_col is None:
        pid_col = df.columns[0]

    label_col = None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["diagnos", "group", "status", "condition", "dx", "phenotype", "disease"]):
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else pid_col

    out = {}
    for _, row in df.iterrows():
        out[str(row[pid_col])] = str(row[label_col])
    return out


# --------------------------
# NEW: detect BIDS root (for ADFTD container)
# --------------------------
def is_bids_like_root(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    if os.path.exists(os.path.join(root, "participants.tsv")):
        return True
    # quick check for sub-*/eeg
    try:
        for entry in os.listdir(root):
            if entry.lower().startswith("sub-"):
                eeg_dir = os.path.join(root, entry, "eeg")
                if os.path.isdir(eeg_dir):
                    return True
    except Exception:
        return False
    return False


def discover_adftd_roots(datasets_root: str) -> List[Tuple[str, str]]:
    """
    If datasets_root contains folder ADFTD (container), it may hold:
      - A BIDS dataset directly (participants.tsv at ADFTD/)
      - One or more BIDS datasets inside (e.g., ADFTD/ds004504/)
    Returns list of (dataset_name, dataset_root_path).
    """
    out: List[Tuple[str, str]] = []
    adftd_dir = None
    for d in os.listdir(datasets_root):
        if d.lower() == "adftd":
            adftd_dir = os.path.join(datasets_root, d)
            break
    if adftd_dir is None or not os.path.isdir(adftd_dir):
        return out

    # case 1: ADFTD itself is a BIDS root
    if is_bids_like_root(adftd_dir):
        out.append(("ADFTD", adftd_dir))
        return out

    # case 2: ADFTD contains BIDS dataset(s), use child folder name (e.g., ds004504) to keep old naming
    for child in sorted(os.listdir(adftd_dir)):
        cpath = os.path.join(adftd_dir, child)
        if os.path.isdir(cpath) and is_bids_like_root(cpath):
            out.append((child, cpath))
    return out


# --------------------------
# NEW: per-dataset stratified splits (subject-level)
# --------------------------
def _stratified_subject_split(
    keys: List[str],
    labels: List[int],
    ratios: Tuple[float, float, float],
    seed: int
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    keys = np.asarray(keys, dtype=object)
    labels = np.asarray(labels, dtype=np.int32)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

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

        train_ids.extend(keys[tr].tolist())
        val_ids.extend(keys[va].tolist())
        test_ids.extend(keys[te].tolist())

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}


def _counts_ad_hc(keys: List[str], subject_data: Dict[str, dict]) -> Dict[str, int]:
    ad = 0
    hc = 0
    for k in keys:
        lb = int(subject_data[k]["label"])
        if lb == 0:
            ad += 1
        elif lb == 1:
            hc += 1
    return {"hc": hc, "ad": ad, "total": len(keys)}


# --------------------------
# .set reading (BrainLat-like) + enforce 19ch
# --------------------------
def find_closest_biosemi_channels(input_channels: List[str]) -> List[str]:
    biosemi = mne.channels.make_standard_montage("biosemi128")
    std1020 = mne.channels.make_standard_montage("standard_1020")

    bio_pos = biosemi.get_positions()["ch_pos"]
    std_pos = std1020.get_positions()["ch_pos"]

    closest = []
    for std_ch in input_channels:
        if std_ch not in std_pos:
            continue
        p = std_pos[std_ch]
        best = min(biosemi.ch_names, key=lambda ch: np.linalg.norm(bio_pos[ch] - p))
        closest.append(best)
    return closest


def ensure_raw_19ch(raw: mne.io.BaseRaw, desired: List[str] = CHANNELS_19) -> mne.io.BaseRaw:
    raw = raw.copy()

    ren = {k: v for k, v in LEGACY_TO_MODERN.items() if k in raw.ch_names}
    if ren:
        raw.rename_channels(ren)

    keep = [ch for ch in desired if ch in raw.ch_names]
    if not keep:
        raise RuntimeError("No overlap with desired 19 channels.")

    raw.pick(keep)

    mont = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(mont, on_missing="ignore")

    missing = [ch for ch in desired if ch not in raw.ch_names]
    if missing:
        info2 = mne.create_info(
            ch_names=missing,
            sfreq=float(raw.info["sfreq"]),
            ch_types=["eeg"] * len(missing),
        )
        raw2 = mne.io.RawArray(
            np.zeros((len(missing), raw.n_times), dtype=np.float32),
            info2,
            verbose="ERROR",
        )
        raw2.set_montage(mont, on_missing="ignore")
        raw.add_channels([raw2], force_update_info=True)
        raw.set_montage(mont, on_missing="ignore")

        raw.info["bads"] = missing
        try:
            raw.interpolate_bads(reset_bads=True, mode="accurate")
        except Exception:
            pass

    raw.pick(desired)
    raw.reorder_channels(desired)
    return raw


def read_set_eeglab_filter_resample_pick19(
    fpath: str,
    fs_std: int,
    fs_target: int,
) -> mne.io.Raw:
    raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)

    if abs(raw.info["sfreq"] - fs_std) > 1:
        raw.resample(fs_std, npad="auto", verbose="ERROR")

    raw.filter(0.5, 45.0, method="fir", phase="zero-double", verbose="ERROR")

    if abs(raw.info["sfreq"] - fs_target) > 1:
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    if len(raw.ch_names) > 30:
        to_pick = find_closest_biosemi_channels(CHANNELS_19)
        to_pick = [ch for ch in to_pick if ch in raw.ch_names]
    else:
        to_pick = [ch for ch in CHANNELS_19 if ch in raw.ch_names]
        if not to_pick:
            to_pick = [ch for ch in CHANNELS_19_LEGACY if ch in raw.ch_names]

    if not to_pick:
        raise RuntimeError("No usable channels found to map to 19ch.")

    raw.pick(to_pick)
    raw = ensure_raw_19ch(raw, CHANNELS_19)
    return raw


def read_filter_resample_robust(fpath: str, fs_std_default: int, fs_target: int) -> mne.io.Raw:
    try:
        raw = read_filter_resample(fpath, fs_std_default, fs_target, CHANNELS_19)
        raw = ensure_raw_19ch(raw, CHANNELS_19)
        return raw
    except Exception:
        raw = read_filter_resample(fpath, fs_std_default, fs_target, CHANNELS_19_LEGACY)
        try:
            raw.rename_channels({k: v for k, v in LEGACY_TO_MODERN.items() if k in raw.ch_names})
        except Exception:
            pass
        raw = ensure_raw_19ch(raw, CHANNELS_19)
        return raw


# --------------------------
# Scanners
# --------------------------
def scan_bids_set(root: str) -> List[Tuple[str, str, str]]:
    """
    BIDS-like: **/sub-*/eeg/*.set
    -> (fpath, disease_text, subject_dir_key_source)
    """
    pmap = load_participants_map(root)
    items: List[Tuple[str, str, str]] = []

    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() != "eeg":
            continue
        for fn in filenames:
            if not fn.lower().endswith(".set"):
                continue
            fpath = os.path.join(dirpath, fn)
            m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
            sub = m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(fpath)))

            disease = (
                pmap.get(sub, "")
                or pmap.get(sub.replace("sub-", ""), "")
                or find_first_ancestor_label(fpath)
            )

            subject_dir = os.path.join(root, sub)
            items.append((fpath, disease, subject_dir))

    items.sort()
    return items


def scan_adfsu(root: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for disease_folder in sorted(os.listdir(root)):
        disease_path = os.path.join(root, disease_folder)
        if not os.path.isdir(disease_path):
            continue

        for state in sorted(os.listdir(disease_path)):
            state_path = os.path.join(disease_path, state)
            if not os.path.isdir(state_path):
                continue

            for paciente in sorted(os.listdir(state_path)):
                paciente_path = os.path.join(state_path, paciente)
                if not os.path.isdir(paciente_path):
                    continue

                has_txt = False
                for dp, _, fns in os.walk(paciente_path):
                    for fn in fns:
                        if fn.lower().endswith(".txt"):
                            has_txt = True
                            break
                    if has_txt:
                        break

                if has_txt:
                    items.append((paciente_path, disease_folder, paciente_path))

    items.sort()
    return items


def scan_apava(root: str) -> List[Tuple[str, str, str]]:
    pmap = load_participants_map(root)
    mats = [os.path.join(root, fn) for fn in sorted(os.listdir(root)) if fn.lower().endswith(".mat")]

    items: List[Tuple[str, str, str]] = []
    for i, fpath in enumerate(mats, start=1):
        m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
        sub = m.group(1) if m else f"sub-{i:03d}"

        disease = pmap.get(sub, "") or pmap.get(sub.replace("sub-", ""), "")
        if not disease:
            AD_positive = {1, 3, 6, 8, 9, 11, 12, 13, 15, 17, 19, 21}
            if len(mats) == 23 and i in AD_positive:
                disease = "AD"
            elif len(mats) == 23:
                disease = "HC"
            else:
                disease = find_first_ancestor_label(fpath)

        items.append((fpath, disease, os.path.dirname(fpath)))
    return items


# --------------------------
# ADFSU (.txt per channel): load -> tc (T,19)
# --------------------------
def canonical_channel_name(name: str) -> str:
    name = (name or "").strip()
    for c in CHANNELS_19 + CHANNELS_19_LEGACY:
        if name.lower() == c.lower():
            return c
    return name


def _parse_channel_name_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    b = re.sub(r"[^A-Za-z0-9]+", "", base).lower()

    def norm_token(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", x).lower()

    candidates = CHANNELS_19 + CHANNELS_19_LEGACY
    candidates_sorted = sorted(candidates, key=lambda x: len(norm_token(x)), reverse=True)

    for ch in candidates_sorted:
        if norm_token(ch) in b:
            return canonical_channel_name(ch)

    for ch in candidates_sorted:
        if ch.lower() in base.lower():
            return canonical_channel_name(ch)

    return base.strip()


def _read_txt_1d(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float32, delimiter=None)
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.shape[1] >= 1:
            return arr[:, 0].astype(np.float32, copy=False)
        return arr.reshape(-1).astype(np.float32, copy=False)
    except Exception:
        vals = []
        num_re = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                found = num_re.findall(line)
                if not found:
                    continue
                vals.extend([float(x) for x in found])
        return np.asarray(vals, dtype=np.float32)


def load_adfsu_paciente_to_tc(paciente_dir: str, fs_in: float) -> Tuple[np.ndarray, float]:
    txts = []
    for dp, _, fns in os.walk(paciente_dir):
        for fn in fns:
            if fn.lower().endswith(".txt"):
                txts.append(os.path.join(dp, fn))
    if not txts:
        return np.empty((0, 19), dtype=np.float32), fs_in

    series: Dict[str, np.ndarray] = {}
    for fp in sorted(txts):
        ch = _parse_channel_name_from_filename(fp)
        ch = canonical_channel_name(ch)
        ch = LEGACY_TO_MODERN.get(ch, ch)

        if ch not in CHANNELS_19:
            continue

        x = _read_txt_1d(fp)
        if x.size == 0:
            continue

        if ch in series:
            if x.size > series[ch].size:
                series[ch] = x
        else:
            series[ch] = x

    present = [ch for ch in CHANNELS_19 if ch in series]
    if not present:
        return np.empty((0, 19), dtype=np.float32), fs_in

    T = int(min(series[ch].size for ch in present))
    if T <= 0:
        return np.empty((0, 19), dtype=np.float32), fs_in

    data19 = np.zeros((T, 19), dtype=np.float32)
    for j, ch in enumerate(CHANNELS_19):
        if ch in series:
            data19[:, j] = series[ch][:T]
        else:
            data19[:, j] = 0.0

    missing = [ch for ch in CHANNELS_19 if ch not in series]
    if missing:
        info = mne.create_info(ch_names=CHANNELS_19, sfreq=float(fs_in), ch_types=["eeg"] * 19)
        raw = mne.io.RawArray(data19.T, info, verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        raw.info["bads"] = missing
        try:
            raw.interpolate_bads(reset_bads=True, mode="accurate")
            data19 = raw.get_data().T.astype(np.float32, copy=False)
        except Exception:
            pass

    return data19, fs_in


# --------------------------
# APAVA: load + interpolate -> 19ch
# --------------------------
def interpolate_to_19_channels(
    eeg_tc: np.ndarray,
    input_channels: List[str],
    bad_channels: List[str],
    sfreq: float
) -> np.ndarray:
    input_channels = [LEGACY_TO_MODERN.get(ch, ch) for ch in input_channels]
    bad_channels = [LEGACY_TO_MODERN.get(ch, ch) for ch in bad_channels]

    eeg_tc = eeg_tc.astype(np.float32, copy=False)

    ch_all = input_channels + bad_channels
    data = np.zeros((len(ch_all), eeg_tc.shape[0]), dtype=np.float32)
    data[:len(input_channels), :] = eeg_tc.T

    info = mne.create_info(ch_names=ch_all, sfreq=float(sfreq), ch_types=["eeg"] * len(ch_all))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=True, mode="accurate")

    missing = [ch for ch in CHANNELS_19 if ch not in raw.ch_names]
    if missing:
        info2 = mne.create_info(ch_names=missing, sfreq=float(sfreq), ch_types=["eeg"] * len(missing))
        raw2 = mne.io.RawArray(np.zeros((len(missing), raw.n_times), dtype=np.float32), info2, verbose="ERROR")
        raw.add_channels([raw2], force_update_info=True)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        raw.info["bads"] = missing
        try:
            raw.interpolate_bads(reset_bads=True, mode="accurate")
        except Exception:
            pass

    raw.pick(CHANNELS_19)
    raw.reorder_channels(CHANNELS_19)
    return raw.get_data().T.astype(np.float32, copy=False)


def load_apava_mat_to_tc(mat_path: str) -> Tuple[np.ndarray, float]:
    if sio is None:
        raise RuntimeError("scipy.io not available -> cannot load .mat")

    mat = sio.loadmat(mat_path)
    if "data" not in mat:
        raise RuntimeError("APAVA .mat missing key 'data'")

    mat_np = mat["data"]
    epoch_list = mat_np[0, 0][2][0]
    epoch_num = len(epoch_list)

    input_channels = [
        "C3", "C4", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
        "O1", "O2", "P3", "P4", "T3", "T4", "T5", "T6"
    ]
    bad_channels = ["Fz", "Cz", "Pz"]
    sfreq = 256.0

    chunks = []
    for j in range(epoch_num):
        temp = np.transpose(epoch_list[j]).astype(np.float32, copy=False)  # (T,16)
        temp19 = interpolate_to_19_channels(temp, input_channels, bad_channels, sfreq=sfreq)
        chunks.append(temp19)

    data_tc = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 19), dtype=np.float32)
    return data_tc, sfreq


def preprocess_tc_to_band_segs(
    data_tc: np.ndarray,
    fs_in: float,
    fs_target: int,
    sample_len: int,
    hop_len: int
) -> Dict[str, np.ndarray]:
    if data_tc.size == 0:
        return {b: np.empty((0, sample_len, 19), dtype=np.float32) for b in BANDS}

    info = mne.create_info(ch_names=CHANNELS_19, sfreq=float(fs_in), ch_types=["eeg"] * 19)
    raw = mne.io.RawArray(data_tc.T.astype(np.float32, copy=False), info, verbose="ERROR")
    raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
    if float(fs_in) != float(fs_target):
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    raw = ensure_raw_19ch(raw, CHANNELS_19)
    data_tc_128 = raw.get_data().T.astype(np.float32, copy=False)

    out: Dict[str, np.ndarray] = {}
    for band_name in BANDS:
        data_band = swt_band_extract(
            data_tc_128, band_name, fs_target,
            band_presets=BAND_PRESETS,
            swt_recon_map=SWT_RECONSTRUCTION_MAP,
            swt_level=4, wavelet="sym4",
        )
        segs = segment_signal(data_band, sample_len, hop_len)
        out[band_name] = (
            zscore_per_segment(segs).astype(np.float32, copy=False)
            if segs.shape[0] else np.empty((0, sample_len, 19), dtype=np.float32)
        )
    return out


# --------------------------
# Main
# --------------------------
def main(args):
    ensure_dir(args.out_dir)
    device_used = resolve_device(args.device)
    print(f"[DEVICE] requested={args.device} | used={device_used}")

    sample_len = int(args.fs_target * args.seg_seconds)
    hop_len = int(sample_len * (1 - args.overlap))

    all_items: List[Tuple[str, str, str, str]] = []  # (dataset_name, fpath_or_dir, disease_text, subject_dir)

    scanned_roots: set = set()

    # 1) First, auto-discover ADFTD from datasets_root/ADFTD (container)
    adftd_discovered = discover_adftd_roots(args.datasets_root)
    for dn_name, root in adftd_discovered:
        rp = os.path.realpath(root)
        if rp in scanned_roots:
            continue
        scanned_roots.add(rp)

        items = scan_bids_set(root)
        print(f"[SCAN] {dn_name}: found {len(items)} candidate files (from ADFTD container)")
        for fpath, disease_text, subject_dir in items:
            all_items.append((dn_name, fpath, disease_text, subject_dir))

    # 2) Scan other datasets directly under datasets_root (skip ADSZ and skip ADFTD container itself)
    for dname in sorted(os.listdir(args.datasets_root)):
        droot = os.path.join(args.datasets_root, dname)
        if not os.path.isdir(droot):
            continue
        if "adsz" in dname.lower():
            continue
        if dname.lower() == "adftd":
            # already handled via discover_adftd_roots
            continue

        rp = os.path.realpath(droot)
        if rp in scanned_roots:
            continue
        scanned_roots.add(rp)

        if "apava" in dname.lower():
            items = scan_apava(droot)
        elif "adfsu" in dname.lower():
            items = scan_adfsu(droot)
        else:
            items = scan_bids_set(droot)

        print(f"[SCAN] {dname}: found {len(items)} candidate files")
        for fpath, disease_text, subject_dir in items:
            all_items.append((dname, fpath, disease_text, subject_dir))

    # --------------------------
    # Preprocess all items
    # --------------------------
    subject_data: Dict[str, dict] = {}
    other_map: Dict[str, int] = {}

    files_attempted = 0
    files_success = 0
    per_dataset: Dict[str, dict] = {}

    KEY_MULT = 1_000_000_000

    def make_key_uid(dn: str, subject_dir: str) -> Tuple[str, str]:
        pid = extract_digits_id(subject_dir)
        uid = f"{dn}:{pid}"
        key = str(dataset_code(dn) * KEY_MULT + pid)  # digits-only
        return key, uid

    for dn, fpath, disease_text, subject_dir in tqdm(all_items, desc="Preprocessing", total=len(all_items)):
        files_attempted += 1
        per_dataset.setdefault(
            dn,
            {"files_attempted": 0, "files_success": 0, "subjects": set(), "errors_fdt_missing": 0, "errors_adfsu_txt": 0}
        )
        per_dataset[dn]["files_attempted"] += 1

        try:
            key, uid = make_key_uid(dn, subject_dir)
            label = int(label_from_text(disease_text, other_map))

            dn_l = dn.lower()

            if "apava" in dn_l:
                data_tc, fs_in = load_apava_mat_to_tc(fpath)
                bands_segs = preprocess_tc_to_band_segs(
                    data_tc=data_tc,
                    fs_in=fs_in,
                    fs_target=args.fs_target,
                    sample_len=sample_len,
                    hop_len=hop_len,
                )

            elif "adfsu" in dn_l:
                data_tc, fs_in = load_adfsu_paciente_to_tc(str(fpath), fs_in=float(args.adfsu_fs_in))
                bands_segs = preprocess_tc_to_band_segs(
                    data_tc=data_tc,
                    fs_in=fs_in,
                    fs_target=args.fs_target,
                    sample_len=sample_len,
                    hop_len=hop_len,
                )

            else:
                if str(fpath).lower().endswith(".set"):
                    raw = read_set_eeglab_filter_resample_pick19(
                        fpath=str(fpath),
                        fs_std=int(args.fs_std_default),
                        fs_target=int(args.fs_target),
                    )
                else:
                    raw = read_filter_resample_robust(str(fpath), int(args.fs_std_default), int(args.fs_target))

                raw = ensure_raw_19ch(raw, CHANNELS_19)
                data_tc = raw.get_data().T.astype(np.float32, copy=False)  # (T,19) guaranteed

                bands_segs = {}
                for band_name in BANDS:
                    data_band = swt_band_extract(
                        data_tc, band_name, args.fs_target,
                        band_presets=BAND_PRESETS,
                        swt_recon_map=SWT_RECONSTRUCTION_MAP,
                        swt_level=4, wavelet="sym4",
                    )
                    segs = segment_signal(data_band, sample_len, hop_len)
                    bands_segs[band_name] = (
                        zscore_per_segment(segs).astype(np.float32, copy=False)
                        if segs.shape[0] else np.empty((0, sample_len, 19), dtype=np.float32)
                    )

            if key in subject_data:
                for band_name in BANDS:
                    subject_data[key]["bands"][band_name] = np.concatenate(
                        [subject_data[key]["bands"][band_name], bands_segs[band_name]],
                        axis=0
                    )
            else:
                subject_data[key] = {
                    "bands": bands_segs,
                    "label": label,
                    "uid": uid,
                    "dataset": dn,
                }

            files_success += 1
            per_dataset[dn]["files_success"] += 1
            per_dataset[dn]["subjects"].add(key)

        except Exception as e:
            msg = str(e).lower()
            if str(fpath).lower().endswith(".set") and ("fdt" in msg or "not found" in msg):
                per_dataset[dn]["errors_fdt_missing"] += 1
            if "adfsu" in dn.lower():
                per_dataset[dn]["errors_adfsu_txt"] += 1
            print(f"[ERROR] {dn} | {os.path.basename(str(fpath))}: {e}")

    for dn in per_dataset:
        per_dataset[dn]["subjects"] = len(per_dataset[dn]["subjects"])

    print(f"[DONE] subjects: {len(subject_data)} files_attempted: {files_attempted} files_success: {files_success}")
    if len(subject_data) == 0:
        raise RuntimeError("No subjects processed. Check dataset paths / scanners / file types.")

    # --------------------------
    # Group subjects by dataset
    # --------------------------
    subjects_by_dataset: Dict[str, List[str]] = {}
    for k, v in subject_data.items():
        subjects_by_dataset.setdefault(v["dataset"], []).append(k)
    for dn in subjects_by_dataset:
        subjects_by_dataset[dn].sort()

    # --------------------------
    # Split per dataset (subject-level stratified)
    # --------------------------
    splits_by_dataset: Dict[str, dict] = {}
    for dn, keys in subjects_by_dataset.items():
        labels = [int(subject_data[k]["label"]) for k in keys]
        sp = _stratified_subject_split(keys, labels, tuple(args.ratios), int(args.seed))
        sp["counts"] = {
            "train": _counts_ad_hc(sp["train_ids"], subject_data),
            "val": _counts_ad_hc(sp["val_ids"], subject_data),
            "test": _counts_ad_hc(sp["test_ids"], subject_data),
        }
        splits_by_dataset[dn] = sp

    # --------------------------
    # Flatten helper (kept structure)
    # --------------------------
    def flatten(ids: List[str], band_name: str):
        Xs, ys, ss = [], [], []
        for key in ids:
            feats = subject_data[key]["bands"][band_name]
            if feats.shape[0] == 0:
                continue
            Xs.append(feats)
            ys.extend([subject_data[key]["label"]] * feats.shape[0])
            ss.extend([subject_data[key]["uid"]] * feats.shape[0])

        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, sample_len, 19), dtype=np.float32)
        y = np.asarray(ys, dtype=np.int32)
        s = np.asarray(ss, dtype=str)
        return X, y, s

    # ============================================================
    # SAVE: SINGLE (per dataset, per band)
    # ============================================================
    single_root = os.path.join(args.out_dir, "single")
    ensure_dir(single_root)

    for dn in sorted(splits_by_dataset.keys()):
        dn_root = os.path.join(single_root, dn)
        ensure_dir(dn_root)

        train_ids = splits_by_dataset[dn]["train_ids"]
        val_ids = splits_by_dataset[dn]["val_ids"]
        test_ids = splits_by_dataset[dn]["test_ids"]

        band_shapes = {}
        for band_name in BANDS:
            out_band_dir = os.path.join(dn_root, band_name)
            ensure_dir(out_band_dir)
            out_npz = os.path.join(out_band_dir, f"{band_name}.npz")

            Xtr, ytr, str_ = flatten(train_ids, band_name)
            Xva, yva, sva  = flatten(val_ids, band_name)
            Xte, yte, ste  = flatten(test_ids, band_name)

            payload = {
                "y_train": ytr, "s_train": str_,
                "y_val": yva, "s_val": sva,
                "y_test": yte, "s_test": ste,
                f"X_train_{band_name}": Xtr,
                f"X_val_{band_name}": Xva,
                f"X_test_{band_name}": Xte,
            }
            save_npz(out_npz, **payload)
            band_shapes[band_name] = {k: list(v.shape) for k, v in payload.items() if isinstance(v, np.ndarray)}

        subject_index = []
        for k in subjects_by_dataset[dn]:
            subject_index.append({
                "key": k,
                "uid": subject_data[k]["uid"],
                "label": int(subject_data[k]["label"]),
                "dataset": dn
            })

        out_meta = os.path.join(dn_root, f"{dn}_meta.json")
        save_json(out_meta, {
            "device": {"requested": args.device, "used": device_used},
            "dataset": dn,
            "paths": {
                "datasets_root": args.datasets_root,
                "out_dir": args.out_dir,
                "single_dataset_dir": dn_root,
            },
            "channels_19": CHANNELS_19,
            "signal": {
                "bandpass_hz": [0.5, 45.0],
                "fs_target": args.fs_target,
                "seg_seconds": args.seg_seconds,
                "overlap": args.overlap,
                "sample_len": sample_len,
                "hop_len": hop_len,
                "adfsu_fs_in": float(args.adfsu_fs_in),
            },
            "swt": {
                "wavelet": "sym4",
                "level": 4,
                "bands": BANDS,
                "reconstruction_map": {k: sorted(list(v)) for k, v in SWT_RECONSTRUCTION_MAP.items()},
            },
            "label_policy": {
                "AD": 0,
                "HC": 1,
                "others": "2+ assigned by first appearance (normalized text)",
                "unknown": -1,
                "other_map": other_map,
            },
            "counts": {"subjects": len(subjects_by_dataset[dn])},
            "subject_index": subject_index,
            "splits_subject_level": splits_by_dataset[dn],
            "band_files": {b: os.path.join(dn_root, b, f"{b}.npz") for b in BANDS},
            "shapes": band_shapes,
        })

        print(f"[SINGLE] saved dataset={dn} -> {dn_root}")

    # ============================================================
    # SAVE: MULTI (split per dataset, then merged)
    # ============================================================
    multi_root = os.path.join(args.out_dir, "multidataset")
    ensure_dir(multi_root)

    merged_train_ids: List[str] = []
    merged_val_ids: List[str] = []
    merged_test_ids: List[str] = []
    for dn in sorted(splits_by_dataset.keys()):
        merged_train_ids.extend(splits_by_dataset[dn]["train_ids"])
        merged_val_ids.extend(splits_by_dataset[dn]["val_ids"])
        merged_test_ids.extend(splits_by_dataset[dn]["test_ids"])

    merged_splits = {
        "train_ids": merged_train_ids,
        "val_ids": merged_val_ids,
        "test_ids": merged_test_ids,
        "counts": {
            "train": _counts_ad_hc(merged_train_ids, subject_data),
            "val": _counts_ad_hc(merged_val_ids, subject_data),
            "test": _counts_ad_hc(merged_test_ids, subject_data),
        }
    }

    multi_band_shapes = {}
    for band_name in BANDS:
        out_band_dir = os.path.join(multi_root, band_name)
        ensure_dir(out_band_dir)
        out_npz = os.path.join(out_band_dir, f"{band_name}.npz")

        Xtr, ytr, str_ = flatten(merged_train_ids, band_name)
        Xva, yva, sva  = flatten(merged_val_ids, band_name)
        Xte, yte, ste  = flatten(merged_test_ids, band_name)

        payload = {
            "y_train": ytr, "s_train": str_,
            "y_val": yva, "s_val": sva,
            "y_test": yte, "s_test": ste,
            f"X_train_{band_name}": Xtr,
            f"X_val_{band_name}": Xva,
            f"X_test_{band_name}": Xte,
        }
        save_npz(out_npz, **payload)
        multi_band_shapes[band_name] = {k: list(v.shape) for k, v in payload.items() if isinstance(v, np.ndarray)}

    subject_index_all = []
    for k, v in subject_data.items():
        subject_index_all.append({
            "key": k,
            "uid": v["uid"],
            "label": int(v["label"]),
            "dataset": v["dataset"],
        })
    subject_index_all.sort(key=lambda x: (x["dataset"], x["key"]))

    out_multi_meta = os.path.join(multi_root, "multidataset_meta.json")
    save_json(out_multi_meta, {
        "device": {"requested": args.device, "used": device_used},
        "paths": {
            "datasets_root": args.datasets_root,
            "out_dir": args.out_dir,
            "multidataset_dir": multi_root,
        },
        "adftd_detected_from_datasets_root": adftd_discovered,  # NEW: explicit trace
        "channels_19": CHANNELS_19,
        "signal": {
            "bandpass_hz": [0.5, 45.0],
            "fs_target": args.fs_target,
            "seg_seconds": args.seg_seconds,
            "overlap": args.overlap,
            "sample_len": sample_len,
            "hop_len": hop_len,
            "adfsu_fs_in": float(args.adfsu_fs_in),
        },
        "swt": {
            "wavelet": "sym4",
            "level": 4,
            "bands": BANDS,
            "reconstruction_map": {k: sorted(list(v)) for k, v in SWT_RECONSTRUCTION_MAP.items()},
        },
        "label_policy": {
            "AD": 0,
            "HC": 1,
            "others": "2+ assigned by first appearance (normalized text)",
            "unknown": -1,
            "other_map": other_map,
        },
        "counts": {
            "files_attempted": files_attempted,
            "files_success": files_success,
            "subjects": len(subject_data),
            "per_dataset": per_dataset,
        },
        "subject_index": subject_index_all,
        "splits_subject_level_by_dataset": splits_by_dataset,
        "splits_subject_level_merged": merged_splits,
        "band_files": {b: os.path.join(multi_root, b, f"{b}.npz") for b in BANDS},
        "shapes": multi_band_shapes,
    })

    print(f"[MULTI] saved -> {multi_root}")
    print(f"[MULTI SPLIT] train={len(merged_train_ids)} val={len(merged_val_ids)} test={len(merged_test_ids)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--fs_std_default", type=int, default=500)
    p.add_argument("--fs_target", type=int, default=128)
    p.add_argument("--seg_seconds", type=float, default=1.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    p.add_argument("--adfsu_fs_in", type=float, default=128.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)




