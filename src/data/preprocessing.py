# src/data/preprocessing.py
# ------------------------------------------------------------
# Core preprocessing utilities used by scripts/prepare_data.py
#
# Provides:
# - constants: CHANNELS_19, BANDS, SWT_RECONSTRUCTION_MAP, KEY_MULT
# - label mapping: HC=0, AD=1, others=2+, unknown=-1
# - dataset discovery (incl. ADFTD container)
# - scanners per dataset type (BIDS/.set, ADFSU txt, APAVA .mat)
# - channel enforcing (19ch), read/filter/resample
# - SWT band reconstruction, segmentation, per-segment z-score
# - array-based preprocess (APAVA/ADFSU) to band segments
# - helpers to flatten subjects into NPZ payload and to write sidecar JSON
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import zlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import mne
import pywt

from src.utils.io import save_json

mne.set_log_level("ERROR")

# ============================================================
# Constants
# ============================================================

CHANNELS_19: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"
]

CHANNELS_19_LEGACY: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

LEGACY_TO_MODERN: Dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

BANDS: List[str] = ["fullband", "delta", "theta", "alpha", "beta", "gamma"]

SWT_RECONSTRUCTION_MAP: Dict[str, set] = {
    "delta": {"A4"},
    "theta": {"D4"},
    "alpha": {"D3"},
    "beta":  {"D2"},
    "gamma": {"D1"},
    "fullband": {"A0"},
}

KEY_MULT: int = 1_000_000_000  # key_int = dataset_code*1e9 + pid (pid < 1e9)


# Optional deps
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


# ============================================================
# Device helper
# ============================================================

def resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    d = (device or "cpu").strip().lower()
    if d.startswith("cuda"):
        return device if torch.cuda.is_available() else "cpu"
    return "cpu" if d == "cpu" else device


# ============================================================
# Label policy  (HC=0, AD=1)
# ============================================================

def _norm(x: str) -> str:
    x = ("" if x is None else str(x)).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def _has_token(s: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", s) is not None

def label_from_text(text: str, other_map: Dict[str, int]) -> int:
    """
    Global mapping:
      HC -> 0
      AD -> 1
      Other diseases -> 2+
      Unknown/unlabeled -> -1

    Supports ADFTD short codes:
      C -> HC, A -> AD, F -> Other
    """
    t = _norm(text)

    # ADFTD short codes
    if t in {"c"}:
        return 0
    if t in {"a"}:
        return 1
    if t in {"f"}:
        if t not in other_map:
            other_map[t] = 2 if not other_map else max(other_map.values()) + 1
        return other_map[t]

    # AD
    if ("alzheimer" in t) or ("alzheim" in t) or _has_token(t, "ad"):
        return 1

    # HC
    if any(k in t for k in ["healthy", "control", "normal", "cn"]) or _has_token(t, "hc"):
        return 0

    # unknown
    if t in {"", "n/a", "na", "unknown", "unlabeled", "none", "nan"}:
        return -1

    # other diseases
    if t not in other_map:
        other_map[t] = 2 if not other_map else max(other_map.values()) + 1
    return other_map[t]


# ============================================================
# IDs + dataset code
# ============================================================

def extract_digits_id(s: str) -> int:
    m = re.findall(r"\d+", str(s))
    return int(m[-1]) if m else 0

def stable_pid_from_token(token: str) -> int:
    """
    Ensure pid digits-only and < 1e9.
    If token has digits -> use last digits.
    Else -> crc32(token) mapped into [10_000_000 .. 999_999_999]
    """
    d = extract_digits_id(token)
    if d > 0:
        return int(d)
    h = zlib.crc32(str(token).encode("utf-8")) & 0xFFFFFFFF
    return 10_000_000 + (h % 990_000_000)

def dataset_code(name: str) -> int:
    """
    Known datasets fixed code; unknown -> stable 100..999.
    """
    n = (name or "").lower()
    if "brainlat" in n:
        return 1
    if "aud" in n or "auditory" in n:
        return 2
    if "adfsu" in n:
        return 3
    if "apava" in n:
        return 4
    if "adftd" in n or "ds004504" in n:
        return 5

    h = zlib.crc32((name or "").encode("utf-8")) & 0xFFFFFFFF
    return 100 + (h % 900)

def make_key_uid(dataset_name: str, subject_token: str) -> Tuple[str, str, int, int]:
    """
    Returns:
      key_str (digits-only), uid "dataset:pid", key_int, pid
    """
    pid = stable_pid_from_token(subject_token)
    uid = f"{dataset_name}:{pid}"
    code = dataset_code(dataset_name)
    key_int = int(code * KEY_MULT + pid)
    key_str = str(key_int)
    return key_str, uid, key_int, pid


# ============================================================
# Participants.tsv (BIDS)
# ============================================================

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


# ============================================================
# ADFTD discovery (container)
# ============================================================

def is_bids_like_root(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    if os.path.exists(os.path.join(root, "participants.tsv")):
        return True
    try:
        for entry in os.listdir(root):
            if entry.lower().startswith("sub-") and os.path.isdir(os.path.join(root, entry, "eeg")):
                return True
    except Exception:
        return False
    return False

def discover_adftd_roots(datasets_root: str) -> List[Tuple[str, str]]:
    """
    datasets_root/ADFTD may be:
      - BIDS root directly
      - or contain sub-datasets (e.g. ADFTD/ds004504)
    Returns list of (dataset_name, root_path).
    """
    out: List[Tuple[str, str]] = []
    adftd_dir = None
    for d in os.listdir(datasets_root):
        if d.lower() == "adftd":
            adftd_dir = os.path.join(datasets_root, d)
            break
    if adftd_dir is None or not os.path.isdir(adftd_dir):
        return out

    if is_bids_like_root(adftd_dir):
        return [("ADFTD", adftd_dir)]

    children: List[Tuple[str, str]] = []
    for child in sorted(os.listdir(adftd_dir)):
        cpath = os.path.join(adftd_dir, child)
        if os.path.isdir(cpath) and is_bids_like_root(cpath):
            children.append((child, cpath))

    # alias: if only one child, allow selecting "ADFTD"
    if len(children) == 1 and "ADFTD" not in [n for n, _ in children]:
        return children + [("ADFTD", children[0][1])]
    return children

def discover_all_dataset_roots(datasets_root: str) -> Dict[str, str]:
    """
    Build name -> root_path map, including ADFTD container discovery.
    """
    roots: Dict[str, str] = {}
    for dn, rp in discover_adftd_roots(datasets_root):
        roots[dn] = rp

    for dname in sorted(os.listdir(datasets_root)):
        droot = os.path.join(datasets_root, dname)
        if not os.path.isdir(droot):
            continue
        if "adsz" in dname.lower():
            continue
        if dname.lower() == "adftd":
            continue
        roots[dname] = droot

    return roots


# ============================================================
# Channel mapping + enforcing 19ch
# ============================================================

def find_closest_biosemi_channels(input_channels: List[str]) -> List[str]:
    """
    Map standard 10-20 channel names to closest Biosemi128 channel names (heuristic).
    """
    biosemi = mne.channels.make_standard_montage("biosemi128")
    std = mne.channels.make_standard_montage("standard_1020")
    closest = []
    bpos = biosemi.get_positions()["ch_pos"]
    spos = std.get_positions()["ch_pos"]

    for ch in input_channels:
        if ch in spos:
            std_pos = spos[ch]
            dists = {bch: np.linalg.norm(pos - std_pos) for bch, pos in bpos.items()}
            closest.append(min(dists, key=dists.get))
    return closest

def ensure_raw_19ch(raw: mne.io.BaseRaw, desired: List[str] = CHANNELS_19) -> mne.io.BaseRaw:
    """
    Ensure exact 19 channels exist in raw:
    - rename legacy T3/T4/T5/T6 -> T7/T8/P7/P8
    - keep available desired
    - add missing channels as zeros, mark bads, interpolate (best-effort)
    - reorder to desired
    """
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


# ============================================================
# Generic file reader (BIDS/.set/.fif)
# ============================================================

def read_filter_resample(file_path: str, fs_std: int, fs_target: int, pick_channels_19: List[str]) -> mne.io.Raw:
    """
    Read (.fif or EEGLAB .set), pick EEG, pick ~19 channels, resample/filter:
      - pick channels first
      - resample to fs_std (optional)
      - bandpass 0.5-45
      - resample to fs_target
    """
    raw = (
        mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
        if file_path.lower().endswith(".fif")
        else mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    )
    raw.pick("eeg")

    # pick channels first
    if len(raw.ch_names) > 30:
        channels_to_pick = find_closest_biosemi_channels(pick_channels_19)
        channels_to_pick = [ch for ch in channels_to_pick if ch in raw.ch_names]
    else:
        channels_to_pick = [ch for ch in pick_channels_19 if ch in raw.ch_names]

    if len(channels_to_pick) == 0:
        raise ValueError("No matching EEG channels found for the requested 19-channel set.")

    raw.pick(channels_to_pick)

    if abs(raw.info["sfreq"] - fs_std) > 1:
        raw.resample(fs_std, npad="auto", verbose="ERROR")

    raw.filter(0.5, 45.0, method="fir", phase="zero-double", verbose="ERROR")

    if abs(raw.info["sfreq"] - fs_target) > 1:
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    return raw


# ============================================================
# SWT band reconstruction
# ============================================================

def swt_band_extract(
    data_tc: np.ndarray,
    band_name: str,
    fs_target: int,
    band_presets=None,
    swt_recon_map: Optional[Dict[str, set]] = None,
    swt_level: int = 4,
    wavelet: str = "sym4"
) -> np.ndarray:
    """
    data_tc: [T, C] float32 (already bandpass 0.5-45 and resampled to fs_target)
    returns: [T, C]

    SWT level=4 @ fs=128:
      A4 ~ low freq (delta-ish), D4 theta, D3 alpha, D2 beta, D1 gamma
      fullband returns original data_tc
    """
    band_name = (band_name or "").lower().strip()

    default_map = {
        "delta": {"A4"},
        "theta": {"D4"},
        "alpha": {"D3"},
        "beta":  {"D2"},
        "gamma": {"D1"},
        "fullband": {"A0"},
    }
    if swt_recon_map is None:
        swt_recon_map = default_map
    else:
        for k, v in default_map.items():
            swt_recon_map.setdefault(k, v)

    if band_name in ("fullband", "a0"):
        return data_tc.astype(np.float32, copy=False)

    if band_name not in swt_recon_map:
        raise ValueError(f"Unknown band_name={band_name}. Expected one of {list(swt_recon_map.keys())}")

    keep = {x.upper() for x in swt_recon_map[band_name]}
    T, C = data_tc.shape
    out = np.empty_like(data_tc, dtype=np.float32)

    m = 2 ** int(swt_level)

    for c in range(C):
        x = data_tc[:, c].astype(np.float32, copy=False)
        orig_len = x.shape[0]

        padded_len = orig_len if (orig_len % m == 0) else (orig_len + (m - orig_len % m))
        pad_amount = padded_len - orig_len
        left_pad = pad_amount // 2
        x_pad = np.pad(x, (left_pad, pad_amount - left_pad), mode="symmetric")

        coeffs = pywt.swt(x_pad, wavelet, level=int(swt_level), trim_approx=False)

        def recon_A_L():
            kept = [(cA, np.zeros_like(cD)) for (cA, cD) in coeffs]
            return pywt.iswt(kept, wavelet)

        def recon_D_k(k_level: int):
            kept = []
            for j, (cA, cD) in enumerate(coeffs, start=1):
                cA0 = np.zeros_like(cA)
                cDk = cD if (j == k_level) else np.zeros_like(cD)
                kept.append((cA0, cDk))
            return pywt.iswt(kept, wavelet)

        y = np.zeros_like(x_pad, dtype=np.float32)
        for nm in keep:
            if nm == "A0":
                y += x_pad.astype(np.float32, copy=False)
            elif nm == f"A{int(swt_level)}":
                y += recon_A_L().astype(np.float32, copy=False)
            elif nm.startswith("D"):
                k = int(re.findall(r"\d+", nm)[0])
                y += recon_D_k(k).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Invalid SWT component name: {nm}")

        y = y[left_pad:left_pad + orig_len]
        out[:, c] = y.astype(np.float32, copy=False)

    return out


# ============================================================
# Segment + normalize
# ============================================================

def segment_signal(data_tc: np.ndarray, segment_length: int, hop_length: int) -> np.ndarray:
    """
    data_tc: (T, C) -> segs: (N, segment_length, C)
    """
    T, C = data_tc.shape
    if T < segment_length:
        return np.empty((0, segment_length, C), dtype=np.float32)
    starts = np.arange(0, T - segment_length + 1, hop_length, dtype=int)
    segs = np.array([data_tc[s:s + segment_length] for s in starts], dtype=np.float32)
    return segs

def zscore_per_segment(segs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    segs: (N, L, C) -> zscore per segment along axis=1
    """
    mu = segs.mean(axis=1, keepdims=True)
    sd = segs.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return ((segs - mu) / sd).astype(np.float32)


# ============================================================
# Dataset scanners (organized by dataset type)
# ============================================================

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

# ---- BIDS / .set / .fif ----

def scan_set_recursive(root: str) -> List[Tuple[str, str, str, str]]:
    """
    Recursive scan all .set:
      -> (fpath, disease_text, subject_token, subject_dir)
    """
    pmap = load_participants_map(root)
    items: List[Tuple[str, str, str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".set"):
                continue
            fpath = os.path.join(dirpath, fn)

            m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
            sub = m.group(1) if m else os.path.basename(os.path.dirname(fpath))
            disease = pmap.get(sub, "") or pmap.get(sub.replace("sub-", ""), "") or find_first_ancestor_label(fpath)

            subject_dir = os.path.join(root, sub) if os.path.isdir(os.path.join(root, sub)) else os.path.dirname(fpath)
            items.append((fpath, disease, sub, subject_dir))

    items.sort()
    return items

def scan_bids_set(root: str) -> List[Tuple[str, str, str, str]]:
    """
    If BIDS-like (sub-*/eeg/*.set), scan there; else fallback recursive.
    """
    has_eeg = False
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath).lower() == "eeg":
            has_eeg = True
            break
    if not has_eeg:
        return scan_set_recursive(root)

    pmap = load_participants_map(root)
    items: List[Tuple[str, str, str, str]] = []

    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() != "eeg":
            continue
        for fn in filenames:
            if not fn.lower().endswith(".set"):
                continue
            fpath = os.path.join(dirpath, fn)

            m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
            sub = m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            disease = pmap.get(sub, "") or pmap.get(sub.replace("sub-", ""), "") or find_first_ancestor_label(fpath)
            subject_dir = os.path.join(root, sub)
            items.append((fpath, disease, sub, subject_dir))

    items.sort()
    return items

# ---- ADFSU (txt per channel) ----

def _parse_channel_name_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    b = re.sub(r"[^A-Za-z0-9]+", "", base).lower()

    def norm_token(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", x).lower()

    candidates = CHANNELS_19 + CHANNELS_19_LEGACY
    candidates_sorted = sorted(candidates, key=lambda x: len(norm_token(x)), reverse=True)

    for ch in candidates_sorted:
        if norm_token(ch) in b:
            return ch
    for ch in candidates_sorted:
        if ch.lower() in base.lower():
            return ch
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

def scan_adfsu(root: str) -> List[Tuple[str, str, str, str]]:
    """
    Return (paciente_dir, disease_folder, paciente_token, paciente_dir)
    """
    items: List[Tuple[str, str, str, str]] = []
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
                has_txt = any(fn.lower().endswith(".txt") for _, _, fns in os.walk(paciente_path) for fn in fns)
                if has_txt:
                    items.append((paciente_path, disease_folder, paciente, paciente_path))
    items.sort()
    return items

def load_adfsu_paciente_to_tc(paciente_dir: str, fs_in: float) -> Tuple[np.ndarray, float]:
    """
    paciente_dir -> (T,19), fs_in
    Missing channels filled with zeros (no interpolation here).
    """
    txts = [os.path.join(dp, fn) for dp, _, fns in os.walk(paciente_dir) for fn in fns if fn.lower().endswith(".txt")]
    if not txts:
        return np.empty((0, 19), dtype=np.float32), float(fs_in)

    series: Dict[str, np.ndarray] = {}
    for fp in sorted(txts):
        ch = _parse_channel_name_from_filename(fp)
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
        return np.empty((0, 19), dtype=np.float32), float(fs_in)

    T = int(min(series[ch].size for ch in present))
    if T <= 0:
        return np.empty((0, 19), dtype=np.float32), float(fs_in)

    data19 = np.zeros((T, 19), dtype=np.float32)
    for j, ch in enumerate(CHANNELS_19):
        data19[:, j] = series[ch][:T] if ch in series else 0.0

    return data19, float(fs_in)

# ---- APAVA (.mat) ----

def scan_apava(root: str) -> List[Tuple[str, str, str, str]]:
    """
    Return (mat_path, disease_text, token, subject_dir)
    """
    pmap = load_participants_map(root)
    mats = [os.path.join(root, fn) for fn in sorted(os.listdir(root)) if fn.lower().endswith(".mat")]
    items: List[Tuple[str, str, str, str]] = []
    for i, fpath in enumerate(mats, start=1):
        m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
        sub = m.group(1) if m else f"sub-{i:03d}"
        disease = pmap.get(sub, "") or pmap.get(sub.replace("sub-", ""), "")
        if not disease:
            disease = find_first_ancestor_label(fpath)
        items.append((fpath, disease, sub, os.path.dirname(fpath)))
    return items

def load_apava_mat_to_tc(mat_path: str) -> Tuple[np.ndarray, float]:
    """
    APAVA mat -> concatenate epochs -> (T,19), sfreq=256
    Uses MNE interpolation from 16ch + 3 bad channels.
    """
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

    def interpolate_to_19_channels(eeg_tc: np.ndarray, input_ch: List[str], bad_ch: List[str], sf: float) -> np.ndarray:
        input_ch = [LEGACY_TO_MODERN.get(ch, ch) for ch in input_ch]
        bad_ch = [LEGACY_TO_MODERN.get(ch, ch) for ch in bad_ch]
        eeg_tc = eeg_tc.astype(np.float32, copy=False)

        ch_all = input_ch + bad_ch
        data = np.zeros((len(ch_all), eeg_tc.shape[0]), dtype=np.float32)
        data[:len(input_ch), :] = eeg_tc.T

        info = mne.create_info(ch_names=ch_all, sfreq=float(sf), ch_types=["eeg"] * len(ch_all))
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

        raw.info["bads"] = bad_ch
        raw.interpolate_bads(reset_bads=True, mode="accurate")

        missing = [ch for ch in CHANNELS_19 if ch not in raw.ch_names]
        if missing:
            info2 = mne.create_info(ch_names=missing, sfreq=float(sf), ch_types=["eeg"] * len(missing))
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

    chunks: List[np.ndarray] = []
    for j in range(epoch_num):
        temp = np.transpose(epoch_list[j]).astype(np.float32, copy=False)  # (T,16)
        chunks.append(interpolate_to_19_channels(temp, input_channels, bad_channels, sfreq))

    data_tc = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 19), dtype=np.float32)
    return data_tc, float(sfreq)

# ---- Switch per dataset ----

def scan_dataset_items(dataset_name: str, dataset_root: str) -> List[Tuple[str, str, str, str]]:
    """
    Return list of (path_or_dir, disease_text, subject_token, subject_dir)
    """
    dn = dataset_name.lower()
    if "apava" in dn:
        return scan_apava(dataset_root)
    if "adfsu" in dn:
        return scan_adfsu(dataset_root)
    return scan_bids_set(dataset_root)


# ============================================================
# Array-based preprocessing (APAVA / ADFSU) -> per-band segments
# ============================================================

def preprocess_tc_to_band_segs(
    data_tc: np.ndarray,
    fs_in: float,
    fs_target: int,
    sample_len: int,
    hop_len: int,
    *,
    swt_recon_map: Optional[Dict[str, set]] = None,
    swt_level: int = 4,
    wavelet: str = "sym4"
) -> Dict[str, np.ndarray]:
    """
    For array-based sources (APAVA/ADFSU):
      - bandpass 0.5–45 + resample via MNE
      - ensure 19ch
      - SWT extract + segment + zscore
    """
    if data_tc.size == 0:
        return {b: np.empty((0, sample_len, 19), dtype=np.float32) for b in BANDS}

    info = mne.create_info(ch_names=CHANNELS_19, sfreq=float(fs_in), ch_types=["eeg"] * 19)
    raw = mne.io.RawArray(data_tc.T.astype(np.float32, copy=False), info, verbose="ERROR")
    raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
    if abs(float(fs_in) - float(fs_target)) > 1e-6:
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    raw = ensure_raw_19ch(raw, CHANNELS_19)
    tc = raw.get_data().T.astype(np.float32, copy=False)

    out: Dict[str, np.ndarray] = {}
    for band_name in BANDS:
        band_tc = swt_band_extract(tc, band_name, fs_target, swt_recon_map=(swt_recon_map or SWT_RECONSTRUCTION_MAP),
                                   swt_level=swt_level, wavelet=wavelet)
        segs = segment_signal(band_tc, sample_len, hop_len)
        out[band_name] = (
            zscore_per_segment(segs).astype(np.float32, copy=False)
            if segs.shape[0] else np.empty((0, sample_len, 19), dtype=np.float32)
        )
    return out


# ============================================================
# Unified per-item preprocessing (used by scripts/prepare_data.py)
# ============================================================

def preprocess_item_to_bands(
    dataset_name: str,
    path_or_dir: str,
    *,
    fs_std_default: int,
    fs_target: int,
    sample_len: int,
    hop_len: int,
    adfsu_fs_in: float,
) -> Dict[str, np.ndarray]:
    """
    Given (dataset_name, path_or_dir), return dict band -> segments (N,L,19).
    Dataset types:
      - APAVA: .mat file
      - ADFSU: paciente dir
      - else: .set/.fif (BIDS-like)
    """
    dn_l = dataset_name.lower()

    if "apava" in dn_l:
        data_tc, fs_in = load_apava_mat_to_tc(path_or_dir)
        return preprocess_tc_to_band_segs(data_tc, fs_in, fs_target, sample_len, hop_len)

    if "adfsu" in dn_l:
        data_tc, fs_in = load_adfsu_paciente_to_tc(path_or_dir, fs_in=float(adfsu_fs_in))
        return preprocess_tc_to_band_segs(data_tc, fs_in, fs_target, sample_len, hop_len)

    # generic .set/.fif
    raw = read_filter_resample(path_or_dir, int(fs_std_default), int(fs_target), CHANNELS_19)
    raw = ensure_raw_19ch(raw, CHANNELS_19)
    data_tc = raw.get_data().T.astype(np.float32, copy=False)

    bands_segs: Dict[str, np.ndarray] = {}
    for band_name in BANDS:
        band_tc = swt_band_extract(
            data_tc,
            band_name,
            fs_target,
            swt_recon_map=SWT_RECONSTRUCTION_MAP,
            swt_level=4,
            wavelet="sym4",
        )
        segs = segment_signal(band_tc, sample_len, hop_len)
        bands_segs[band_name] = (
            zscore_per_segment(segs).astype(np.float32, copy=False)
            if segs.shape[0] else np.empty((0, sample_len, 19), dtype=np.float32)
        )
    return bands_segs


# ============================================================
# Flatten + sidecar JSON (for NPZ)
# ============================================================

def flatten_subjects(
    subject_data: Dict[str, dict],
    keys: List[str],
    band_name: str,
    *,
    include_dataset_name: bool,
    sample_len: int,
) -> Dict[str, np.ndarray]:
    """
    Flatten subject->segments dict into concatenated arrays.

    Returns dict containing:
      X, y, s, k   (+ d if include_dataset_name)
    """
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    ss: List[str] = []
    ks: List[int] = []
    ds: List[str] = []

    for key in keys:
        feats = subject_data[key]["bands"][band_name]
        if feats.shape[0] == 0:
            continue
        n = int(feats.shape[0])
        Xs.append(feats)
        ys.extend([int(subject_data[key]["label"])] * n)
        ss.extend([str(subject_data[key]["uid"])] * n)
        ks.extend([int(subject_data[key]["key_int"])] * n)
        if include_dataset_name:
            ds.extend([str(subject_data[key]["dataset"])] * n)

    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, sample_len, 19), dtype=np.float32)
    out = {
        "X": X,
        "y": np.asarray(ys, dtype=np.int32),
        "s": np.asarray(ss, dtype=str),
        "k": np.asarray(ks, dtype=np.int64),
    }
    if include_dataset_name:
        out["d"] = np.asarray(ds, dtype=str)
    return out

def _count_labels(y: np.ndarray) -> Dict[str, int]:
    if y is None or y.size == 0:
        return {}
    vals, cnts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

def _npz_stats(y: np.ndarray, s: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y) if y is not None else np.asarray([], dtype=np.int32)
    s = np.asarray(s) if s is not None else np.asarray([], dtype=str)
    return {
        "n_segments": int(y.shape[0]),
        "n_unique_subjects": int(len(np.unique(s))) if s.size else 0,
        "label_counts": _count_labels(y),
    }

def save_npz_sidecar(
    json_path: str,
    *,
    scope: str,
    dataset_name: str,
    band_name: str,
    args: Any,
    device_used: str,
    subject_index: List[Dict[str, Any]],
    payload: Dict[str, np.ndarray],
    sample_len: int,
    hop_len: int,
    other_map: Dict[str, int],
    npz_path: str,
) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": scope,
        "dataset": dataset_name,
        "band": band_name,
        "paths": {"npz": npz_path, "json": json_path},
        "device": {"requested": getattr(args, "device", None), "used": device_used},
        "signal": {
            "bandpass_hz": [0.5, 45.0],
            "fs_target": int(getattr(args, "fs_target", 128)),
            "seg_seconds": float(getattr(args, "seg_seconds", 1.0)),
            "overlap": float(getattr(args, "overlap", 0.5)),
            "sample_len": int(sample_len),
            "hop_len": int(hop_len),
            "adfsu_fs_in": float(getattr(args, "adfsu_fs_in", 128.0)),
        },
        "channels_19": CHANNELS_19,
        "swt": {
            "wavelet": "sym4",
            "level": 4,
            "bands": BANDS,
            "reconstruction_map": {k: sorted(list(v)) for k, v in SWT_RECONSTRUCTION_MAP.items()},
        },
        "label_policy": {
            "HC": 0,
            "AD": 1,
            "unknown": -1,
            "others": "2+ assigned by normalized text",
            "other_map": other_map,
        },
        "npz_keys": sorted(list(payload.keys())),
        "npz_shapes": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in payload.items()},
        "stats": _npz_stats(payload.get("y"), payload.get("s")),
        "subject_index": subject_index,
    }
    save_json(json_path, meta)
