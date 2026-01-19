import os
import re
import random
import numpy as np
import mne
import pywt

# ---------------- SWT helpers ----------------
def _next_pow2_multiple(n, level):
    m = 2**level
    return n if n % m == 0 else n + (m - n % m)

def _swt_reconstruct(coeffs, wavelet, keep_components):
    kept = []
    for j, (cA, cD) in enumerate(coeffs, start=1):
        cA_keep = cA if f"A{j}" in keep_components else np.zeros_like(cA)
        cD_keep = cD if f"D{j}" in keep_components else np.zeros_like(cD)
        kept.append((cA_keep, cD_keep))
    return pywt.iswt(kept, wavelet)

def fir_bandpass(data_1d, fs, l_freq, h_freq):
    return mne.filter.filter_data(
        data_1d.astype(np.float64),
        sfreq=fs, l_freq=l_freq, h_freq=h_freq,
        method="fir", phase="zero-double", verbose="ERROR"
    ).astype(np.float32)

def find_closest_biosemi_channels(input_channels):
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

# ---------------- Main steps ----------------
def read_filter_resample(file_path, fs_std, fs_target, pick_channels_19):
    raw = (
        mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
        if file_path.endswith(".fif")
        else mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    )
    raw.pick("eeg")

    if abs(raw.info["sfreq"] - fs_std) > 1:
        raw.resample(fs_std, npad="auto", verbose="ERROR")

    # broad filter
    raw.filter(0.5, 45.0, method="fir", phase="zero-double", verbose="ERROR")

    if abs(raw.info["sfreq"] - fs_target) > 1:
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    # channel selection
    if len(raw.ch_names) > 30:
        channels_to_pick = find_closest_biosemi_channels(pick_channels_19)
    else:
        channels_to_pick = [ch for ch in pick_channels_19 if ch in raw.ch_names]
    raw.pick(channels_to_pick)
    return raw

def swt_band_extract(data_tc, band_name, fs_target, band_presets,
                     swt_recon_map, swt_level=6, wavelet="db8"):
    """
    data_tc: [T, C]
    returns: [T, C]
    """
    T, C = data_tc.shape
    out = np.empty_like(data_tc, dtype=np.float32)
    keep = swt_recon_map[band_name]
    l_freq, h_freq = band_presets[band_name]

    for c in range(C):
        x = data_tc[:, c].astype(np.float32)
        orig_len = len(x)

        padded_len = _next_pow2_multiple(orig_len, swt_level)
        pad_amount = padded_len - orig_len
        left_pad = pad_amount // 2

        x_pad = np.pad(x, (left_pad, pad_amount - left_pad), mode="symmetric")
        coeffs = pywt.swt(x_pad, wavelet, level=swt_level, trim_approx=False)
        recon = _swt_reconstruct(coeffs, wavelet, keep)
        recon = recon[left_pad:left_pad + orig_len]

        out[:, c] = fir_bandpass(recon, fs_target, l_freq, h_freq)
    return out

def segment_signal(data_tc, segment_length, hop_length):
    T, C = data_tc.shape
    if T < segment_length:
        return np.empty((0, segment_length, C), dtype=np.float32)
    starts = np.arange(0, T - segment_length + 1, hop_length, dtype=int)
    segs = np.array([data_tc[s:s+segment_length] for s in starts], dtype=np.float32)
    return segs

def zscore_per_segment(segs, eps=1e-6):
    mu = segs.mean(axis=1, keepdims=True)
    sd = segs.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return ((segs - mu) / sd).astype(np.float32)

def extract_pid(subject_dir_name: str):
    m = re.search(r"\d+", subject_dir_name)
    return int(m.group(0)) if m else subject_dir_name

def scan_subject_files(raw_root, label_map):
    """
    yields tuples: (file_path, disease_folder, subject_dir)
    """
    for disease_folder in sorted(os.listdir(raw_root)):
        if not any(k in disease_folder for k in label_map.keys()):
            continue
        for sub_folder_name in ["AR", "CL"]:
            sub_path = os.path.join(raw_root, disease_folder, sub_folder_name)
            if not os.path.isdir(sub_path):
                continue
            for subject_dir in sorted(os.listdir(sub_path)):
                subject_path = os.path.join(sub_path, subject_dir)
                if not os.path.isdir(subject_path):
                    continue
                set_files = [
                    os.path.join(root, f)
                    for root, _, files in os.walk(subject_path)
                    for f in files if f.endswith(".set")
                ]
                for fpath in set_files:
                    yield fpath, disease_folder, subject_dir
