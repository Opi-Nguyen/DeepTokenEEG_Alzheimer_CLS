import os
import re
import numpy as np
import mne
import pywt

# ---------------- SWT helpers ----------------
def _next_pow2_multiple(n, level):
    m = 2 ** level
    return n if n % m == 0 else n + (m - n % m)

def fir_bandpass(data_1d, fs, l_freq, h_freq):
    # giữ lại nếu bạn muốn dùng ở nơi khác; swt_band_extract sẽ không dùng nữa
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

    # --- channel selection FIRST (19 channels) ---
    if len(raw.ch_names) > 30:
        channels_to_pick = find_closest_biosemi_channels(pick_channels_19)
    else:
        channels_to_pick = [ch for ch in pick_channels_19 if ch in raw.ch_names]

    if len(channels_to_pick) == 0:
        raise ValueError("No matching EEG channels found for the requested 19-channel set.")

    raw.pick(channels_to_pick)

    # optional: standardize to fs_std first (giữ logic cũ, nhưng giờ chỉ resample 19 kênh)
    if abs(raw.info["sfreq"] - fs_std) > 1:
        raw.resample(fs_std, npad="auto", verbose="ERROR")

    # broad bandpass filter 0.5–45 Hz
    raw.filter(0.5, 45.0, method="fir", phase="zero-double", verbose="ERROR")

    # resample to fs_target (expect 128)
    if abs(raw.info["sfreq"] - fs_target) > 1:
        raw.resample(fs_target, npad="auto", verbose="ERROR")

    return raw

def swt_band_extract(
    data_tc, band_name, fs_target, band_presets,
    swt_recon_map, swt_level=4, wavelet="sym4"
):
    """
    data_tc: [T, C]  (đã bandpass 0.5–45 và resample fs_target=128)
    returns: [T, C]

    SWT level=4 @ fs=128:
      A4: 0–4   (Delta)
      D4: 4–8   (Theta)
      D3: 8–16  (Alpha)
      D2: 16–32 (Beta)
      D1: 32–64 (Gamma)  (thực tế bị giới hạn đến 45Hz vì đã lowpass 45)
    fullband (A0): trả về data_tc
    """
    band_name = band_name.lower().strip()

    # default map chuẩn nếu không truyền map hoặc map thiếu key
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
        # bổ sung key nào thiếu
        for k, v in default_map.items():
            swt_recon_map.setdefault(k, v)

    if band_name in ("fullband", "a0"):
        return data_tc.astype(np.float32, copy=False)

    if band_name not in swt_recon_map:
        raise ValueError(f"Unknown band_name={band_name}. Expected one of {list(swt_recon_map.keys())}")

    keep = {x.upper() for x in swt_recon_map[band_name]}

    T, C = data_tc.shape
    out = np.empty_like(data_tc, dtype=np.float32)

    # SWT requires length multiple of 2^level -> pad and crop back
    m = 2 ** swt_level

    for c in range(C):
        x = data_tc[:, c].astype(np.float32)
        orig_len = x.shape[0]

        padded_len = orig_len if (orig_len % m == 0) else (orig_len + (m - orig_len % m))
        pad_amount = padded_len - orig_len
        left_pad = pad_amount // 2
        x_pad = np.pad(x, (left_pad, pad_amount - left_pad), mode="symmetric")

        # Use SWT then reconstruct by selecting components at the requested level(s)
        # coeffs list index: level 1..L, each is (cA, cD)
        coeffs = pywt.swt(x_pad, wavelet, level=swt_level, trim_approx=False)

        # reconstruct by summing selected MRA components:
        # We build MRA components via "keep sets" using iswt trick:
        # For A4: keep A4 and zero all D's
        # For Dk: keep only Dk (and keep A at all levels as required by iswt with zeros elsewhere)
        def _recon_component(comp_name):
            comp_name = comp_name.upper()
            kept = []
            for j, (cA, cD) in enumerate(coeffs, start=1):
                if comp_name == f"A{swt_level}":
                    # keep only the approximation at the FINAL level; set all details to 0
                    cA_keep = cA if (j == swt_level) else cA * 0.0
                    cD_keep = cD * 0.0
                else:
                    # keep only the requested detail Dk
                    cA_keep = cA * 0.0
                    cD_keep = cD if comp_name == f"D{j}" else cD * 0.0
                kept.append((cA_keep, cD_keep))
            return pywt.iswt(kept, wavelet)

        y = np.zeros_like(x_pad, dtype=np.float32)
        for nm in keep:
            if nm == "A0":
                y += x_pad
            elif nm.startswith("A") or nm.startswith("D"):
                y += _recon_component(nm).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Invalid SWT component name: {nm}")

        # crop back
        y = y[left_pad:left_pad + orig_len]
        out[:, c] = y.astype(np.float32, copy=False)

    return out

def segment_signal(data_tc, segment_length, hop_length):
    T, C = data_tc.shape
    if T < segment_length:
        return np.empty((0, segment_length, C), dtype=np.float32)
    starts = np.arange(0, T - segment_length + 1, hop_length, dtype=int)
    segs = np.array([data_tc[s:s + segment_length] for s in starts], dtype=np.float32)
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
