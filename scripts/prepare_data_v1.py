# Đây là bản code cũ  ( phiên bản đầu tiên), có 2 vấn đề: 1- SWT chưa đúng wavelet đang xét (code này dùng db8 nhưng bài báo của mình là sym4) + level 2- gán nhãn các dải tần sai với tiêu chuẩn

import os
import re
import yaml
import numpy as np
from tqdm import tqdm

from src.data.preprocessing import (
    scan_subject_files, read_filter_resample, swt_band_extract,
    segment_signal, zscore_per_segment, extract_pid
)
from src.data.split import make_subject_splits
from src.utils.io import ensure_dir, save_npz, save_json

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_data_path="configs/data.yaml"):
    cfg = load_yaml(cfg_data_path)

    raw_root = cfg["paths"]["raw_root"]
    cache_dir = cfg["paths"]["cache_dir"]
    ensure_dir(cache_dir)

    fs_std = cfg["signal"]["fs_std"]
    fs_target = cfg["signal"]["fs_target"]
    seg_seconds = cfg["signal"]["seg_seconds"]
    overlap = cfg["signal"]["overlap"]
    band_name = cfg["signal"]["band_name"]

    channels_19 = []
    for row in cfg["channels"]["standard_19"]:
        channels_19.extend([x.strip() for x in row.split("-")])
    label_map = cfg["labels"]
    ratios = cfg["split"]["ratios"]
    seed = cfg["split"]["seed"]

    SAMPLE_LEN = int(fs_target * seg_seconds)
    HOP_LEN = int(SAMPLE_LEN * (1 - overlap))

    BAND_PRESETS = {
        "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.9),
        "beta": (13.0, 30.0), "gamma": (32.0, 45.0)
    }
    SWT_RECONSTRUCTION_MAP = {
        "delta": {"A6", "D6", "D5"}, "theta": {"D4"}, "alpha": {"D3"},
        "beta": {"D3", "D2"}, "gamma": {"D1"}
    }

    subject_data = {}  # pid -> {"features":[N,T,C], "label": int}
    files_attempted = 0
    files_success = 0

    items = list(scan_subject_files(raw_root, label_map))
    print(f"Found {len(items)} .set files to scan. Band={band_name}")

    for fpath, disease_folder, subject_dir in tqdm(items):
        files_attempted += 1
        try:
            raw = read_filter_resample(fpath, fs_std, fs_target, channels_19)
            data_tc = raw.get_data().T.astype(np.float32)   # [T,C]

            data_band = swt_band_extract(
                data_tc, band_name, fs_target,
                band_presets=BAND_PRESETS,
                swt_recon_map=SWT_RECONSTRUCTION_MAP,
                swt_level=6, wavelet="db8"
            )
            segs = segment_signal(data_band, SAMPLE_LEN, HOP_LEN)  # [N, T, C]
            if segs.shape[0] == 0:
                continue
            segs = zscore_per_segment(segs)

            pid = extract_pid(subject_dir)
            # label by folder contains "HC" or "AD"
            label_key = next(k for k in label_map.keys() if k in disease_folder)
            label = int(label_map[label_key])

            if pid in subject_data:
                subject_data[pid]["features"] = np.concatenate([subject_data[pid]["features"], segs], axis=0)
            else:
                subject_data[pid] = {"features": segs, "label": label}

            files_success += 1
        except Exception as e:
            print(f"[ERROR] {os.path.basename(fpath)}: {e}")

    print("Subjects:", len(subject_data), "files_attempted:", files_attempted, "files_success:", files_success)

    splits = make_subject_splits(subject_data, ratios, seed)
    train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

    def flatten(ids):
        Xs, ys, ss = [], [], []
        for pid in ids:
            feats = subject_data[pid]["features"]
            Xs.append(feats)
            ys.extend([subject_data[pid]["label"]] * feats.shape[0])
            ss.extend([pid] * feats.shape[0])
        X = np.concatenate(Xs, axis=0) if len(Xs) else np.empty((0, SAMPLE_LEN, len(channels_19)), dtype=np.float32)
        y = np.asarray(ys, dtype=int)
        s = np.asarray(ss, dtype=int)
        return X, y, s


    out_npz = os.path.join(cache_dir, f"dataset_{band_name}.npz")
    out_meta = os.path.join(cache_dir, f"dataset_{band_name}_meta.json")

    save_npz(
        out_npz,
        X_train=X_train, y_train=y_train, s_train=s_train,
        X_val=X_val, y_val=y_val, s_val=s_val,
        X_test=X_test, y_test=y_test, s_test=s_test,
    )
    save_json(out_meta, {
        "band_name": band_name,
        "fs_std": fs_std,
        "fs_target": fs_target,
        "seg_seconds": seg_seconds,
        "overlap": overlap,
        "sample_len": SAMPLE_LEN,
        "hop_len": HOP_LEN,
        "channels_19": channels_19,
        "files_attempted": files_attempted,
        "files_success": files_success,
        "n_subjects": len(subject_data),
        "splits": splits,
    })

    print("Saved:", out_npz)
    print("Saved:", out_meta)
    print("Train:", X_train.shape, y_train.shape, s_train.shape)
    print("Val:  ", X_val.shape, y_val.shape, s_val.shape)
    print("Test: ", X_test.shape, y_test.shape, s_test.shape)

if __name__ == "__main__":
    main()
