from __future__ import annotations

import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

from src.utils.io import ensure_dir, save_npz, save_json

from src.data.preprocessing import (
    # constants
    CHANNELS_19, BANDS, SWT_RECONSTRUCTION_MAP,
    # discovery + scanning
    discover_all_dataset_roots, scan_dataset_items,
    # ids + labels
    make_key_uid, label_from_text,
    # preprocessing
    preprocess_item_to_bands,
    # flatten + sidecar
    flatten_subjects, save_npz_sidecar,
    # device
    resolve_device,
)


# src/data/prepare_data.py
# ------------------------------------------------------------
# Save ONLY IDs + labels (no train/val/test split here).
#
# SINGLE:
#   {out_dir}/single/{dataset}/{band}/{band}.npz + {band}.json
#   {out_dir}/single/{dataset}/{dataset}_meta.json
#   {out_dir}/single/single_meta.json
"""
Single + Multi cho tất cả dataset:

cd /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS
python -m scripts.prepare_data \
  --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
  --out_dir /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/outputs/cache_unified \
  --device cuda:0

Chỉ SINGLE cho 1–3 dataset

python -m scripts.prepare_data \
  --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
  --out_dir /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/outputs/cache_unified \
  --run single \
  --single_datasets ADFSU ADFTD


"""

# MULTI:
#   {out_dir}/multidataset/{band}/{band}.npz + {band}.json
#   {out_dir}/multidataset/multidataset_meta.json
# ------------------------------------------------------------
"""
Chỉ MULTI cho dataset chọn:

python -m scripts.prepare_data \
  --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
  --out_dir /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/outputs/cache_unified \
  --run multi \
  --multi_datasets BrainLat ADFTD


"""


# scripts/prepare_data.py
# ------------------------------------------------------------
# Save ONLY cached segments + ids + labels.
# No train/val/test split here (split later in training by SUBJECT).
#
# SINGLE:
#   {out_dir}/single/{dataset}/{band}/{band}.npz + {band}.json
#   {out_dir}/single/{dataset}/{dataset}_meta.json
#   {out_dir}/single/single_meta.json
#
# MULTI:
#   {out_dir}/multidataset/{band}/{band}.npz + {band}.json
#   {out_dir}/multidataset/multidataset_meta.json
#
# IMPORTANT LABELS:
#   HC=0, AD=1
#   This script will SKIP subjects with labels not in {0,1} (others / unknown),
#   to prevent invalid labels during binary training.
# ------------------------------------------------------------



def main(args: Any) -> None:
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1).")

    ensure_dir(args.out_dir)
    device_used = resolve_device(args.device)
    print(f"[DEVICE] requested={args.device} | used={device_used}")

    sample_len = int(args.fs_target * args.seg_seconds)
    hop_len = int(sample_len * (1 - args.overlap))
    if hop_len <= 0:
        raise ValueError("Invalid hop_len (check seg_seconds/overlap).")

    discovered = discover_all_dataset_roots(args.datasets_root)

    wanted_single = set(args.single_datasets or [])
    wanted_multi = set(args.multi_datasets or [])
    wanted_any = set(args.datasets or [])

    if wanted_single or wanted_multi:
        preprocess_set = wanted_single | wanted_multi
    elif wanted_any:
        preprocess_set = wanted_any
    else:
        preprocess_set = set(discovered.keys())

    preprocess_set = {d for d in preprocess_set if d in discovered}
    if not preprocess_set:
        raise RuntimeError("No datasets selected/found. Check --datasets/--single_datasets/--multi_datasets names.")

    # ---------- scan items ----------
    all_items: List[Tuple[str, str, str, str, str]] = []
    for dn in sorted(preprocess_set):
        root = discovered[dn]
        items = scan_dataset_items(dn, root)
        print(f"[SCAN] {dn}: found {len(items)} items @ {root}")
        all_items.extend([(dn, *it) for it in items])

    # ---------- preprocess + accumulate by subject ----------
    subject_data: Dict[str, dict] = {}
    other_map: Dict[str, int] = {}

    files_attempted = 0
    files_success = 0
    excluded_nonbinary = 0

    per_dataset: Dict[str, dict] = {}

    for dn, path_or_dir, disease_text, subject_token, subject_dir in tqdm(all_items, desc="Preprocessing", total=len(all_items)):
        files_attempted += 1
        per_dataset.setdefault(dn, {"files_attempted": 0, "files_success": 0, "subjects": set(), "errors": 0, "excluded_nonbinary": 0})
        per_dataset[dn]["files_attempted"] += 1

        try:
            label = int(label_from_text(disease_text, other_map))

            # binary only (HC=0, AD=1)
            if label not in (0, 1):
                excluded_nonbinary += 1
                per_dataset[dn]["excluded_nonbinary"] += 1
                continue

            key_str, uid, key_int, pid = make_key_uid(dn, subject_token)

            bands_segs = preprocess_item_to_bands(
                dataset_name=dn,
                path_or_dir=str(path_or_dir),
                fs_std_default=int(args.fs_std_default),
                fs_target=int(args.fs_target),
                sample_len=int(sample_len),
                hop_len=int(hop_len),
                adfsu_fs_in=float(args.adfsu_fs_in),
            )

            if key_str in subject_data:
                for band_name in BANDS:
                    subject_data[key_str]["bands"][band_name] = np.concatenate(
                        [subject_data[key_str]["bands"][band_name], bands_segs[band_name]],
                        axis=0,
                    )
            else:
                subject_data[key_str] = {
                    "bands": bands_segs,
                    "label": label,
                    "uid": uid,
                    "pid": int(pid),
                    "key_int": int(key_int),
                    "dataset": dn,
                    "subject_token": subject_token,
                    "subject_dir": subject_dir,
                }

            files_success += 1
            per_dataset[dn]["files_success"] += 1
            per_dataset[dn]["subjects"].add(key_str)

        except Exception as e:
            per_dataset[dn]["errors"] += 1
            print(f"[ERROR] {dn} | {os.path.basename(str(path_or_dir))}: {e}")

    for dn in per_dataset:
        per_dataset[dn]["subjects"] = len(per_dataset[dn]["subjects"])

    print(f"[DONE] subjects={len(subject_data)} files_attempted={files_attempted} files_success={files_success} excluded_nonbinary={excluded_nonbinary}")
    if len(subject_data) == 0:
        raise RuntimeError("No subjects processed (binary HC/AD). Check labels/scanners/paths.")

    # group by dataset
    subjects_by_dataset: Dict[str, List[str]] = {}
    for k, v in subject_data.items():
        subjects_by_dataset.setdefault(v["dataset"], []).append(k)
    for dn in subjects_by_dataset:
        subjects_by_dataset[dn].sort()

    processed_datasets = sorted(subjects_by_dataset.keys())

    def resolve_save_set(wanted: set) -> List[str]:
        if wanted:
            return sorted([d for d in wanted if d in subjects_by_dataset])
        if wanted_any:
            return sorted([d for d in wanted_any if d in subjects_by_dataset])
        return processed_datasets

    save_single_datasets = resolve_save_set(wanted_single)
    save_multi_datasets = resolve_save_set(wanted_multi)

    # ============================================================
    # SAVE: SINGLE
    # ============================================================
    if args.run in {"single", "both"}:
        single_root = os.path.join(args.out_dir, "single")
        ensure_dir(single_root)

        single_global = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {"datasets_root": args.datasets_root, "out_dir": args.out_dir, "single_root": single_root},
            "device": {"requested": args.device, "used": device_used},
            "signal": {
                "bandpass_hz": [0.5, 45.0],
                "fs_target": int(args.fs_target),
                "seg_seconds": float(args.seg_seconds),
                "overlap": float(args.overlap),
                "sample_len": int(sample_len),
                "hop_len": int(hop_len),
                "adfsu_fs_in": float(args.adfsu_fs_in),
            },
            "channels_19": CHANNELS_19,
            "label_policy": {"HC": 0, "AD": 1, "unknown": -1, "other_map": other_map},
            "datasets": [],
            "dataset_metas": {},
        }

        for dn in save_single_datasets:
            dn_root = os.path.join(single_root, dn)
            ensure_dir(dn_root)

            subject_keys = subjects_by_dataset[dn]
            subject_index = [{
                "key": int(subject_data[k]["key_int"]),
                "key_str": k,
                "uid": subject_data[k]["uid"],
                "label": int(subject_data[k]["label"]),
                "pid": int(subject_data[k]["pid"]),
                "subject_token": subject_data[k]["subject_token"],
                "subject_dir": subject_data[k]["subject_dir"],
            } for k in subject_keys]

            band_files, band_jsons, band_shapes = {}, {}, {}

            for band_name in BANDS:
                out_band_dir = os.path.join(dn_root, band_name)
                ensure_dir(out_band_dir)

                out_npz = os.path.join(out_band_dir, f"{band_name}.npz")
                out_json = os.path.join(out_band_dir, f"{band_name}.json")

                flat = flatten_subjects(subject_data, subject_keys, band_name, include_dataset_name=False, sample_len=sample_len)
                payload = {
                    f"X_{band_name}": flat["X"],
                    "y": flat["y"],
                    "s": flat["s"],
                    "k": flat["k"],
                }

                save_npz(out_npz, **payload)
                save_npz_sidecar(
                    out_json,
                    scope="single",
                    dataset_name=dn,
                    band_name=band_name,
                    args=args,
                    device_used=device_used,
                    subject_index=subject_index,
                    payload={"X": payload[f"X_{band_name}"], "y": payload["y"], "s": payload["s"], "k": payload["k"]},
                    sample_len=sample_len,
                    hop_len=hop_len,
                    other_map=other_map,
                    npz_path=out_npz,
                )

                band_files[band_name] = out_npz
                band_jsons[band_name] = out_json
                band_shapes[band_name] = {k: list(v.shape) for k, v in payload.items()}

            dataset_meta_path = os.path.join(dn_root, f"{dn}_meta.json")
            save_json(dataset_meta_path, {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dn,
                "paths": {"dataset_root": discovered.get(dn, ""), "single_dataset_dir": dn_root},
                "device": {"requested": args.device, "used": device_used},
                "signal": {
                    "bandpass_hz": [0.5, 45.0],
                    "fs_target": int(args.fs_target),
                    "seg_seconds": float(args.seg_seconds),
                    "overlap": float(args.overlap),
                    "sample_len": int(sample_len),
                    "hop_len": int(hop_len),
                    "adfsu_fs_in": float(args.adfsu_fs_in),
                },
                "channels_19": CHANNELS_19,
                "swt": {"wavelet": "sym4", "level": 4, "bands": BANDS,
                        "reconstruction_map": {k: sorted(list(v)) for k, v in SWT_RECONSTRUCTION_MAP.items()}},
                "label_policy": {"HC": 0, "AD": 1, "unknown": -1, "other_map": other_map},
                "counts": {
                    "subjects": len(subject_keys),
                    "files_attempted": per_dataset.get(dn, {}).get("files_attempted", 0),
                    "files_success": per_dataset.get(dn, {}).get("files_success", 0),
                    "errors": per_dataset.get(dn, {}).get("errors", 0),
                    "excluded_nonbinary": per_dataset.get(dn, {}).get("excluded_nonbinary", 0),
                },
                "subject_index": subject_index,
                "band_files": band_files,
                "band_jsons": band_jsons,
                "shapes": band_shapes,
                "split_policy": "none (split later in training by SUBJECT ids)",
            })

            single_global["datasets"].append(dn)
            single_global["dataset_metas"][dn] = dataset_meta_path
            print(f"[SINGLE] saved dataset={dn} -> {dn_root}")

        save_json(os.path.join(single_root, "single_meta.json"), single_global)

    # ============================================================
    # SAVE: MULTI
    # ============================================================
    if args.run in {"multi", "both"}:
        multi_root = os.path.join(args.out_dir, "multidataset")
        ensure_dir(multi_root)

        merged_subject_keys: List[str] = []
        for dn in save_multi_datasets:
            merged_subject_keys.extend(subjects_by_dataset.get(dn, []))

        # unique subject index
        seen = set()
        subject_index_all = []
        for k in merged_subject_keys:
            if k in seen:
                continue
            seen.add(k)
            subject_index_all.append({
                "key": int(subject_data[k]["key_int"]),
                "key_str": k,
                "uid": subject_data[k]["uid"],
                "label": int(subject_data[k]["label"]),
                "pid": int(subject_data[k]["pid"]),
                "dataset": subject_data[k]["dataset"],
                "subject_token": subject_data[k]["subject_token"],
                "subject_dir": subject_data[k]["subject_dir"],
            })
        subject_index_all.sort(key=lambda x: (x["dataset"], x["key"]))

        band_files, band_jsons, band_shapes = {}, {}, {}

        for band_name in BANDS:
            out_band_dir = os.path.join(multi_root, band_name)
            ensure_dir(out_band_dir)

            out_npz = os.path.join(out_band_dir, f"{band_name}.npz")
            out_json = os.path.join(out_band_dir, f"{band_name}.json")

            flat = flatten_subjects(subject_data, merged_subject_keys, band_name, include_dataset_name=True, sample_len=sample_len)
            payload = {
                f"X_{band_name}": flat["X"],
                "y": flat["y"],
                "s": flat["s"],
                "k": flat["k"],
                "d": flat["d"],
            }

            save_npz(out_npz, **payload)
            save_npz_sidecar(
                out_json,
                scope="multidataset",
                dataset_name="multidataset",
                band_name=band_name,
                args=args,
                device_used=device_used,
                subject_index=subject_index_all,
                payload={"X": payload[f"X_{band_name}"], "y": payload["y"], "s": payload["s"], "k": payload["k"]},
                sample_len=sample_len,
                hop_len=hop_len,
                other_map=other_map,
                npz_path=out_npz,
            )

            band_files[band_name] = out_npz
            band_jsons[band_name] = out_json
            band_shapes[band_name] = {k: list(v.shape) for k, v in payload.items()}

        out_multi_meta = os.path.join(multi_root, "multidataset_meta.json")
        save_json(out_multi_meta, {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {"datasets_root": args.datasets_root, "out_dir": args.out_dir, "multidataset_dir": multi_root},
            "device": {"requested": args.device, "used": device_used},
            "signal": {
                "bandpass_hz": [0.5, 45.0],
                "fs_target": int(args.fs_target),
                "seg_seconds": float(args.seg_seconds),
                "overlap": float(args.overlap),
                "sample_len": int(sample_len),
                "hop_len": int(hop_len),
                "adfsu_fs_in": float(args.adfsu_fs_in),
            },
            "channels_19": CHANNELS_19,
            "swt": {"wavelet": "sym4", "level": 4, "bands": BANDS,
                    "reconstruction_map": {k: sorted(list(v)) for k, v in SWT_RECONSTRUCTION_MAP.items()}},
            "label_policy": {"HC": 0, "AD": 1, "unknown": -1, "other_map": other_map},
            "counts": {
                "files_attempted": int(files_attempted),
                "files_success": int(files_success),
                "subjects_total_processed": int(len(subject_data)),
                "subjects_in_multi": int(len(subject_index_all)),
                "excluded_nonbinary": int(excluded_nonbinary),
                "per_dataset": per_dataset,
            },
            "selected_datasets_for_multi": save_multi_datasets,
            "subject_index": subject_index_all,
            "band_files": band_files,
            "band_jsons": band_jsons,
            "shapes": band_shapes,
            "split_policy": "none (split later in training by SUBJECT ids)",
        })

        print(f"[MULTI] saved -> {multi_root} (datasets={save_multi_datasets}, subjects={len(subject_index_all)})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--fs_std_default", type=int, default=500)
    p.add_argument("--fs_target", type=int, default=128)
    p.add_argument("--seg_seconds", type=float, default=1.0)
    p.add_argument("--overlap", type=float, default=0.5)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--adfsu_fs_in", type=float, default=128.0)

    p.add_argument("--run", type=str, default="both", choices=["single", "multi", "both"])
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--single_datasets", nargs="*", default=None)
    p.add_argument("--multi_datasets", nargs="*", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
