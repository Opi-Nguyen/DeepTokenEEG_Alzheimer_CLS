# scripts/train.py
import os
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.seed import seed_everything
from src.utils.io import load_json, load_npz, ensure_dir, save_json
from src.data.dataset import IndexedEEGDataset, make_loader
from src.data.split import split_by_dataset_then_merge
from src.models.model import Model
from src.train.trainer import train_model

"""
Train one model (model1 - model 2 - ...)

python -m scripts.train --cfg_data configs/data.yaml --cfg_model configs/model1.yaml --cfg_train configs/train.yaml --cfg_exp configs/experiment.yaml

Train all model 

python -m scripts.train --cfg_data configs/data.yaml --cfg_models configs/model1.yaml configs/model2.yaml configs/model3.yaml configs/model4.yaml configs/model5.yaml --cfg_train configs/train.yaml --cfg_exp configs/experiment.yaml

"""

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(train_cfg):
    dev = train_cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _load_cache_from_band_dir(cache_dir: str, band_name: str):
    npz_path = os.path.join(cache_dir, f"{band_name}.npz")
    json_path = os.path.join(cache_dir, f"{band_name}.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing sidecar JSON: {json_path}")
    cache = load_npz(npz_path)
    meta = load_json(json_path)
    cache_dict = {k: cache[k] for k in cache.files}
    return cache_dict, meta, npz_path, json_path


def _build_segment_index_by_subject(s: np.ndarray):
    mp = {}
    for i, uid in enumerate(s.tolist()):
        mp.setdefault(uid, []).append(i)
    return mp


def _infer_dataset_from_uid(uid: str) -> str:
    return uid.split(":", 1)[0] if ":" in uid else "single"


def _normalize_subject_index(subject_index: list):
    # ensure each item has dataset field
    out = []
    for it in subject_index:
        it2 = dict(it)
        uid = str(it2.get("uid", ""))
        if "dataset" not in it2 or not it2["dataset"]:
            it2["dataset"] = _infer_dataset_from_uid(uid)
        out.append(it2)
    return out


def _filter_only_hc_ad_and_remap(y, subject_index, cfg_labels):
    """
    Enforce ONLY HC/AD and map to cfg: HC=0, AD=1 by default.
    Preprocessing sidecar already should be HC=0 AD=1, nhưng vẫn check để tránh lỗi CUDA assert.
    """
    hc_id = int(cfg_labels.get("HC", 0))
    ad_id = int(cfg_labels.get("AD", 1))
    allowed = {hc_id, ad_id}

    # subject_index filter
    kept_uids = set()
    new_subject_index = []
    for it in subject_index:
        lb = int(it.get("label", -999))
        if lb in allowed:
            new_subject_index.append(it)
            kept_uids.add(str(it["uid"]))

    # segment-level filter: keep only y in allowed (an toàn)
    y = np.asarray(y, dtype=np.int32)
    keep_seg = np.isin(y, np.array(sorted(list(allowed)), dtype=np.int32))
    return keep_seg, new_subject_index, kept_uids


def _derive_tags_from_cache_dir(cache_dir: str):
    # expected .../single/<DATASET>/<BAND>
    band = os.path.basename(cache_dir.rstrip("/"))
    parent = os.path.basename(os.path.dirname(cache_dir.rstrip("/")))
    # scope
    scope = "single" if ("/single/" in cache_dir.replace("\\", "/")) else ("multidataset" if ("/multidataset/" in cache_dir.replace("\\", "/")) else "cache")
    dataset_tag = parent if scope == "single" else "multidataset"
    return scope, dataset_tag, band


def _iter_model_paths(cfg_model: str, cfg_models: list | None):
    if cfg_models and len(cfg_models) > 0:
        return cfg_models
    return [cfg_model]


def run_one(
    *,
    data_cfg: dict,
    train_cfg: dict,
    exp_cfg: dict,
    model_yaml_path: str,
    model_base_cfg: dict,
    sidecar: dict,
    cache_dict: dict,
    npz_path: str,
    json_path: str,
    seed: int,
    n_blocks_override: int | None,
):
    # ---------- load arrays ----------
    band_name = str(data_cfg["signal"]["band_name"])
    X_key = f"X_{band_name}"
    if X_key not in cache_dict:
        raise KeyError(f"NPZ missing key {X_key}. Available keys={list(cache_dict.keys())}")

    X = cache_dict[X_key]
    y = cache_dict["y"]
    s = cache_dict["s"].astype(str)
    k = cache_dict.get("k", None)

    # ---------- subject_index ----------
    subject_index = sidecar.get("subject_index", [])
    subject_index = _normalize_subject_index(subject_index)

    cfg_labels = data_cfg.get("labels", {"HC": 0, "AD": 1})
    keep_seg_mask, subject_index, kept_uids = _filter_only_hc_ad_and_remap(y, subject_index, cfg_labels)

    X = X[keep_seg_mask]
    y = np.asarray(y[keep_seg_mask], dtype=np.int32)
    s = s[keep_seg_mask]
    if k is not None:
        k = k[keep_seg_mask]

    # ---------- label sanity ----------
    num_class = int(model_base_cfg["num_class"])
    uniq = sorted(set(int(v) for v in y.tolist()))
    bad = [v for v in uniq if v < 0 or v >= num_class]
    if bad:
        raise RuntimeError(f"Invalid labels found {bad} with num_class={num_class}. Check labels in cache/config.")

    # ---------- split subjects ----------
    ratios = tuple(float(x) for x in data_cfg["split"]["ratios"])
    split_pack = split_by_dataset_then_merge(
        subject_index=subject_index,
        ratios=ratios,
        seed=int(seed),
        allowed_labels=tuple(sorted(set(uniq))),
    )
    splits = split_pack["splits"]
    per_dataset = split_pack["per_dataset"]

    # compute all_uids and test_uids as remaining (đúng yêu cầu “trừ ids đã dùng”)
    all_uids = sorted(list({str(it["uid"]) for it in subject_index}))
    used_trainval = set(splits["train_uids"]) | set(splits["val_uids"])
    test_uids = sorted([u for u in all_uids if u not in used_trainval])
    splits = dict(splits)
    splits["test_uids"] = test_uids

    uid2idx = _build_segment_index_by_subject(s)

    def collect_indices(uids):
        out = []
        for uid in uids:
            out.extend(uid2idx.get(uid, []))
        return np.asarray(out, dtype=np.int64)

    train_idx = collect_indices(splits["train_uids"])
    val_idx = collect_indices(splits["val_uids"])
    test_idx = collect_indices(splits["test_uids"])

    # ---------- loaders ----------
    batch_size = int(train_cfg["batch_size"])
    train_ds = IndexedEEGDataset(X, y, s, k, train_idx)
    val_ds = IndexedEEGDataset(X, y, s, k, val_idx)

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = pick_device(train_cfg)

    # ---------- effective model cfg (override n_blocks if requested) ----------
    model_cfg = copy.deepcopy(model_base_cfg)
    if n_blocks_override is not None:
        model_cfg["resnet"]["n_blocks"] = int(n_blocks_override)

    n_blocks = int(model_cfg["resnet"]["n_blocks"])
    dilations = model_cfg["resnet"]["dilations"]
    model_tag = os.path.splitext(os.path.basename(model_yaml_path))[0]

    cache_dir = data_cfg["paths"]["cache_dir"]
    scope, dataset_tag, band_tag = _derive_tags_from_cache_dir(cache_dir)

    run_dir = os.path.join(
        exp_cfg["out_dir"],
        exp_cfg["name"],
        scope,
        dataset_tag,
        band_tag,
        model_tag,
        f"seed_{seed}",
        f"blocks_{n_blocks}_dilations_{'_'.join(map(str, dilations))}",
    )
    ensure_dir(run_dir)

    best_path = os.path.join(run_dir, "best.pth")

    # ---------- save splits + snapshot (eval đọc dùng chung) ----------
    save_json(os.path.join(run_dir, "splits_subject.json"), {
        "cache": {"npz": npz_path, "json": json_path, "cache_dir": cache_dir, "band": band_name},
        "labels_cfg": cfg_labels,
        "num_class": num_class,
        "split": {"ratios": ratios, "seed": int(seed)},
        "all_uids": all_uids,
        "splits": splits,
        "per_dataset": per_dataset,
        "counts": {
            "subjects_total": len(all_uids),
            "subjects_train": len(splits["train_uids"]),
            "subjects_val": len(splits["val_uids"]),
            "subjects_test": len(splits["test_uids"]),
            "segments_train": int(train_idx.size),
            "segments_val": int(val_idx.size),
            "segments_test": int(test_idx.size),
        },
    })

    save_json(os.path.join(run_dir, "config_snapshot.json"), {
        "data": data_cfg,
        "train": train_cfg,
        "experiment": exp_cfg,
        "model_effective": model_cfg,
        "model_yaml": model_yaml_path,
        "sidecar": {
            "channels_19": sidecar.get("channels_19", None),
            "label_policy": sidecar.get("label_policy", None),
            "signal": sidecar.get("signal", None),
        },
    })

    # ---------- init model ----------
    enc_in = len(sidecar.get("channels_19", [])) or 19
    model = Model(
        enc_in=int(enc_in),
        num_class=int(num_class),
        d_model=int(model_cfg["d_model"]),
        dropout=float(model_cfg["dropout"]),
        n_blocks=int(model_cfg["resnet"]["n_blocks"]),
        dilations=model_cfg["resnet"]["dilations"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min",
        factor=float(train_cfg["lr_factor"]),
        patience=int(train_cfg["lr_patience"]),
    )

    # optional training log yaml
    log_yaml_path = os.path.join(run_dir, "train_log.yaml")

    model, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device=device,
        epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]),
        best_model_path=best_path,
        log_yaml_path=log_yaml_path,
        run_info={"run_dir": run_dir, "model_tag": model_tag, "seed": int(seed)},
    )

    save_json(os.path.join(run_dir, "train_summary.json"), {
        "best_val_loss": float(best_val_loss),
        "best_ckpt": best_path,
        "device": str(device),
    })


def main(cfg_data="configs/data.yaml", cfg_model="configs/model.yaml",
         cfg_train="configs/train.yaml", cfg_exp="configs/experiment.yaml",
         cfg_models=None):

    data_cfg = load_yaml(cfg_data)
    train_cfg = load_yaml(cfg_train)["train"]
    exp_all = load_yaml(cfg_exp)
    exp_cfg = exp_all["run"]
    ab_cfg = exp_all.get("ablation", {})

    band_name = data_cfg["signal"]["band_name"]
    cache_dir = data_cfg["paths"]["cache_dir"]
    cache_dict, sidecar, npz_path, json_path = _load_cache_from_band_dir(cache_dir, band_name)

    model_paths = _iter_model_paths(cfg_model, cfg_models)

    seeds = ab_cfg.get("seeds", [int(data_cfg["split"]["seed"])])
    n_blocks_list = ab_cfg.get("n_blocks_list", None)
    do_ablation = bool(train_cfg.get("ablation", False)) and (n_blocks_list is not None)

    for model_yaml_path in model_paths:
        model_base_cfg = load_yaml(model_yaml_path)["model"]

        for seed in seeds:
            seed_everything(int(seed))

            if do_ablation:
                for nb in n_blocks_list:
                    run_one(
                        data_cfg=data_cfg,
                        train_cfg=train_cfg,
                        exp_cfg=exp_cfg,
                        model_yaml_path=model_yaml_path,
                        model_base_cfg=model_base_cfg,
                        sidecar=sidecar,
                        cache_dict=cache_dict,
                        npz_path=npz_path,
                        json_path=json_path,
                        seed=int(seed),
                        n_blocks_override=int(nb),
                    )
            else:
                run_one(
                    data_cfg=data_cfg,
                    train_cfg=train_cfg,
                    exp_cfg=exp_cfg,
                    model_yaml_path=model_yaml_path,
                    model_base_cfg=model_base_cfg,
                    sidecar=sidecar,
                    cache_dict=cache_dict,
                    npz_path=npz_path,
                    json_path=json_path,
                    seed=int(seed),
                    n_blocks_override=None,
                )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_data", default="configs/data.yaml")
    ap.add_argument("--cfg_model", default="configs/model.yaml")
    ap.add_argument("--cfg_models", nargs="*", default=None)  # NEW: many model yamls
    ap.add_argument("--cfg_train", default="configs/train.yaml")
    ap.add_argument("--cfg_exp", default="configs/experiment.yaml")
    args = ap.parse_args()
    main(args.cfg_data, args.cfg_model, args.cfg_train, args.cfg_exp, args.cfg_models)
