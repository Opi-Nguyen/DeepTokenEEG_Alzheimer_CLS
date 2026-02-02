from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------
# Defaults (khi không truyền config)
# ---------------------------
DEFAULTS: Dict[str, Any] = {
    # dataset
    "data_root": "",
    "scope": "single",         # single | multidataset
    "dataset": "",
    "band_set": ["alpha"],

    # output
    "outdir": "outputs/runs",
    "run_name": "run",
    "save_resolved_config": True,

    # device
    "device": "cuda:0",
    "amp": True,
    "cudnn_benchmark": True,
    "cuda_deterministic": False,
    "num_workers": 8,
    "pin_memory": True,
    "seed": 45,

    # split
    "train_loso": True,
    "folds": 5,
    "split_ratios": [0.6, 0.2, 0.2],
    "stratify": True,
    "split_seed": 45,

    # training
    "model_name": "DeepTokenEEG",
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,

    # model
    "d_model": 32,
    "dropout": 0.5,
    "num_class": 2,
    "tokenizer_method": "conv",
    "tokenizer_kernel_size": 7,
    "resnet_n_blocks": 5,
    "resnet_dilations": [2, 2, 2, 2, 2],

    # eval
    "subject_agg": "mean_prob",
    "threshold": 0.5,
    "metrics": ["acc", "f1", "auc"],
    "save_best_on": "subject_f1",
}


# ---------------------------
# IO
# ---------------------------
def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping (dict). Got: {type(data)}")
    return data


def _write_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _cli_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    """Lấy các key CLI có giá trị != None để override lên config."""
    d = vars(ns)
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "config":
            continue
        if v is None:
            continue
        out[k] = v
    return out


def _merge_config(file_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    merged.update(file_cfg or {})
    merged.update(overrides or {})
    return merged


# ---------------------------
# Argparse helpers (bool tri-state)
# ---------------------------
def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default_none: bool = True, help_txt: str = "") -> None:
    """
    Tạo cặp flag: --name / --no_name.
    Mặc định = None để không override config nếu user không truyền.
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_txt)
    group.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable {help_txt}".strip())
    if default_none:
        parser.set_defaults(**{name: None})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_config")

    p.add_argument("--config", type=str, default=None, help="Path to unified YAML config.")

    # dataset
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--scope", type=str, default=None, choices=["single", "multidataset"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--band_set", nargs="+", default=None, help="e.g. --band_set alpha beta gamma")

    # output
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    _add_bool_flag(p, "save_resolved_config", help_txt="Save resolved config to outdir")

    # device
    p.add_argument("--device", type=str, default=None)
    _add_bool_flag(p, "amp", help_txt="Use AMP mixed precision")
    _add_bool_flag(p, "cudnn_benchmark", help_txt="Enable cudnn benchmark")
    _add_bool_flag(p, "cuda_deterministic", help_txt="Enable deterministic CUDA")
    p.add_argument("--num_workers", type=int, default=None)
    _add_bool_flag(p, "pin_memory", help_txt="Pin memory for DataLoader")
    p.add_argument("--seed", type=int, default=None)

    # split
    _add_bool_flag(p, "train_loso", help_txt="Use subject-wise 5-fold (60/20/20 rotate)")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--split_ratios", nargs=3, type=float, default=None)
    _add_bool_flag(p, "stratify", help_txt="Stratify split by subject label (if supported)")
    p.add_argument("--split_seed", type=int, default=None)

    # training
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw", "sgd"])
    p.add_argument("--scheduler", type=str, default=None, choices=["none", "cosine", "step"])
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--grad_clip_norm", type=float, default=None)

    # model
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--num_class", type=int, default=None)
    p.add_argument("--tokenizer_method", type=str, default=None, choices=["conv", "linear"])
    p.add_argument("--tokenizer_kernel_size", type=int, default=None)
    p.add_argument("--resnet_n_blocks", type=int, default=None)
    p.add_argument("--resnet_dilations", nargs="+", type=int, default=None)

    # eval
    p.add_argument("--subject_agg", type=str, default=None, choices=["mean_prob", "vote"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--save_best_on", type=str, default=None)

    # --- build_parser add ---
    _add_bool_flag(p, "early_stop", help_txt="Enable early stopping")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min_delta", type=float, default=None)

    p.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)


    return p


# ---------------------------
# Public API
# ---------------------------
def load_config(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Rule:
      - load DEFAULTS
      - load YAML (nếu có)
      - CLI overrides (nếu có) -> ghi đè YAML
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    file_cfg: Dict[str, Any] = {}
    if args.config:
        file_cfg = _read_yaml(args.config)

    overrides = _cli_overrides(args)
    cfg = _merge_config(file_cfg, overrides)

    validate_config(cfg)

    # dump resolved config nếu cần
    if cfg.get("save_resolved_config", True):
        resolved_path = os.path.join(cfg["outdir"], cfg["run_name"], "resolved_config.yaml")
        _write_yaml(resolved_path, cfg)

    return cfg


# ---------------------------
# Dataset path resolver
# ---------------------------
def resolve_band_paths(cfg: Dict[str, Any], band_name: str) -> Dict[str, str]:
    """
    Trả về đường dẫn chuẩn tới npz/json theo band, dựa theo cấu trúc:
      data_root/
        single/{dataset}/{band}/{band}.npz
        multidataset/{band}/{band}.npz
    """
    data_root = cfg["data_root"]
    scope = cfg["scope"]

    if scope == "single":
        dataset = cfg["dataset"]
        if not dataset:
            raise ValueError("cfg.dataset is required when scope='single'")
        band_dir = os.path.join(data_root, "single", dataset, band_name)
        meta_json = os.path.join(data_root, "single", dataset, f"{dataset}_meta.json")
    else:
        band_dir = os.path.join(data_root, "multidataset", band_name)
        meta_json = os.path.join(data_root, "multidataset", "multidataset_meta.json")

    npz_path = os.path.join(band_dir, f"{band_name}.npz")
    sidecar_json = os.path.join(band_dir, f"{band_name}.json")

    return {
        "band_dir": band_dir,
        "npz": npz_path,
        "json": sidecar_json,
        "meta": meta_json,
    }


def validate_config(cfg: Dict[str, Any]) -> None:
    if not cfg.get("data_root"):
        raise ValueError("cfg.data_root is empty. Set --data_root or data_root in YAML.")

    if cfg["scope"] not in ("single", "multidataset"):
        raise ValueError("cfg.scope must be 'single' or 'multidataset'.")

    if cfg["scope"] == "single" and not cfg.get("dataset"):
        raise ValueError("cfg.dataset is required for scope='single'.")

    ratios = cfg.get("split_ratios", [0.6, 0.2, 0.2])
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("cfg.split_ratios must have 3 values summing to 1.0, e.g. [0.6,0.2,0.2].")

    bands = cfg.get("band_set") or []
    if not isinstance(bands, list) or len(bands) == 0:
        raise ValueError("cfg.band_set must be a non-empty list, e.g. ['alpha'].")

    # outdir/run_name
    if not cfg.get("outdir") or not cfg.get("run_name"):
        raise ValueError("cfg.outdir and cfg.run_name are required.")

    if not cfg.get("model_name"):
        raise ValueError("cfg.model_name is required.")

    if cfg["model_name"] == "DeepTokenEEG":
        required = [
            "d_model",
            "dropout",
            "num_class",
            "tokenizer_method",
            "tokenizer_kernel_size",
            "resnet_n_blocks",
            "resnet_dilations",
        ]
        missing = [k for k in required if cfg.get(k) is None]
        if missing:
            raise ValueError(f"Missing DeepTokenEEG params: {missing}")
        

def get_model_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg["model_name"] == "DeepTokenEEG":
        return {
            "d_model": cfg["d_model"],
            "dropout": cfg["dropout"],
            "num_class": cfg["num_class"],
            "tokenizer_method": cfg["tokenizer_method"],
            "tokenizer_kernel_size": cfg["tokenizer_kernel_size"],
            "resnet_n_blocks": cfg["resnet_n_blocks"],
            "resnet_dilations": cfg["resnet_dilations"],
        }
    return {}
