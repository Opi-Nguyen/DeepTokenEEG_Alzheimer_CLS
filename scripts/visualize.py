# scripts/visualize.py
# -----------------------------------------------------------------------------
# Visualize a single run by snapshot OR batch-visualize many runs by scanning
# a folder recursively for config_snapshot.json.
#
# Usage (single run):
#   python scripts/visualize.py --snapshot /path/to/run/config_snapshot.json
#
# Usage (batch):
#   python scripts/visualize.py --scan_root /path/to/outputs/runs
#
# Notes:
# - We assume: the folder containing config_snapshot.json is the run_dir.
# - We try to load checkpoint from run_dir/best.pth (fallback to common names).
# - Cache NPZ is loaded using snapshot["data"]["paths"]["cache_dir"] (relative to base_dir if needed).
# - meta is embedded in snapshot["meta"] (so no need to load *_meta.json).
#
# What this script generates (more samples for nicer figure selection):
# - Multiple segment heatmaps: K samples per class (HC/AD), multiple sets (n_sets)
# - Multiple timeseries compares per segment (several channels)
# - Multiple feature plots by calling compare_tokenizer_and_preclf on sampled subsets
# -----------------------------------------------------------------------------

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.io import load_npz, ensure_dir
from src.models.model import Model
from data.old.dataset_v1 import make_loaders

from src.viz.feature_plots import compare_tokenizer_and_preclf
from src.viz.stage_plots import plot_segment_heatmap
from src.utils.plotting import plot_timeseries_compare


# ---------------------------
# IO helpers
# ---------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: str, maybe_rel_path: str) -> str:
    if os.path.isabs(maybe_rel_path):
        return maybe_rel_path
    return os.path.normpath(os.path.join(base_dir, maybe_rel_path))


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def safe_float(x, default=None) -> float:
    if x is None:
        if default is None:
            raise ValueError("Expected a numeric value, got None.")
        return float(default)
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        return float(s)
    raise TypeError(f"Cannot convert type {type(x)} to float")


def find_checkpoint(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "best.pth"),
        os.path.join(run_dir, "best_model.pth"),
        os.path.join(run_dir, "best_model_new.pth"),
        os.path.join(run_dir, "checkpoint.pth"),
        os.path.join(run_dir, "model.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def find_snapshots(scan_root: str, pattern: str = "config_snapshot.json") -> List[str]:
    scan_root = os.path.abspath(scan_root)
    hits = glob.glob(os.path.join(scan_root, "**", pattern), recursive=True)
    hits = [os.path.abspath(p) for p in hits]
    hits.sort()
    return hits


# ---------------------------
# Sampling helpers
# ---------------------------
def _to_numpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x)


def pick_indices_per_class(
    y: np.ndarray,
    n_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = _to_numpy(y).astype(int)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n0 = min(n_per_class, len(idx0))
    n1 = min(n_per_class, len(idx1))

    pick0 = rng.choice(idx0, size=n0, replace=False)
    pick1 = rng.choice(idx1, size=n1, replace=False)

    return np.sort(pick0), np.sort(pick1)


def subset_loader_from_cache(
    X: np.ndarray, y: np.ndarray, s: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = False
) -> DataLoader:
    Xs = X[indices]
    ys = y[indices]
    ss = s[indices]

    # Ensure torch tensors are float32/long
    X_t = torch.from_numpy(Xs).float()
    y_t = torch.from_numpy(ys).long()
    s_t = torch.from_numpy(ss).long()
    ds = TensorDataset(X_t, y_t, s_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def flatten_channel_groups(channels_cfg) -> List[str]:
    """
    Your snapshot sample shows channels.standard_19 stored as 2 long strings.
    This function tries to normalize them into a list of channel names if needed.
    """
    if channels_cfg is None:
        return []
    if isinstance(channels_cfg, list):
        # If entries look like "Fp1 - Fp2 - ...", split them
        out = []
        for item in channels_cfg:
            if isinstance(item, str) and " - " in item:
                out.extend([x.strip() for x in item.split("-") if x.strip()])
            elif isinstance(item, str):
                out.append(item.strip())
        # de-dup keep order
        seen = set()
        uniq = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq
    return []


# ---------------------------
# Core per-run visualize
# ---------------------------
def visualize_one_snapshot(snapshot_path: str, args) -> Dict[str, Any]:
    run_dir = os.path.dirname(os.path.abspath(snapshot_path))
    snap = load_json(snapshot_path)

    data_cfg = snap["data"]
    model_cfg = snap["model"]
    train_cfg = snap["train"]
    meta = snap.get("meta", {})

    base_dir = data_cfg["paths"]["base_dir"]
    cache_dir = resolve_path(base_dir, data_cfg["paths"]["cache_dir"])
    band_name = data_cfg["signal"]["band_name"]

    print(snapshot_path)
    cache_npz = os.path.join(cache_dir, f"dataset_{band_name}.npz")
    if not os.path.exists(cache_npz):
        return {
            "snapshot": snapshot_path,
            "run_dir": run_dir,
            "status": "skip",
            "reason": f"cache not found: {cache_npz}",
        }

    cache = load_npz(cache_npz)
    cache_dict = {k: cache[k] for k in cache.files}

    # loaders (full) – useful if you want entire test set
    full_train_loader, full_val_loader, full_test_loader = make_loaders(
        cache_dict, batch_size=int(train_cfg["batch_size"])
    )

    device_cfg = args.device if args.device is not None else train_cfg.get("device", "auto")
    device = pick_device(device_cfg)
    print(f"\n[RUN] {run_dir}")
    print("  Device:", device)

    # Model params
    channels_19 = meta.get("channels_19", [])
    if not channels_19:
        # try from data.channels.standard_19 (may be weird formatted)
        channels_19 = flatten_channel_groups(data_cfg.get("channels", {}).get("standard_19"))
    if not channels_19:
        # fallback infer from cache shape
        any_key = "X_test" if "X_test" in cache_dict else "X_train"
        channels_19 = list(range(int(cache_dict[any_key].shape[-1])))
        print("  [WARN] channels_19 missing; inferred enc_in from cache shape.")

    enc_in = len(channels_19)
    d_model = int(model_cfg["d_model"])
    dropout = safe_float(model_cfg.get("dropout", 0.0), default=0.0)
    num_class = int(model_cfg["num_class"])

    tok_cfg = model_cfg.get("tokenizer", {})
    tokenizer_method = tok_cfg.get("method", "conv")
    tokenizer_kernel = int(tok_cfg.get("kernel_size", 7))

    res_cfg = model_cfg.get("resnet", {})
    n_blocks = int(res_cfg.get("n_blocks", 3))
    dilations = res_cfg.get("dilations", [2] * n_blocks)

    # Prepare dirs
    fig_dir = os.path.join(run_dir, "figures")
    ensure_dir(fig_dir)

    # Build model
    model = Model(
        enc_in=enc_in,
        num_class=num_class,
        d_model=d_model,
        dropout=dropout,
        n_blocks=n_blocks,
        dilations=dilations,
        tokenizer_method=tokenizer_method,
        tokenizer_kernel_size=tokenizer_kernel,
    ).to(device)

    ckpt = find_checkpoint(run_dir)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("  Loaded checkpoint:", ckpt)
    else:
        print("  [WARN] checkpoint not found; feature plots will reflect random weights.")

    # ---------------------------------
    # (A) Multiple feature visualizations
    # ---------------------------------
    # We’ll sample multiple subsets from test set to produce multiple candidate figures.
    X_test = cache_dict.get("X_test")
    y_test = cache_dict.get("y_test")
    s_test = cache_dict.get("s_test")

    if X_test is not None and y_test is not None and s_test is not None and len(X_test) > 0:
        for set_i in range(args.n_sets):
            seed_i = args.seed + set_i
            idx0, idx1 = pick_indices_per_class(y_test, args.samples_per_class, seed_i)

            if len(idx0) == 0 or len(idx1) == 0:
                print("  [WARN] cannot sample both classes from test set.")
                break

            idx = np.concatenate([idx0, idx1])
            sub_loader = subset_loader_from_cache(
                X_test, y_test, s_test,
                indices=idx,
                batch_size=min(int(train_cfg["batch_size"]), len(idx)),
                shuffle=False
            )

            tag = f"blocks{n_blocks}_set{set_i}_n{args.samples_per_class}perclass"
            compare_tokenizer_and_preclf(model, sub_loader, device, fig_dir, tag=tag)

    # ---------------------------------
    # (B) Multiple segment heatmaps (AD vs HC)
    # ---------------------------------
    # Create many heatmaps to choose the “nicest looking” ones
    if X_test is not None and y_test is not None and len(X_test) > 0:
        for set_i in range(args.n_sets):
            seed_i = args.seed + 1000 + set_i
            idx0, idx1 = pick_indices_per_class(y_test, args.heatmaps_per_class, seed_i)
            # heatmaps for HC
            for j, ix in enumerate(idx0.tolist()):
                seg = X_test[ix]  # [T, C]
                plot_segment_heatmap(seg, fig_dir, tag=f"HC_set{set_i}_idx{ix}_h{j}")
                # also draw a few channel timeseries comparisons for that segment
                _plot_timeseries_bundle(seg, fig_dir, prefix=f"HC_set{set_i}_idx{ix}_h{j}", max_points=args.max_points)

            # heatmaps for AD
            for j, ix in enumerate(idx1.tolist()):
                seg = X_test[ix]
                plot_segment_heatmap(seg, fig_dir, tag=f"AD_set{set_i}_idx{ix}_h{j}")
                _plot_timeseries_bundle(seg, fig_dir, prefix=f"AD_set{set_i}_idx{ix}_h{j}", max_points=args.max_points)

    print("  Figures saved in:", fig_dir)

    return {
        "snapshot": snapshot_path,
        "run_dir": run_dir,
        "status": "ok",
        "fig_dir": fig_dir,
        "checkpoint": ckpt if ckpt else None,
    }


def _plot_timeseries_bundle(seg_tc: np.ndarray, fig_dir: str, prefix: str, max_points: int = 512):
    """
    seg_tc: [T, C] float
    Produce multiple channel comparisons to help pick nicer figures.
    """
    if seg_tc is None or seg_tc.ndim != 2:
        return
    T, C = seg_tc.shape
    if T == 0 or C == 0:
        return

    max_points = min(max_points, T)

    # Define a small set of channel indices to compare
    # (0,1), (2,3), (0,2), (0, C//2) if possible
    pairs = []
    if C >= 2:
        pairs.append((0, 1))
    if C >= 4:
        pairs.append((2, 3))
    if C >= 3:
        pairs.append((0, 2))
    if C >= 2:
        pairs.append((0, C // 2))

    # Ensure unique pairs
    uniq = []
    seen = set()
    for a, b in pairs:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen and a < C and b < C:
            seen.add(key)
            uniq.append((a, b))

    for k, (a, b) in enumerate(uniq):
        ch_a = seg_tc[:max_points, a]
        ch_b = seg_tc[:max_points, b]
        out_path = os.path.join(fig_dir, f"{prefix}_ch{a}_vs_ch{b}_{k}.png")
        plot_timeseries_compare(
            [ch_a, ch_b],
            [f"Ch{a}", f"Ch{b}"],
            f"{prefix} - channel compare",
            out_path,
            max_points=max_points,
        )


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", help="Path to a single config_snapshot.json")
    group.add_argument("--scan_root", help="Scan this folder recursively for config_snapshot.json")

    ap.add_argument("--pattern", default="config_snapshot.json", help="Snapshot filename pattern to scan for")
    ap.add_argument("--max_runs", type=int, default=999999, help="Limit number of runs when scanning")

    ap.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu, auto). If None, use snapshot train.device")

    # More samples = more candidate figures
    ap.add_argument("--n_sets", type=int, default=3, help="Number of different random sample sets per run")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed")

    ap.add_argument("--samples_per_class", type=int, default=64,
                    help="How many test segments per class to include in feature plots (per set)")
    ap.add_argument("--heatmaps_per_class", type=int, default=3,
                    help="How many segments per class to draw heatmaps for (per set)")
    ap.add_argument("--max_points", type=int, default=512,
                    help="Max points for channel timeseries compare plots")

    args = ap.parse_args()

    snapshots: List[str]
    if args.snapshot:
        snapshots = [os.path.abspath(args.snapshot)]
    else:
        snapshots = find_snapshots(args.scan_root, pattern=args.pattern)
        snapshots = snapshots[: args.max_runs]

    if not snapshots:
        print("[ERROR] No snapshots found.")
        return

    results = []
    for sp in snapshots:
        try:
            res = visualize_one_snapshot(sp, args)
        except Exception as e:
            res = {
                "snapshot": sp,
                "run_dir": os.path.dirname(os.path.abspath(sp)),
                "status": "error",
                "reason": repr(e),
            }
        results.append(res)

    # Save a small summary near scan_root or near the single snapshot
    if args.scan_root:
        out_summary_dir = os.path.abspath(args.scan_root)
    else:
        out_summary_dir = os.path.dirname(os.path.abspath(args.snapshot))

    summary_path = os.path.join(out_summary_dir, "visualize_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. OK={ok}/{len(results)}")
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
