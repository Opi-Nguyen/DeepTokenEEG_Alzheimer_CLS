import os
import yaml
import numpy as np
import torch

from src.utils.io import load_json, load_npz, ensure_dir
from src.models.model import Model
from src.data.dataset import make_loaders
from src.viz.feature_plots import compare_tokenizer_and_preclf
from src.viz.stage_plots import plot_segment_heatmap
from src.utils.plotting import plot_timeseries_compare

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pick_device(device_cfg):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)

def main(cfg_data="configs/data.yaml", cfg_model="configs/model.yaml",
         cfg_train="configs/train.yaml", cfg_exp="configs/experiment.yaml"):
    data = load_yaml(cfg_data)
    model_cfg = load_yaml(cfg_model)["model"]
    train_cfg = load_yaml(cfg_train)["train"]
    exp_run = load_yaml(cfg_exp)["run"]

    cache_dir = data["paths"]["cache_dir"]
    band_name = data["signal"]["band_name"]
    cache_npz = os.path.join(cache_dir, f"dataset_{band_name}.npz")
    meta_path = os.path.join(cache_dir, f"dataset_{band_name}_meta.json")

    cache = load_npz(cache_npz)
    meta = load_json(meta_path)
    cache_dict = {k: cache[k] for k in cache.files}

    train_loader, val_loader, test_loader = make_loaders(cache_dict, batch_size=train_cfg["batch_size"])
    device = pick_device(train_cfg["device"])
    print("Device:", device)

    enc_in = len(meta["channels_19"])
    d_model = int(model_cfg["d_model"])
    dropout = float(model_cfg["dropout"])
    num_class = int(model_cfg["num_class"])
    n_blocks = int(model_cfg["resnet"]["n_blocks"])
    dilations = model_cfg["resnet"]["dilations"]

    run_dir = os.path.join(exp_run["out_dir"], exp_run["name"], f"blocks_{n_blocks}_dilations_{'_'.join(map(str, dilations))}")
    ckpt = os.path.join(run_dir, "best.pth")

    fig_dir = os.path.join(run_dir, "figures")
    ensure_dir(fig_dir)

    model = Model(
        enc_in=enc_in, num_class=num_class, d_model=d_model, dropout=dropout,
        n_blocks=n_blocks, dilations=dilations
    ).to(device)

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Loaded:", ckpt)
    else:
        print("WARNING: checkpoint not found, visualizing with random weights.")

    # 1) Feature visualization: AD vs HC after tokenizer/backbone and pre-classifier vector
    compare_tokenizer_and_preclf(model, test_loader, device, fig_dir, tag=f"blocks{n_blocks}")

    # 2) Basic segment heatmap example (from cache) to reflect preprocessing output
    # choose one segment from train set
    X_train = cache_dict["X_train"]
    if len(X_train) > 0:
        seg = X_train[0]  # [T,C]
        plot_segment_heatmap(seg, fig_dir, tag="segment_example")

        # plot few channels compare within the segment (optional)
        ch0 = seg[:, 0]
        ch1 = seg[:, 1]
        plot_timeseries_compare([ch0, ch1], ["Ch0", "Ch1"],
                                "Segment channels compare", f"{fig_dir}/segment_ch_compare.png",
                                max_points=512)

    print("Figures saved in:", fig_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_data", default="configs/data.yaml")
    ap.add_argument("--cfg_model", default="configs/model.yaml")
    ap.add_argument("--cfg_train", default="configs/train.yaml")
    ap.add_argument("--cfg_exp", default="configs/experiment.yaml")
    args = ap.parse_args()
    main(args.cfg_data, args.cfg_model, args.cfg_train, args.cfg_exp)
