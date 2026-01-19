import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.seed import seed_everything
from src.utils.io import load_json, load_npz, ensure_dir, save_json
from src.data.dataset import make_loaders
from src.models.model import Model
from src.train.trainer import train_model

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pick_device(cfg):
    if cfg["train"]["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["train"]["device"])

def main(cfg_data="configs/data.yaml", cfg_model="configs/model.yaml",
         cfg_train="configs/train.yaml", cfg_exp="configs/experiment.yaml"):
    data = load_yaml(cfg_data)
    model_cfg = load_yaml(cfg_model)["model"]
    train_cfg = load_yaml(cfg_train)["train"]
    exp_cfg = load_yaml(cfg_exp)["run"]

    seed = data["split"]["seed"]
    seed_everything(seed)

    cache_dir = data["paths"]["cache_dir"]
    band_name = data["signal"]["band_name"]
    cache_npz = os.path.join(cache_dir, f"dataset_{band_name}.npz")
    meta_path = os.path.join(cache_dir, f"dataset_{band_name}_meta.json")

    cache = load_npz(cache_npz)
    meta = load_json(meta_path)

    # to dict for dataset.py
    cache_dict = {k: cache[k] for k in cache.files}

    batch_size = train_cfg["batch_size"]
    train_loader, val_loader, _test_loader = make_loaders(cache_dict, batch_size=batch_size)

    device = pick_device({"train": train_cfg})
    print("Device:", device)

    enc_in = len(meta["channels_19"])
    d_model = int(model_cfg["d_model"])
    dropout = float(model_cfg["dropout"])
    num_class = int(model_cfg["num_class"])
    n_blocks = int(model_cfg["resnet"]["n_blocks"])
    dilations = model_cfg["resnet"]["dilations"]

    run_dir = os.path.join(exp_cfg["out_dir"], exp_cfg["name"], f"blocks_{n_blocks}")
    ensure_dir(run_dir)
    best_path = os.path.join(run_dir, "best.pth")
    save_json(os.path.join(run_dir, "config_snapshot.json"), {
        "data": data, "model": model_cfg, "train": train_cfg, "meta": meta
    })

    model = Model(
        enc_in=enc_in, num_class=num_class, d_model=d_model, dropout=dropout,
        n_blocks=n_blocks, dilations=dilations
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=train_cfg["lr_factor"], patience=train_cfg["lr_patience"]
    )

    model, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device=device, epochs=train_cfg["epochs"], patience=train_cfg["patience"],
        best_model_path=best_path
    )

    save_json(os.path.join(run_dir, "train_summary.json"), {"best_val_loss": best_val_loss})
    print("Saved best:", best_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_data", default="configs/data.yaml")
    ap.add_argument("--cfg_model", default="configs/model.yaml")
    ap.add_argument("--cfg_train", default="configs/train.yaml")
    ap.add_argument("--cfg_exp", default="configs/experiment.yaml")
    args = ap.parse_args()
    main(args.cfg_data, args.cfg_model, args.cfg_train, args.cfg_exp)
