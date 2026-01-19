import os
import yaml
import torch

from src.utils.io import load_json, load_npz, save_json, ensure_dir
from src.data.dataset import make_loaders
from src.models.model import Model
from src.train.thresholds import find_best_segment_threshold, find_best_subject_threshold
from src.train.evaluator import evaluate_with_thresholds

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pick_device(device_cfg):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)

def main(cfg_data="configs/data.yaml", cfg_model="configs/model.yaml",
         cfg_train="configs/train.yaml", cfg_exp="configs/experiment.yaml",
         checkpoint_path=None):
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


    ablation = train_cfg.get("ablation", False)
    if ablation:
        ablation_name = f"ablation_{n_blocks}_blocks"
    else:
        ablation_name = ""
        
    run_dir = os.path.join(exp_run["out_dir"], exp_run["name"], ablation_name, f"blocks_{n_blocks}_dilations_{'_'.join(map(str, dilations))}")
    ensure_dir(run_dir)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_dir, "best.pth")

    model = Model(
        enc_in=enc_in, num_class=num_class, d_model=d_model, dropout=dropout,
        n_blocks=n_blocks, dilations=dilations
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    seg_thr, seg_f1 = find_best_segment_threshold(model, val_loader, device)
    subj_thr, subj_f1 = find_best_subject_threshold(model, val_loader, device)

    report = evaluate_with_thresholds(
        model, test_loader, device,
        seg_thr=seg_thr, subj_thr=subj_thr,
        n_bootstraps=1000, seed=data["split"]["seed"]
    )

    report["threshold_search"] = {
        "segment_best_f1_val": seg_f1,
        "subject_best_f1_val": subj_f1
    }
    report["checkpoint"] = checkpoint_path

    out_path = os.path.join(run_dir, "eval_report.json")
    save_json(out_path, report)
    print("Saved:", out_path)
    print(report)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_data", default="configs/data.yaml")
    ap.add_argument("--cfg_model", default="configs/model.yaml")
    ap.add_argument("--cfg_train", default="configs/train.yaml")
    ap.add_argument("--cfg_exp", default="configs/experiment.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    main(args.cfg_data, args.cfg_model, args.cfg_train, args.cfg_exp, checkpoint_path=args.checkpoint)
