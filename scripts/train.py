## script/train_cv.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.config_loader import load_config, resolve_band_paths
from src.data.npz_dataloader import load_band_npz, EEGSegmentDataset, make_loader
from src.data.subject_cv import make_subject_folds, split_fold_rotate_60202, indices_for_subjects
from src.train.metrics import compute_metrics, confusion_2x2
from src.train.trainer import Trainer
from src.models.build_model import build_model

# ------------------
# IO helpers
# ------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_npz(path: str, **kwargs) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)


def standardize_X_to_NTC(X: np.ndarray, max_channels: int = 64) -> tuple[np.ndarray, int]:
    """
    Ensure X is [N, T, C]. Return (X_ntc, enc_in=C).
    Heuristic: channel dimension usually <= max_channels, time dimension usually larger.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim==3, got shape={X.shape}")

    a, b = X.shape[1], X.shape[2]  # could be (T,C) or (C,T)

    # case 1: already [N, T, C]
    if b <= max_channels and a > b:
        return X, int(b)

    # case 2: [N, C, T] -> transpose to [N, T, C]
    if a <= max_channels and b > a:
        X = np.transpose(X, (0, 2, 1))
        return X, int(a)

    # fallback: assume last dim is channels
    return X, int(b)


# ------------------
# Visualize
# ------------------
def plot_loss_curve(history: List[Dict[str, Any]], out_path: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]

    plt.figure()
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, va, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_confusion(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str) -> None:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    # simple ROC points
    thr = np.unique(y_prob)[::-1]
    tpr, fpr = [], []
    P = max(int((y_true == 1).sum()), 1)
    N = max(int((y_true == 0).sum()), 1)
    for t in thr:
        y_pred = (y_prob >= t).astype(np.int64)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tpr.append(tp / P)
        fpr.append(fp / N)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ------------------
# Eval
# ------------------
@torch.no_grad()
def predict_probs(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      y_true (N,)
      y_prob_pos (N,)  probability of class 1
      sid (N,)  subject id per segment
    """
    model.eval()
    ys, ps, sids = [], [], []

    for batch in loader:
        x, y, sid = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        ys.append(y.detach().cpu().numpy())
        ps.append(prob)
        sids.extend(list(sid))

    y_true = np.concatenate(ys, axis=0).astype(np.int64) if ys else np.array([], dtype=np.int64)
    y_prob = np.concatenate(ps, axis=0).astype(float) if ps else np.array([], dtype=float)
    sid_arr = np.asarray(sids)
    return y_true, y_prob, sid_arr


def aggregate_subject(y_true_seg: np.ndarray, y_prob_seg: np.ndarray, sid: np.ndarray, agg: str, threshold: float):
    """
    Produce subject-level (y_true_sub, y_prob_sub)
    """
    sid = np.asarray(sid)
    y_true_seg = np.asarray(y_true_seg).astype(np.int64)
    y_prob_seg = np.asarray(y_prob_seg, dtype=float)

    subj_ids = np.unique(sid)
    y_true_sub, y_prob_sub = [], []

    for s in subj_ids:
        idx = np.where(sid == s)[0]
        # subject label: majority from segments
        lbs = y_true_seg[idx]
        c0 = int((lbs == 0).sum())
        c1 = int((lbs == 1).sum())
        ysub = 1 if c1 >= c0 else 0

        probs = y_prob_seg[idx]
        if agg == "vote":
            yhat = (probs >= threshold).astype(np.int64)
            psub = float(yhat.mean())  # vote-rate
        else:
            psub = float(probs.mean())

        y_true_sub.append(ysub)
        y_prob_sub.append(psub)

    return np.asarray(y_true_sub, dtype=np.int64), np.asarray(y_prob_sub, dtype=float), subj_ids



def make_optimizer(cfg: Dict[str, Any], model: nn.Module):
    opt = cfg.get("optimizer", "adamw")
    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt}")


def make_scheduler(cfg: Dict[str, Any], optimizer):
    sch = cfg.get("scheduler", "cosine")
    if sch == "none":
        return None
    if sch == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    if sch == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(int(cfg["epochs"]) // 3, 1), gamma=0.1)
    return None


def main():
    cfg = load_config()

    device = torch.device(cfg["device"] if cfg["device"] != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = bool(cfg.get("cuda_deterministic", False))

    # root_run = os.path.join(cfg["outdir"], cfg["model_name"], cfg["run_name"])
    if cfg["scope"] == "single":
        root_run = os.path.join(cfg["outdir"], cfg["model_name"], cfg["dataset"])
    else:
        root_run = os.path.join(cfg["outdir"], cfg["model_name"], "multi_dataset")
    
    ensure_dir(root_run)

    # wandb (optional)
    wandb_run = None
    if cfg.get("wandb_mode", "online") != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "DeepTokenEEG"),
                entity=cfg.get("wandb_entity", None),
                name=f"{cfg['run_name']}",
                config=cfg,
                dir=root_run,
                mode=cfg.get("wandb_mode", "online"),
            )
        except Exception as e:
            print(f"[WARN] wandb init failed -> disabled. err={e}")
            wandb_run = None

    all_band_summaries = {}

    for band in cfg["band_set"]:
        band_paths = resolve_band_paths(cfg, band)
        cache = load_band_npz(band_paths["npz"], band)

        X, y, sid = cache.X, cache.y, cache.sid
        enc_in = int(X.shape[1])
        seq_len = int(X.shape[2])
        print("DEBUG X:", X.shape, "enc_in:", enc_in, "seq_len:", seq_len)

        enc_in = int(X.shape[-1]) if X.ndim == 3 else 19

        # subject folds
        folds = make_subject_folds(
            y=y,
            sid=sid,
            n_folds=int(cfg["folds"]),
            seed=int(cfg["split_seed"]),
            stratify=bool(cfg.get("stratify", True)),
        )

        band_dir = os.path.join(root_run, band)
        ensure_dir(band_dir)

        fold_metrics_seg = []
        fold_metrics_sub = []

        for fold_idx in range(int(cfg["folds"])):
            train_sub, val_sub, test_sub = split_fold_rotate_60202(folds, fold_idx)

            train_idx = indices_for_subjects(sid, train_sub)
            val_idx = indices_for_subjects(sid, val_sub)
            test_idx = indices_for_subjects(sid, test_sub)

            fold_dir = os.path.join(band_dir, f"fold_{fold_idx}")
            ensure_dir(fold_dir)

            # datasets/loaders
            train_ds = EEGSegmentDataset(X, y, sid, train_idx)
            val_ds = EEGSegmentDataset(X, y, sid, val_idx)
            test_ds = EEGSegmentDataset(X, y, sid, test_idx)

            train_loader = make_loader(
                train_ds,
                batch_size=int(cfg["batch_size"]),
                shuffle=True,
                num_workers=int(cfg["num_workers"]),
                pin_memory=bool(cfg["pin_memory"]),
            )
            val_loader = make_loader(
                val_ds,
                batch_size=int(cfg["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["num_workers"]),
                pin_memory=bool(cfg["pin_memory"]),
            )
            test_loader = make_loader(
                test_ds,
                batch_size=int(cfg["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["num_workers"]),
                pin_memory=bool(cfg["pin_memory"]),
            )

            # model/trainer
            model = build_model(cfg, enc_in=enc_in, seq_len=seq_len).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = make_optimizer(cfg, model)
            scheduler = make_scheduler(cfg, optimizer)

            best_path = os.path.join(fold_dir, "best.pth")

            fold_wandb = None
            if wandb_run is not None:
                try:
                    import wandb
                    fold_wandb = wandb_run  # 1 run chung; log kèm fold tag
                    fold_wandb.log({"fold": fold_idx, "band": band})
                except Exception:
                    fold_wandb = None

            trainer = Trainer(
                model=model,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=bool(cfg.get("amp", True)),
                grad_clip_norm=float(cfg.get("grad_clip_norm", 0.0)),
                early_stop=True,  # always True as requested
                patience=int(cfg.get("patience", 12)),
                min_delta=float(cfg.get("min_delta", 0.0)),
                best_path=best_path,
                wandb_run=fold_wandb,
            )

            fit_info = trainer.fit(train_loader, val_loader, epochs=int(cfg["epochs"]))

            # save history + curves
            save_json(os.path.join(fold_dir, "history.json"), fit_info)
            plot_loss_curve(fit_info["history"], os.path.join(fold_dir, "loss_curve.png"))

            # evaluate on test (segment-level)
            y_true_seg, y_prob_seg, sid_seg = predict_probs(model, test_loader, device=device)
            metrics_seg = compute_metrics(
                y_true=y_true_seg,
                y_prob=y_prob_seg,
                threshold=float(cfg["threshold"]),
                metrics=list(cfg["metrics"]),
            )

            # subject-level
            y_true_sub, y_prob_sub, subj_ids = aggregate_subject(
                y_true_seg, y_prob_seg, sid_seg,
                agg=str(cfg.get("subject_agg", "mean_prob")),
                threshold=float(cfg["threshold"]),
            )
            metrics_sub = compute_metrics(
                y_true=y_true_sub,
                y_prob=y_prob_sub,
                threshold=float(cfg["threshold"]),
                metrics=list(cfg["metrics"]),
            )

            # visualize confusion / roc
            cm_seg = confusion_2x2(y_true_seg, (y_prob_seg >= float(cfg["threshold"])).astype(np.int64))
            cm_sub = confusion_2x2(y_true_sub, (y_prob_sub >= float(cfg["threshold"])).astype(np.int64))
            plot_confusion(cm_seg, os.path.join(fold_dir, "cm_segment.png"), "Confusion (segment)")
            plot_confusion(cm_sub, os.path.join(fold_dir, "cm_subject.png"), "Confusion (subject)")

            if "auc" in cfg["metrics"]:
                plot_roc(y_true_seg, y_prob_seg, os.path.join(fold_dir, "roc_segment.png"), "ROC (segment)")
                plot_roc(y_true_sub, y_prob_sub, os.path.join(fold_dir, "roc_subject.png"), "ROC (subject)")

            # save raw preds (optional but useful)
            save_npz(
                os.path.join(fold_dir, "test_preds.npz"),
                y_true_seg=y_true_seg,
                y_prob_seg=y_prob_seg,
                sid_seg=sid_seg,
                y_true_sub=y_true_sub,
                y_prob_sub=y_prob_sub,
                subj_ids=subj_ids,
            )

            fold_pack = {
                "band": band,
                "fold": int(fold_idx),
                "paths": {"npz": band_paths["npz"], "meta": band_paths["meta"]},
                "counts": {
                    "subjects_train": int(len(train_sub)),
                    "subjects_val": int(len(val_sub)),
                    "subjects_test": int(len(test_sub)),
                    "segments_train": int(train_idx.size),
                    "segments_val": int(val_idx.size),
                    "segments_test": int(test_idx.size),
                },
                "best": {"epoch": fit_info["best_epoch"], "val_loss": fit_info["best_val_loss"], "best_path": best_path},
                "metrics_segment": metrics_seg,
                "metrics_subject": metrics_sub,
            }
            save_json(os.path.join(fold_dir, "metrics.json"), fold_pack)

            fold_metrics_seg.append(metrics_seg)
            fold_metrics_sub.append(metrics_sub)

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {f"{band}/fold{fold_idx}/seg_{k}": v for k, v in metrics_seg.items()}
                        | {f"{band}/fold{fold_idx}/sub_{k}": v for k, v in metrics_sub.items()},
                        step=int(fit_info["best_epoch"]),
                    )
                except Exception:
                    pass

        # summary mean±std
        def summarize(m_list: List[Dict[str, float]]) -> Dict[str, Any]:
            out = {}
            for m in cfg["metrics"]:
                vals = np.array([d.get(m, float("nan")) for d in m_list], dtype=float)
                out[m] = {
                    "per_fold": [float(x) for x in vals.tolist()],
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                }
            return out

        band_summary = {
            "band": band,
            "folds": int(cfg["folds"]),
            "segment": summarize(fold_metrics_seg),
            "subject": summarize(fold_metrics_sub),
        }
        save_json(os.path.join(band_dir, "summary.json"), band_summary)
        all_band_summaries[band] = band_summary

    save_json(os.path.join(root_run, "all_bands_summary.json"), all_band_summaries)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
