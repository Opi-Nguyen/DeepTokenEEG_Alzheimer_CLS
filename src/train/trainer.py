# src/train/trainer_cv.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _get_lr(optim: torch.optim.Optimizer) -> float:
    try:
        return float(optim.param_groups[0].get("lr", 0.0))
    except Exception:
        return 0.0


class Trainer:
    """
    Trainer for ONE (train_loader, val_loader). Test handled outside.
    Always early stopping = True (as requested).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        amp: bool,
        grad_clip_norm: float,
        early_stop: bool,
        patience: int,
        min_delta: float,
        best_path: str,
        wandb_run: Optional[Any] = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = bool(amp) and (str(device).startswith("cuda"))
        self.scaler = GradScaler(enabled=self.amp)
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else 0.0

        self.early_stop = True if early_stop is None else bool(early_stop)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_path = best_path
        _ensure_dir(os.path.dirname(best_path))

        self.wandb_run = wandb_run
        self.history = []

        self.best_val = float("inf")
        self.best_epoch = -1

    def _step_loader(self, loader, train: bool) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum = 0.0
        n = 0

        for batch in loader:
            x, y, _sid = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if train:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            bs = int(y.size(0))
            loss_sum += float(loss.item()) * bs
            n += bs

        return loss_sum / max(n, 1)

    def fit(self, train_loader, val_loader, epochs: int) -> Dict[str, Any]:
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        patience_ctr = 0

        for epoch in range(1, int(epochs) + 1):
            train_loss = self._step_loader(train_loader, train=True)
            with torch.no_grad():
                val_loss = self._step_loader(val_loader, train=False)

            # scheduler
            if self.scheduler is not None:
                try:
                    # ReduceLROnPlateau
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            lr_now = _get_lr(self.optimizer)

            improved = (self.best_val - val_loss) > self.min_delta
            if improved:
                self.best_val = float(val_loss)
                self.best_epoch = int(epoch)
                patience_ctr = 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                patience_ctr += 1

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(lr_now),
                "improved": bool(improved),
                "patience_ctr": int(patience_ctr),
            }
            self.history.append(row)

            if self.wandb_run is not None:
                self.wandb_run.log(row, step=epoch)

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
                f"| lr={lr_now:.6g} | best={self.best_val:.6f} @ {self.best_epoch}"
            )

            if self.early_stop and patience_ctr >= self.patience:
                break

        # restore best
        if os.path.exists(self.best_path):
            self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))

        finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "started_at": started_at,
            "finished_at": finished_at,
            "best_val_loss": float(self.best_val),
            "best_epoch": int(self.best_epoch),
            "best_path": self.best_path,
            "history": self.history,
        }
