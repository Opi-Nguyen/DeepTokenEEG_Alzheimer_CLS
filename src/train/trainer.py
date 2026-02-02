# train/src_train/trainer.py
import os
import time
import torch

import yaml  # PyYAML


def _unpack_batch(batch):
    """
    Support both:
      (x, y, sid)
      (x, y, sid, k)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            x, y, sid = batch
            return x, y, sid
        if len(batch) >= 4:
            return batch[0], batch[1], batch[2]
    raise ValueError(f"Unexpected batch format: type={type(batch)} len={len(batch) if isinstance(batch,(list,tuple)) else 'NA'}")


def _makedirs_for_file(path: str):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _get_lr(optimizer):
    try:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    except Exception:
        return None


def _dump_yaml(path: str, payload: dict):
    _makedirs_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
    epochs=50,
    patience=10,
    best_model_path="best.pth",
    log_yaml_path=None,            # NEW
    run_info: dict | None = None,  # NEW: nhét thêm config/meta nếu muốn
    write_yaml_every_epoch=True,   # NEW
):
    best_val_loss = float("inf")
    best_epoch = -1
    patience_ctr = 0

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    history = []

    _makedirs_for_file(best_model_path)
    if log_yaml_path:
        _makedirs_for_file(log_yaml_path)

    def write_yaml(final=False):
        if not log_yaml_path:
            return
        payload = {
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S") if final else None,
            "device": str(device),
            "epochs_planned": int(epochs),
            "patience": int(patience),
            "best_model_path": str(best_model_path),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
            "run_info": run_info or {},
            "history": history,
        }
        _dump_yaml(log_yaml_path, payload)

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            x, y, _sid = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = int(y.size(0))
            train_loss_sum += float(loss.item()) * bs
            n_train += bs

        train_loss = train_loss_sum / max(n_train, 1)

        # -------- val --------
        model.eval()
        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y, _sid = _unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)

                bs = int(y.size(0))
                val_loss_sum += float(loss.item()) * bs
                n_val += bs

        val_loss = val_loss_sum / max(n_val, 1)

        # scheduler
        if scheduler is not None:
            try:
                scheduler.step(val_loss)  # ReduceLROnPlateau
            except TypeError:
                scheduler.step()

        lr_now = _get_lr(optimizer)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_ctr += 1

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": lr_now,
            "improved": bool(improved),
            "patience_ctr": int(patience_ctr),
        })

        if log_yaml_path and write_yaml_every_epoch:
            write_yaml(final=False)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| lr={lr_now} | best={best_val_loss:.6f} @ {best_epoch}"
        )

        if patience_ctr >= patience:
            break

    # load best back
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    if log_yaml_path and (not write_yaml_every_epoch):
        write_yaml(final=True)
    elif log_yaml_path and write_yaml_every_epoch:
        write_yaml(final=True)

    return model, float(best_val_loss)
