import copy
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, best_model_path=None):
    best_val_loss = float("inf")
    patience_counter = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, _sid in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _sid in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_wts = copy.deepcopy(model.state_dict())
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Val loss improved. Saving model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(">>> Early stopping!")
                break

    model.load_state_dict(best_wts)
    return model, float(best_val_loss)
