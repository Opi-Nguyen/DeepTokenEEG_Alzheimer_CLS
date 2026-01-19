import numpy as np
import torch
from src.utils.plotting import plot_heatmap, plot_vector_compare

def extract_one_batch_by_label(model, loader, device, target_label, max_batches=200):
    """
    lấy 1 batch có label target_label (ưu tiên batch có nhiều mẫu target_label).
    trả: inputs_subset (B', T, C), labels_subset (B',)
    """
    model.eval()
    best_x, best_y = None, None
    best_count = 0
    for i, (x, y, _sid) in enumerate(loader):
        mask = (y == target_label)
        cnt = int(mask.sum().item())
        if cnt > best_count:
            best_count = cnt
            best_x = x[mask]
            best_y = y[mask]
        if i >= max_batches or best_count >= 16:
            break
    if best_x is None or best_count == 0:
        return None, None
    return best_x.to(device), best_y.to(device)

@torch.no_grad()
def compare_tokenizer_and_preclf(model, loader, device, out_dir, tag="compare"):
    """
    Visualize:
    - tokenizer output map mean (d_model x T) for HC vs AD
    - pre-classifier vector mean (d_model,) for HC vs AD
    Yêu cầu: model có forward_backbone trả (feat, map) hoặc dùng hook.
    """
    x_hc, _ = extract_one_batch_by_label(model, loader, device, target_label=0)
    x_ad, _ = extract_one_batch_by_label(model, loader, device, target_label=1)

    if x_hc is None or x_ad is None:
        print("Not enough samples for HC/AD in loader to visualize.")
        return

    feat_hc, map_hc = model.forward_backbone(x_hc)   # feat: [B,d], map:[B,d,T]
    feat_ad, map_ad = model.forward_backbone(x_ad)

    mean_map_hc = map_hc.mean(dim=0).cpu().numpy()   # [d, T]
    mean_map_ad = map_ad.mean(dim=0).cpu().numpy()

    plot_heatmap(mean_map_hc, f"Tokenizer/Backbone map (HC) {tag}", f"{out_dir}/{tag}_map_hc.png",
                 xlabel="Time", ylabel="d_model")
    plot_heatmap(mean_map_ad, f"Tokenizer/Backbone map (AD) {tag}", f"{out_dir}/{tag}_map_ad.png",
                 xlabel="Time", ylabel="d_model")

    v_hc = feat_hc.mean(dim=0).cpu().numpy()         # [d]
    v_ad = feat_ad.mean(dim=0).cpu().numpy()

    plot_vector_compare(v_hc, v_ad, labels=["HC", "AD"],
                        title=f"Pre-classifier feature mean {tag}",
                        out_path=f"{out_dir}/{tag}_preclf_vector.png")
