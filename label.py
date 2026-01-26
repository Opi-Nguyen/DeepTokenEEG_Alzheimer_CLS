# count_labels_per_dataset.py
# In ra số lượng nhãn (AD=0, HC=1, Others=2+, Unknown=-1) theo từng dataset
# Chạy:
#   python count_labels_per_dataset.py \
#     --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
#     --adftd_root /mnt/sda1/home/sparc/nqthinh/LEAD_DEEPTOKEN/LEAD/data_preprocessing/ADFTD/ds004504

import os
import re
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


# --------------------------
# Label policy (same as preprocessing.py)
# --------------------------
def _norm(x: str) -> str:
    x = ("" if x is None else str(x)).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def _has_token(s: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", s) is not None

def label_from_text(text: str, other_map: Dict[str, int]) -> int:
    t = _norm(text)

    # ADFTD short codes
    if t == "a":
        return 0
    if t == "c":
        return 1
    if t == "f":
        if t not in other_map:
            other_map[t] = 2 if not other_map else max(other_map.values()) + 1
        return other_map[t]

    # AD
    if ("alzheimer" in t) or ("alzheim" in t) or _has_token(t, "ad"):
        return 0

    # HC
    if any(k in t for k in ["healthy", "control", "normal", "cn"]) or _has_token(t, "hc"):
        return 1

    # Unknown
    if t in {"", "n/a", "na", "unknown", "unlabeled", "none", "nan"}:
        return -1

    # Others (dynamic)
    if t not in other_map:
        other_map[t] = 2 if not other_map else max(other_map.values()) + 1
    return other_map[t]


# --------------------------
# Utilities
# --------------------------
def load_participants_map(root: str) -> Dict[str, str]:
    """
    participants.tsv -> {sub-xxx : label_text}
    tự dò cột label hợp lý (diagnosis/group/status/condition/dx/phenotype/disease).
    """
    tsv = os.path.join(root, "participants.tsv")
    if not os.path.exists(tsv):
        return {}

    if pd is None:
        with open(tsv, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if not lines:
            return {}
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]

        pid_idx = 0
        for i, c in enumerate(header):
            if c.strip().lower() in {"participant_id", "subject", "sub_id", "id"}:
                pid_idx = i
                break

        label_idx = 1 if len(header) > 1 else 0
        for i, c in enumerate(header):
            cl = c.strip().lower()
            if any(k in cl for k in ["diagnos", "group", "status", "condition", "dx", "phenotype", "disease"]):
                label_idx = i
                break

        out = {}
        for r in rows:
            if len(r) <= max(pid_idx, label_idx):
                continue
            out[str(r[pid_idx])] = str(r[label_idx])
        return out

    df = pd.read_csv(tsv, sep="\t")
    pid_col = None
    for c in df.columns:
        if c.lower() in {"participant_id", "subject", "sub_id", "id"}:
            pid_col = c
            break
    if pid_col is None:
        pid_col = df.columns[0]

    label_col = None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["diagnos", "group", "status", "condition", "dx", "phenotype", "disease"]):
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else pid_col

    out = {}
    for _, row in df.iterrows():
        out[str(row[pid_col])] = str(row[label_col])
    return out

def find_first_ancestor_label(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    for p in reversed(parts):
        pl = p.lower()
        if "adsz" in pl:
            continue
        if pl == "ad" or "_ad" in pl or "alzheimer" in pl or re.search(r"\b1_ad\b", pl):
            return "AD"
        if "hc" in pl or "healthy" in pl or "control" in pl or "normal" in pl or "cn" in pl:
            return "HC"
        if "mci" in pl:
            return "MCI"
        if "ftd" in pl:
            return "FTD"
    return ""


# --------------------------
# Scanners (labels only; no loading data)
# --------------------------
def scan_bids_set(root: str) -> List[Tuple[str, str]]:
    """
    Returns list of (file_path, disease_text)
    """
    pmap = load_participants_map(root)
    items: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() != "eeg":
            continue
        for fn in filenames:
            if not fn.lower().endswith(".set"):
                continue
            fpath = os.path.join(dirpath, fn)
            m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
            sub = m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            disease = (
                pmap.get(sub, "")
                or pmap.get(sub.replace("sub-", ""), "")
                or find_first_ancestor_label(fpath)
            )
            items.append((fpath, disease))
    items.sort()
    return items

def scan_apava(root: str) -> List[Tuple[str, str]]:
    """
    APAVA: labels from participants.tsv if present, else fallback AD_positive if 23 mats
    """
    pmap = load_participants_map(root)
    mats = [os.path.join(root, fn) for fn in sorted(os.listdir(root)) if fn.lower().endswith(".mat")]
    AD_positive = {1, 3, 6, 8, 9, 11, 12, 13, 15, 17, 19, 21}

    items: List[Tuple[str, str]] = []
    for i, fpath in enumerate(mats, start=1):
        m = re.search(r"(sub-[A-Za-z0-9]+)", fpath)
        sub = m.group(1) if m else f"sub-{i:03d}"
        disease = pmap.get(sub, "") or pmap.get(sub.replace("sub-", ""), "")
        if not disease:
            if len(mats) == 23:
                disease = "AD" if i in AD_positive else "HC"
            else:
                disease = find_first_ancestor_label(fpath)
        items.append((fpath, disease))
    return items

def scan_adfsu(root: str) -> List[Tuple[str, str]]:
    """
    ADFSU: count per Paciente folder (has .txt files)
      ADFSU/AD|Healthy/Eyes_closed|Eyes_open/Paciente*/**/*.txt
    """
    items: List[Tuple[str, str]] = []
    for disease_folder in sorted(os.listdir(root)):
        disease_path = os.path.join(root, disease_folder)
        if not os.path.isdir(disease_path):
            continue
        for state in sorted(os.listdir(disease_path)):
            state_path = os.path.join(disease_path, state)
            if not os.path.isdir(state_path):
                continue
            for paciente in sorted(os.listdir(state_path)):
                paciente_path = os.path.join(state_path, paciente)
                if not os.path.isdir(paciente_path):
                    continue
                has_txt = False
                for dp, _, fns in os.walk(paciente_path):
                    if any(fn.lower().endswith(".txt") for fn in fns):
                        has_txt = True
                        break
                if has_txt:
                    items.append((paciente_path, disease_folder))
    items.sort()
    return items


def main(datasets_root: str, adftd_root: str):
    other_map: Dict[str, int] = {}
    per_dataset_counts = {}
    per_dataset_rawtexts = {}

    # scan datasets_root
    for dname in sorted(os.listdir(datasets_root)):
        droot = os.path.join(datasets_root, dname)
        if not os.path.isdir(droot):
            continue
        if "adsz" in dname.lower():
            continue

        dn_l = dname.lower()
        if "apava" in dn_l:
            items = scan_apava(droot)
        elif "adfsu" in dn_l:
            items = scan_adfsu(droot)
        else:
            items = scan_bids_set(droot)

        cnt = Counter()
        rawtxt = Counter()
        for _, disease_text in items:
            rawtxt[_norm(disease_text)] += 1
            lb = label_from_text(disease_text, other_map)
            cnt[lb] += 1

        per_dataset_counts[dname] = cnt
        per_dataset_rawtexts[dname] = rawtxt
        print(f"[SCAN] {dname}: n_items={len(items)}")

    # scan adftd_root
    if adftd_root and os.path.isdir(adftd_root):
        dn = os.path.basename(os.path.normpath(adftd_root))
        items = scan_bids_set(adftd_root)
        cnt = Counter()
        rawtxt = Counter()
        for _, disease_text in items:
            rawtxt[_norm(disease_text)] += 1
            lb = label_from_text(disease_text, other_map)
            cnt[lb] += 1
        per_dataset_counts[dn] = cnt
        per_dataset_rawtexts[dn] = rawtxt
        print(f"[SCAN] {dn}: n_items={len(items)}")

    # pretty print
    print("\n=== LABEL COUNTS PER DATASET (by scanned items) ===")
    for dn in sorted(per_dataset_counts.keys()):
        cnt = per_dataset_counts[dn]
        print(f"\n{dn}")
        print(f"  AD(0): {cnt.get(0,0)}")
        print(f"  HC(1): {cnt.get(1,0)}")
        others = {k:v for k,v in cnt.items() if k >= 2}
        if others:
            print(f"  Others(2+): {dict(sorted(others.items()))}")
        print(f"  Unknown(-1): {cnt.get(-1,0)}")

        # show top raw label strings (help debug participants.tsv)
        top_raw = per_dataset_rawtexts[dn].most_common(8)
        print("  top disease_text:", top_raw)

    print("\nother_map (dynamic others assignment):", other_map)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", type=str, required=True)
    ap.add_argument("--adftd_root", type=str, default="")
    args = ap.parse_args()
    main(args.datasets_root, args.adftd_root)

"""




python -u "/mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/label.py" \
  --datasets_root /mnt/sda1/home/sparc/nqthinh/DeepTokenEEG_Alzheimer_CLS/dataset \
  --adftd_root /mnt/sda1/home/sparc/nqthinh/LEAD_DEEPTOKEN/LEAD/data_preprocessing/ADFTD/ds004504


"""