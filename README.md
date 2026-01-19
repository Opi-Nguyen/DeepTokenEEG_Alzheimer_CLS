# EEG AD/HC Research Project

This repository implements a **reproducible research pipeline** for **Alzheimer’s Disease (AD)** vs **Healthy Control (HC)** classification using EEG.

## Highlights

- Configurable preprocessing pipeline (**MNE** + **SWT-based band reconstruction** + segmentation + **per-segment z-score**)
- **Subject-stratified** Train/Val/Test split (**no subject leakage**)
- CNN/ResNet-style 1D backbone with **Tokenizer** + **ResidualBlock1D** (supports **ablation** over number of blocks)
- Separate scripts for **data preparation**, **training**, **evaluation**, **ablation**, and **visualization**
- Rich visualizations across stages (**preprocessing**, **features**, **predictions**, **dataset statistics**)

---

## Project Structure

```text
eeg_ad_hc_project/
├─ configs/
│  ├─ data.yaml             # paths + signal parameters + band + channels + split
│  ├─ model.yaml            # model params + n_blocks + dilations
│  ├─ train.yaml            # training params (epochs, lr, patience...)
│  ├─ experiment.yaml       # run name/outdir + ablation list
│  └─ viz.yaml              # (optional) visualization settings
├─ scripts/
│  ├─ prepare_data.py       # scan raw EEG files, preprocess, split by subject, save cache
│  ├─ train.py              # train model and save best checkpoint
│  ├─ eval.py               # find thresholds on val, evaluate test w/ bootstrap
│  ├─ ablation.py           # run n_blocks in [1..4] and build comparison table
│  └─ visualize.py          # dataset + preprocessing + feature + prediction visualizations
├─ src/
│  ├─ data/
│  │  ├─ preprocessing.py   # MNE read/filter/resample + SWT band extract + segmentation + z-score
│  │  ├─ split.py           # subject-stratified split
│  │  └─ dataset.py         # pytorch datasets/loaders + subject-level probability aggregation
│  ├─ models/
│  │  ├─ model.py           # your Model + ablation n_blocks
│  │  └─ blocks.py          # (if used) ResidualBlock1D, Tokenizer, etc.
│  ├─ train/
│  │  ├─ trainer.py         # training loop w/ early stopping + ReduceLROnPlateau
│  │  ├─ thresholds.py      # best segment/subj thresholds search on validation set
│  │  └─ evaluator.py       # bootstrap evaluation on test set (segment and subject levels)
│  ├─ utils/
│  │  ├─ io.py              # save/load npz/json
│  │  ├─ seed.py            # reproducible seeding
│  │  ├─ metrics.py         # metrics + bootstrapping utilities
│  │  ├─ plotting.py        # matplotlib helpers
│  │  └─ hooks.py           # your forward hooks (already implemented by you)
│  └─ viz/
│     ├─ selectors.py       # pick representative AD/HC subjects and sample segments
│     ├─ stats_plots.py     # dataset-level statistics plots
│     ├─ stage_plots.py     # time-series, PSD, segment heatmaps
│     ├─ preprocessing_plots.py
│     ├─ feature_plots.py
│     ├─ prediction_plots.py
│     └─ ablation_plots.py
└─ outputs/
   ├─ cache/                # dataset_*.npz + dataset_*_meta.json
   └─ runs/                 # training runs and figures
```

---

## Environment Setup

Recommended: **Python 3.9+** (works with 3.8+ depending on your MNE build).

### Option A: Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

### Option B: Conda

```bash
conda create -n eeg_ad_hc python=3.9 -y
conda activate eeg_ad_hc
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### GPU Notes

- If you want **CUDA**, install a PyTorch build matching your CUDA version.
- Otherwise, CPU training is supported (slower).

---

## Configuration

Open `configs/data.yaml` and set:

- `paths.raw_root`: path to your EEG data root folder (contains HC*/AD* subfolders)
- `paths.cache_dir`: where processed cache should be stored
- `signal.band_name`: `delta` / `theta` / `alpha` / `beta` / `gamma`
- `signal.fs_std`, `signal.fs_target`, `signal.seg_seconds`, `signal.overlap`
- `split.ratios` and `split.seed`

> **Important:** this project splits by **SUBJECT IDs** to avoid leakage.  
> Each segment belongs to exactly **one subject**.

---

## Pipeline Overview

### Step 1 — Prepare Data (Preprocessing + Caching + Split)

This step scans `.set` (and/or `.fif`) files and applies:

- Read → EEG pick
- Resample to `fs_std` (if needed)
- Broad FIR bandpass (**0.5–45 Hz**)
- Resample to `fs_target`
- Channel selection (closest biosemi if many channels; else 19 standard channels)
- SWT decomposition/reconstruction for band extraction
- FIR bandpass on extracted band
- Segmentation with overlap
- Per-segment z-score normalization
- Subject-level stratified split (train/val/test)
- Flatten segments into arrays `X_*`, `y_*`, `s_*` and save to NPZ

Run:

```bash
python scripts/prepare_data.py
```

Outputs (in `outputs/cache` by default):

- `dataset_<band>.npz`
- `dataset_<band>_meta.json`

Expected shapes:

- Train: `X=(N_train, T, C)`, `y=(N_train,)`, `s=(N_train,)`
- Validation: ...
- Test: ...

---

### Step 2 — Train

Training uses:

- `CrossEntropyLoss`
- Adam + weight decay
- ReduceLROnPlateau
- Early stopping (validation loss)
- Saves best checkpoint to: `outputs/runs/<exp_name>/blocks_<n>/best.pth`

Configure:

- Training: `configs/train.yaml`
- Model: `configs/model.yaml`

Run:

```bash
python scripts/train.py
```

Override config paths:

```bash
python scripts/train.py \
  --cfg_data configs/data.yaml \
  --cfg_model configs/model.yaml \
  --cfg_train configs/train.yaml \
  --cfg_exp configs/experiment.yaml
```

---

### Step 3 — Evaluate (Threshold Search + Bootstrap Metrics)

Evaluation does:

1. Find best **segment-level** threshold on **validation** (`thr=0.01..0.99`)
2. Find best **subject-level** threshold on **validation**  
   (subject probability = mean of segment probabilities)
3. Evaluate **test** with **bootstrapping**
   - Segment-level: sample segments with replacement
   - Subject-level: sample subjects with replacement

Run:

```bash
python scripts/eval.py
```

Output:

- `outputs/runs/<exp_name>/blocks_<n>/eval_report.json`

Metrics reported as **mean ± std**:

- Accuracy, F1, Precision, Recall (segment-level + subject-level)

---

## Ablation Study (n_blocks = 1..4)

Ablation runs training + evaluation for different numbers of ResidualBlock1D blocks.

Configure in `configs/experiment.yaml`:

```yaml
ablation:
  n_blocks_list: [1, 2, 3, 4]
```

Run:

```bash
python scripts/ablation.py
```

Outputs:

- `outputs/runs/<exp_name>/ablation_results.json`
- `outputs/runs/<exp_name>/ablation_results.md`

---

## Visualization (Research-Grade Figures)

This pipeline generates multiple figure groups:

### A) Dataset Statistics

- Segment-level class balance
- Subject-level class balance
- Segments per subject histogram (AD vs HC)

### B) Preprocessing Deep Dive

- Segment heatmaps from cached normalized segments (C × T)
- If raw files are found for representative subjects:
  - Time-series comparisons (broad-filtered vs band-extracted)
  - PSD comparisons

### C) Model Feature Probing (AD vs HC)

- Tokenizer/backbone activation map (mean heatmap, d_model × T)
- Pre-classifier feature vector comparison (mean vectors)
- PCA scatter plot of pre-classifier features

### D) Prediction Distributions (if checkpoint exists)

- Segment-level probability distribution: **P(AD)** for AD vs HC
- Subject-level probability distribution

### E) Ablation Curve (if ablation results exist)

- `subject_f1_mean` vs `n_blocks`

Run:

```bash
python scripts/visualize.py
```

Figures saved to:

- `outputs/runs/<exp_name>/blocks_<n>/figures/`

---

## Interpreting Outputs

### Cached Dataset Format (`dataset_*.npz`)

Contains:

- `X_train`, `y_train`, `s_train`
- `X_val`, `y_val`, `s_val`
- `X_test`, `y_test`, `s_test`

Shapes:

- `X_*`: `(N_segments, T, C)`
- `y_*`: `(N_segments,)`
- `s_*`: `(N_segments,)` subject_id per segment

### Evaluation Report (`eval_report.json`)

Contains:

- `segment`: `{ threshold, accuracy_mean/std, f1_mean/std, ... }`
- `subject`: `{ threshold, accuracy_mean/std, f1_mean/std, ... }`
- `threshold_search`: validation scores during threshold scan

### Figures (`figures/`)

Use figures to justify:

- why band extraction and normalization matter
- how model features differ between AD and HC
- confidence distributions and separability of representations

---

## Tips / Common Issues

- **Channel naming mismatch**  
  Update STANDARD_19 in `configs/data.yaml` and the pick logic in `src/data/preprocessing.py`.

- **Missing channel locations in EEGLAB `.set`**  
  Ensure EEG files include channel info; otherwise set montage manually.

- **Very short recordings**  
  If `T < segment_length`, no segments are produced; these subjects may be skipped.

- **Performance**  
  SWT is expensive. Consider caching more aggressively and/or parallelizing preprocessing.

- **Subject ID parsing**  
  Subject id is extracted using regex digits from the subject folder name.  
  If naming differs, update `extract_pid()` in `src/data/preprocessing.py`.

---

## Quick Run

### Minimal end-to-end

```bash
python scripts/prepare_data.py
python scripts/train.py
python scripts/eval.py
python scripts/visualize.py
```

### Ablation (Residual blocks)

```bash
python scripts/ablation.py
```

---

## Citation / Reporting Notes

If you use this code in a paper/report:

- Report **segment-level** and **subject-level** metrics (mean ± std via bootstrap).
- Explicitly state **subject-stratified** split to avoid leakage.
- Include **ablation** table for `n_blocks` to justify architecture choice.
- Include preprocessing + feature visualizations to support interpretability.
