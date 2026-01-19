#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-eeg_ad_hc_project}"

# Folders
mkdir -p "$PROJECT"/configs
mkdir -p "$PROJECT"/scripts
mkdir -p "$PROJECT"/src/{utils,data,models,train,viz}
mkdir -p "$PROJECT"/outputs/{cache,runs,figures}

# Root files
: > "$PROJECT/README.md"
: > "$PROJECT/requirements.txt"

# Config files
: > "$PROJECT/configs/data.yaml"
: > "$PROJECT/configs/model.yaml"
: > "$PROJECT/configs/train.yaml"
: > "$PROJECT/configs/experiment.yaml"
: > "$PROJECT/configs/viz.yaml"

# Script files
: > "$PROJECT/scripts/prepare_data.py"
: > "$PROJECT/scripts/train.py"
: > "$PROJECT/scripts/eval.py"
: > "$PROJECT/scripts/test.py"
: > "$PROJECT/scripts/ablation.py"
: > "$PROJECT/scripts/visualize.py"

# src/utils
: > "$PROJECT/src/utils/seed.py"
: > "$PROJECT/src/utils/io.py"
: > "$PROJECT/src/utils/metrics.py"
: > "$PROJECT/src/utils/plotting.py"
: > "$PROJECT/src/utils/hooks.py"

# src/data
: > "$PROJECT/src/data/preprocessing.py"
: > "$PROJECT/src/data/dataset.py"
: > "$PROJECT/src/data/split.py"

# src/models
: > "$PROJECT/src/models/blocks.py"
: > "$PROJECT/src/models/model.py"

# src/train
: > "$PROJECT/src/train/trainer.py"
: > "$PROJECT/src/train/thresholds.py"

# src/viz
: > "$PROJECT/src/viz/stage_plots.py"
: > "$PROJECT/src/viz/feature_plots.py"

echo "✅ Created structure at: $PROJECT"
