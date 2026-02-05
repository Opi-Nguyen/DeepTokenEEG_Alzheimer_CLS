python -m scripts.train_cv --config configs/train.yaml \
  --model_name DeepTokenEEG --scope single \
  --run_name exp_best_model_structure


python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --scope single \
  --run_name exp_best_model_structure