#1. DeepTokenEEG model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name DeepTokenEEG --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#2. BIOT model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name BIOT --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#3. Conformer model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Conformer --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#4. TimesNet model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TimesNet --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#5. TCN model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name TCN --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TCN --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name TCN --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TCN --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TCN --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name TCN --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#6. LEAD model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name LEAD --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#7 EEG2Rep model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name EEG2Rep --scope multidataset \
  --run_name exp_best_model_structure --epochs 100

#8. Transformer model - Train Script
python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --dataset AD-Audiotory --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --dataset ADFSU --scope single \
  --run_name exp_best_model_structure --epochs 100 

python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --dataset ADFTD --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --dataset APAVA --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --dataset BrainLat --scope single \
  --run_name exp_best_model_structure --epochs 100

python -m scripts.train --config configs/config.yaml \
  --model_name Transformer --scope multidataset \
  --run_name exp_best_model_structure --epochs 100