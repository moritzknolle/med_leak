#!/bin/bash

PATIENT_LEVEL_ONLY=False # if set to True only patient/record-level MIA analysis is performed (no aggregate attack success). This is faster

python mem_inf_stats.py --logdir="./logs/chexpert/wrn_28_2" --dataset="chexpert" --patient_level_only=$PATIENT_LEVEL_ONLY
python mem_inf_stats.py --logdir="./logs/mimic/wrn_28_2" --dataset="mimic" --patient_level_only=$PATIENT_LEVEL_ONLY
python mem_inf_stats.py --logdir="./logs/embed/wrn_28_2" --dataset="embed" --patient_level_only=$PATIENT_LEVEL_ONLY
python mem_inf_stats.py --logdir="./logs/fairvision/wrn_28_2" --dataset="fairvision" --patient_level_only=$PATIENT_LEVEL_ONLY
python mem_inf_stats.py --logdir="./logs/fitzpatrick/wrn_28_2" --dataset="fitzpatrick" --patient_level_only=$PATIENT_LEVEL_ONLY

python plots.py --log_scale=True --mia_method="rmia"
python plots.py --log_scale=False --mia_method="rmia"
python plots.py --log_scale=True --mia_method="lira"
python plots.py --log_scale=False --mia_method="lira"

python model_scaling_plots.py --dataset_name='chexpert'
python model_scaling_plots.py --dataset_name='fitzpatrick'