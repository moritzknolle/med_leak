#!/bin/bash

conda activate med-leak
# !WARNING! This script will take a LONG time to run on a single GPU.
# The runs can be arbitrarily parrralelised with more GPUS, simply (re)-run each line for each GPU-machine you have available.

python chexpert.py --eval_only=False --n_runs=200 --logdir="./logs/chexpert/wrn_28_2"
python mimic.py --eval_only=False --n_runs=200 --logdir="./logs/mimic/wrn_28_2"
python embed.py --eval_only=False --n_runs=200 --logdir="./logs/embed/wrn_28_2"
python fairvision.py --eval_only=False --n_runs=200 --logdir="./logs/fairvision/wrn_28_2"
python fitzpatrick.py --eval_only=False --n_runs=200 --logdir="./logs/fitzpatrick/wrn_28_2"


# model scaling experiments
python chexpert.py --eval_only=False --n_runs=200 --model='small_cnn' --logdir="./logs/chexpert/small_cnn" --learning_rate=0.1
python chexpert.py --eval_only=False --n_runs=200 --model='wrn_28_5' --logdir="./logs/chexpert/wrn_28_5"
python chexpert.py --eval_only=False --n_runs=200 --model='vit_b_16' --logdir="./logs/chexpert/vit_b_16" --img_size=128,128

python fitzpatrick.py --eval_only=False --n_runs=200 --model='small_cnn' --logdir="./logs/fitzpatrick/small_cnn"
python fitzpatrick.py --eval_only=False --n_runs=200 --model='wrn_28_5' --logdir="./logs/fitzpatrick/wrn_28_5"
python fitzpatrick.py --eval_only=False --n_runs=200 --model='vit_b_16' --logdir="./logs/fitzpatrick/vit_b_16"  --img_size=128,128

# open-source model attacks
conda activate torch
python xrv_scores.py
conda activate med-leak
python xrv_attack.py
