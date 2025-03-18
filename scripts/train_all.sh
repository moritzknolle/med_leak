#!/bin/bash

# !WARNING! This script will take a LONG time to run on a single GPU (>300h on a single A100).
# The runs can be arbitrarily parrralelised with more GPUS, simply (re)-run each line on every GPU-machine you have available.

python chexpert.py --eval_only=False --n_runs=200
python mimic.py --eval_only=False --n_runs=200
python embed.py --eval_only=False --n_runs=200
python fairvision.py --eval_only=False --n_runs=200
python fitzpatrick.py --eval_only=False --n_runs=200