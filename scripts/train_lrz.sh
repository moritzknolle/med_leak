#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1
#SBATCH --job-name=chexpert
#SBATCH --output=slurm_out/%j-%x.out
#SBATCH --mem-per-gpu=100g
#SBATCH -p mcml-dgx-a100-40x8  # mcml-dgx-a100-40x8 # mcml-hgx-h100-94x4  # mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH -t 2-00:00:00

source /dss/dsshome1/03/ga92hev2/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate keras
echo 'Opened virtual environment'

python chexpert.py --eval_only=False --n_runs=200 --save_root="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/npy" --logdir="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/moritz/ckpts" --logdir="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/logs/chexpert/wrn_28_2"

echo 'Done'