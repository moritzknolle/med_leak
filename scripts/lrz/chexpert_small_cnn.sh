#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --job-name=chexpert_small_cnn
#SBATCH --output=slurm_out/%j-%x-%t.out
#SBATCH --mem-per-gpu=100g
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH -q mcml
#SBATCH -t 2-00:00:00

source /dss/dsshome1/03/ga92hev2/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate keras
echo 'Opened virtual environment'

srun python chexpert.py \
    --eval_only=False \
    --n_runs=200 \
    --model='small_cnn' \
    --learning_rate=0.1 \
    --save_root="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/npy" \
    --ckpt_file_path="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/moritz/ckpts" \
    --logdir="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/logs/chexpert/small_cnn" \