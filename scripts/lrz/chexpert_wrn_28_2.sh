#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --job-name=chexpert_wrn_28_2
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
    --model='wrn_28_2' \
    --save_root="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/npy" \
    --ckpt_file_path="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/ckpts" \
    --logdir="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/logs/chexpert/wrn_28_2" \