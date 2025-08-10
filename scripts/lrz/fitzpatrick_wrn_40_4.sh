#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --job-name=fitzpatrick_wrn_40_4
#SBATCH --output=slurm_out/%j-%x.out
#SBATCH --mem-per-gpu=50g
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -t 2-00:00:00

source /dss/dsshome1/03/ga92hev2/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate keras
echo "Opened virtual environment"

srun --output=slurm_out/%j-%x-%t.out bash scripts/run_until_error.sh python fitzpatrick.py \
    --eval_only=False \
    --n_runs=200 \
    --model="wrn_40_4" \
    --save_root="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/npy" \
    --ckpt_file_path="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/ckpts" \
    --logdir="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/logs/fitzpatrick/wrn_40_4" \
