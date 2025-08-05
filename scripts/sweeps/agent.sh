#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --job-name=sweep_agent
#SBATCH --output=slurm_out/%j-%x-%t.out
#SBATCH --mem-per-gpu=100g
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH -q mcml
#SBATCH -t 2-00:00:00

source /dss/dsshome1/03/ga92hev2/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate keras_dp
echo 'Opened virtual environment'

srun wandb agent --count 10 mknolle/mimic-iv/nv76vlga