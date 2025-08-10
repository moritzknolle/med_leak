#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --job-name=ptb-xl_dp_epsinf
#SBATCH --output=slurm_out/%j-%x-%t.out
#SBATCH --mem-per-gpu=50g
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -t 2-00:00:00

source /dss/dsshome1/03/ga92hev2/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate keras_dp
echo "Opened virtual environment"

srun bash scripts/run_until_error.sh python ptb-xl_dp.py \
    --eval_only=False \
    --n_runs=200 \
    --dp=False \
    --save_root="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/npy" \
    --ckpt_file_path="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/ckpts" \
    --logdir="/dss/dssmcmlfs01/pn67bo/pn67bo-dss-0000/moritz/logs/ptb-xl/dp/epsinf"  \