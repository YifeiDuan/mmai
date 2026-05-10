#!/bin/bash
#SBATCH --job-name=exp5_film
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp5_multitask_film/slurm_seed%a_%A.out
#SBATCH --error=results/exp5_multitask_film/slurm_seed%a_%A.err

# Map array index → seed
SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/exp5_multitask_film

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Starting Exp5 seed=$SEED on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$PY scripts/run_exp5_multitask_film.py --seed $SEED
echo "[$(date '+%F %T')] Done seed=$SEED"
