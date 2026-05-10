#!/bin/bash
#SBATCH --job-name=exp6_crsattn
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp6_crossattn/slurm_seed%a_%A.out
#SBATCH --error=results/exp6_crossattn/slurm_seed%a_%A.err

SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/exp6_crossattn

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Starting Exp6 seed=$SEED on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$PY scripts/run_exp6_crossattn.py --seed $SEED
echo "[$(date '+%F %T')] Done seed=$SEED"
