#!/bin/bash
#SBATCH --job-name=exp8_3mod
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp8_three_modality/slurm_seed%a_%A.out
#SBATCH --error=results/exp8_three_modality/slurm_seed%a_%A.err

# Exp 8: 3-modality fusion. Same recipe as Exp 5b plus a frozen ResNet18 image branch.
SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/exp8_three_modality

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Exp8 3-modality seed=$SEED on $(hostname)"
$PY scripts/run_exp8_three_modality.py --seed $SEED
echo "[$(date '+%F %T')] Done seed=$SEED"
