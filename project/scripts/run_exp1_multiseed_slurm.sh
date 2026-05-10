#!/bin/bash
#SBATCH --job-name=exp1_ms
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp1_cgcnn_multiseed/slurm_seed%a_%A.out
#SBATCH --error=results/exp1_cgcnn_multiseed/slurm_seed%a_%A.err

# Multi-seed Exp 1 (CGCNN, structure-only) for ablation in poster.
SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p "results/exp1_cgcnn_multiseed/seed_${SEED}"

TMPCFG=$(mktemp /tmp/exp1_seed${SEED}_XXXXXX.yaml)
sed "s|seed: 42|seed: ${SEED}|; s|output_dir: \"results/exp1_cgcnn\"|output_dir: \"results/exp1_cgcnn_multiseed/seed_${SEED}\"|" \
    configs/exp1_cgcnn.yaml > $TMPCFG

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Exp1 seed=$SEED on $(hostname)"
$PY scripts/run_exp1_cgcnn.py --config $TMPCFG
rm -f $TMPCFG
echo "[$(date '+%F %T')] Done seed=$SEED"
