#!/bin/bash
#SBATCH --job-name=exp3_ms
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp3_fusion_multiseed/slurm_seed%a_%A.out
#SBATCH --error=results/exp3_fusion_multiseed/slurm_seed%a_%A.err

# Multi-seed Exp 3 FiLM (fusion, no multitask, no Huber) for ablation.
# Apples-to-apples baseline for: "what does multitask + Huber buy us on top of plain fusion?"
SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p "results/exp3_fusion_multiseed/seed_${SEED}/film"

TMPCFG=$(mktemp /tmp/exp3_seed${SEED}_XXXXXX.yaml)
sed "s|seed: 42|seed: ${SEED}|; s|output_dir: \"results/exp3_fusion\"|output_dir: \"results/exp3_fusion_multiseed/seed_${SEED}\"|" \
    configs/exp3_fusion.yaml > $TMPCFG

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Exp3 FiLM seed=$SEED on $(hostname)"
$PY scripts/run_exp3_fusion.py --config $TMPCFG --fusion film
rm -f $TMPCFG
echo "[$(date '+%F %T')] Done seed=$SEED"
