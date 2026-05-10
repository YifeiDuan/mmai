#!/bin/bash
#SBATCH --job-name=exp5b_nowarm
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp5b_multitask_film_nowarm/slurm_seed%a_%A.out
#SBATCH --error=results/exp5b_multitask_film_nowarm/slurm_seed%a_%A.err

# Exp 5b: same as Exp 5 but skips FiLM warm-start (uses CGCNN+BERT init only).
# Tests whether the FiLM warm-start ckpt is fighting against the multi-task
# objective (different optimization target than the Huber+BCE we add here).
SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/exp5b_multitask_film_nowarm

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Starting Exp5b (no-warm) seed=$SEED on $(hostname)"

# Use same config but override output_dir + disable warm start
TMPCFG=$(mktemp /tmp/exp5b_seed${SEED}_XXX.yaml)
sed 's|output_dir: "results/exp5_multitask_film"|output_dir: "results/exp5b_multitask_film_nowarm"|; s|enabled: true|enabled: false|' configs/exp5_multitask_film.yaml > $TMPCFG
$PY scripts/run_exp5_multitask_film.py --config $TMPCFG --seed $SEED --no-warm-start
rm -f $TMPCFG
echo "[$(date '+%F %T')] Done seed=$SEED"
