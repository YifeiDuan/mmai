#!/bin/bash
#SBATCH --job-name=exp7_albef
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%4
#SBATCH --output=results/exp7_albef/slurm_seed%a_%A.out
#SBATCH --error=results/exp7_albef/slurm_seed%a_%A.err

# Exp 7: ALBEF-style "align before fuse"
# Initialise CGCNN+BERT from CLIPAlignmentModel (Exp 4) instead of exp1+exp2
# baselines, then continue with the same multi-task FiLM recipe as Exp 5b
# (no FiLM warm-start, multi-task is_metal aux head, Huber loss).

SEEDS=(42 1337 2024 7 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/exp7_albef

# Reuse exp5 yaml but redirect output and disable FiLM warm-start
TMPCFG=$(mktemp /tmp/exp7_seed${SEED}_XXXXXX.yaml)
sed 's|output_dir: "results/exp5_multitask_film"|output_dir: "results/exp7_albef"|' \
    configs/exp5_multitask_film.yaml > $TMPCFG

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Exp7 ALBEF seed=$SEED on $(hostname)"
$PY scripts/run_exp5_multitask_film.py --config $TMPCFG --seed $SEED \
    --no-warm-start \
    --align-init results/exp4_alignment/stage1/best_model.pt
rm -f $TMPCFG
echo "[$(date '+%F %T')] Done seed=$SEED"
