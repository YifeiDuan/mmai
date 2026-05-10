#!/bin/bash
#SBATCH --job-name=inverse_design
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/inverse_design/slurm_%j.out
#SBATCH --error=results/inverse_design/slurm_%j.err

# Inverse Design Agent: 3 targets (1.0, 2.0, 3.0 eV) x 5 iters x 5 candidates
# = 75 candidates evaluated by 5-seed Exp 5b ensemble; LLM proposer is Qwen2.5-VL-3B.
# Single GPU, single job (not array) — full sweep takes ~30 min on l40s.

cd /home/jinzhta/mpppo/MMAI_2026spring/project
mkdir -p results/inverse_design

PY=/home/jinzhta/.conda/envs/mpppo/bin/python
echo "[$(date '+%F %T')] Starting inverse design on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

$PY scripts/run_inverse_design.py --config configs/inverse_design.yaml

echo "[$(date '+%F %T')] Done"
