#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=tokenshift_training/logs/slurm/%x_%A_%a.out
#SBATCH --error=tokenshift_training/logs/slurm/%x_%A_%a.err
set -euo pipefail
cd "${TOKENSHIFT_TRAINING_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
index="${SLURM_ARRAY_TASK_ID:?}"
exec "${ALIGNMENT_V3_PYTHON}" -m tokenshift_training.runner evaluate-seed \
  --pipeline-index "$((index / 3))" --seed-index "$((index % 3))"

