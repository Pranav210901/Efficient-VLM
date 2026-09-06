#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=freezeshift/logs/slurm/%x_%A_%a.out
#SBATCH --error=freezeshift/logs/slurm/%x_%A_%a.err
set -euo pipefail
cd "${FREEZESHIFT_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
index="${SLURM_ARRAY_TASK_ID:?}"
exec "${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner evaluate \
  --pipeline-index "$((index / 72))" --epoch-index "$((index % 72))"
