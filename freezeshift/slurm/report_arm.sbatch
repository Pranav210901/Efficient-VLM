#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=freezeshift/logs/slurm/%x_%A_%a.out
#SBATCH --error=freezeshift/logs/slurm/%x_%A_%a.err
set -euo pipefail
cd "${FREEZESHIFT_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner report-arm --pipeline-index "${SLURM_ARRAY_TASK_ID:?}"
