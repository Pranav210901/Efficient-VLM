#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/final_lr_study/slurm/%x_%A_%a.out
#SBATCH --error=logs/final_lr_study/slurm/%x_%A_%a.err
set -euo pipefail
cd "${FINAL_LR_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
pipeline="$("${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.final_lr_study selected-pipeline)"
[[ "${pipeline}" != SKIP ]] || { echo 'Current LR won screen; confirmation skipped.'; exit 0; }
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner train-sensitivity --pipeline "${pipeline}" --index "$((SLURM_ARRAY_TASK_ID + 1))"
