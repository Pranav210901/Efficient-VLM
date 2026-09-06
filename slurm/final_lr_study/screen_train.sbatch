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
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) pipeline=configs/final_lr_study/pipeline_lr_0p5.yaml ;;
  1) pipeline=configs/final_lr_study/pipeline_lr_0p75.yaml ;;
  2) pipeline=configs/final_lr_study/pipeline_lr_1p25.yaml ;;
  3) pipeline=configs/final_lr_study/pipeline_lr_1p5.yaml ;;
  *) exit 2 ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner train-sensitivity --pipeline "${pipeline}" --index 0
