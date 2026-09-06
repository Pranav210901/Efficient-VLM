#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/final_lr_study/slurm/%x_%A_%a.out
#SBATCH --error=logs/final_lr_study/slurm/%x_%A_%a.err
set -euo pipefail
cd "${FINAL_LR_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
index="${SLURM_ARRAY_TASK_ID:?}"
epoch_index="$((index % 24))"
case "$((index / 24))" in
  0) pipeline=configs/final_lr_study/pipeline_lr_0p5.yaml ;;
  1) pipeline=configs/final_lr_study/pipeline_lr_0p75.yaml ;;
  2) pipeline=configs/final_lr_study/pipeline_lr_1p25.yaml ;;
  3) pipeline=configs/final_lr_study/pipeline_lr_1p5.yaml ;;
  *) exit 2 ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_distillation_trajectory evaluate --pipeline "${pipeline}" --index "${epoch_index}"
