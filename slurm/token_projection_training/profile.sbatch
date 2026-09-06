#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/token_projection_training/slurm/%x_%A_%a.out
#SBATCH --error=logs/token_projection_training/slurm/%x_%A_%a.err
set -euo pipefail
cd "${TOKEN_TRAIN_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) pipeline="configs/token_projection_training/pipeline_a.yaml" ;;
  1) pipeline="configs/token_projection_training/pipeline_b.yaml" ;;
  2) pipeline="configs/token_projection_training/pipeline_c.yaml" ;;
  *) echo "invalid token-projection profile index" >&2; exit 2 ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.resolution_distillation_trajectory profile \
  --pipeline "${pipeline}"

