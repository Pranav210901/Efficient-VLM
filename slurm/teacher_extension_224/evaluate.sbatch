#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/teacher_extension_224/slurm/%x_%A_%a.out
#SBATCH --error=logs/teacher_extension_224/slurm/%x_%A_%a.err
set -euo pipefail

cd "${TEACHER_EXTENSION_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-11}"

index="${SLURM_ARRAY_TASK_ID:?}"
local_index="$((index % 72))"
case "$((index / 72))" in
  0) pipeline="configs/teacher_extension_224/pipeline_siglip2.yaml" ;;
  1) pipeline="configs/teacher_extension_224/pipeline_openclip.yaml" ;;
  2) pipeline="configs/teacher_extension_224/pipeline_multi.yaml" ;;
  *) echo "invalid teacher-extension evaluation index ${index}" >&2; exit 2 ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.resolution_distillation_trajectory evaluate \
  --pipeline "${pipeline}" --index "${local_index}"

