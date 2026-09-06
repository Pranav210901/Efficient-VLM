#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/teacher_extension_224/slurm/%x_%j.out
#SBATCH --error=logs/teacher_extension_224/slurm/%x_%j.err
set -euo pipefail

cd "${TEACHER_EXTENSION_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

case "${TEACHER_EXTENSION_COMMAND:?}" in
  verify)
    for pipeline in \
      configs/teacher_extension_224/pipeline_siglip2.yaml \
      configs/teacher_extension_224/pipeline_openclip.yaml \
      configs/teacher_extension_224/pipeline_multi.yaml
    do
      "${ALIGNMENT_V3_PYTHON}" \
        -m src.alignment_v3.resolution_distillation_trajectory verify-snapshots \
        --pipeline "${pipeline}"
    done
    ;;
  report)
    for pipeline in \
      configs/teacher_extension_224/pipeline_siglip2.yaml \
      configs/teacher_extension_224/pipeline_openclip.yaml \
      configs/teacher_extension_224/pipeline_multi.yaml
    do
      "${ALIGNMENT_V3_PYTHON}" \
        -m src.alignment_v3.resolution_distillation_trajectory report \
        --pipeline "${pipeline}"
    done
    exec "${ALIGNMENT_V3_PYTHON}" \
      -m src.alignment_v3.teacher_extension_report report
    ;;
  *)
    echo "unknown command ${TEACHER_EXTENSION_COMMAND}" >&2
    exit 2
    ;;
esac

