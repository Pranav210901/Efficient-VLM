#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/resolution_distillation_224/slurm/%x_%A_%a.out
#SBATCH --error=logs/resolution_distillation_224/slurm/%x_%A_%a.err
set -euo pipefail

cd "${RESOLUTION_DISTILLATION_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
if [[ "${RESOLUTION_DISTILLATION_KIND:?}" == "runner" ]]; then
  exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner \
    "${RESOLUTION_DISTILLATION_COMMAND:?}" \
    --pipeline "${RESOLUTION_DISTILLATION_PIPELINE:?}"
fi
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_distillation \
  "${RESOLUTION_DISTILLATION_COMMAND:?}" \
  --pipeline "${RESOLUTION_DISTILLATION_PIPELINE:?}"
