#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/resolution_distillation_224_long/slurm/%x_%j.out
#SBATCH --error=logs/resolution_distillation_224_long/slurm/%x_%j.err
set -euo pipefail

cd "${RD224_LONG_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.resolution_distillation_trajectory \
  "${RD224_LONG_COMMAND:?}" --pipeline "${RD224_LONG_PIPELINE:?}"
