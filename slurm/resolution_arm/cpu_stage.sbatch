#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/resolution_arm/slurm/%x_%j.out
#SBATCH --error=logs/resolution_arm/slurm/%x_%j.err
set -euo pipefail

cd "${RESOLUTION_ARM_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_arm \
  "${RESOLUTION_ARM_COMMAND:?}" --pipeline "${RESOLUTION_ARM_PIPELINE:?}"

