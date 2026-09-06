#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/e5_c4_224/slurm/%x_%j.out
#SBATCH --error=logs/e5_c4_224/slurm/%x_%j.err
set -euo pipefail
cd "${E5_C4_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.resolution_distillation_trajectory \
  "${E5_C4_COMMAND:?}" --pipeline "${E5_C4_PIPELINE:?}"

