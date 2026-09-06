#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/resolution_distillation_224_long/slurm/%x_%A_%a.out
#SBATCH --error=logs/resolution_distillation_224_long/slurm/%x_%A_%a.err
set -euo pipefail

cd "${RD224_LONG_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner train-sensitivity \
  --pipeline "${RD224_LONG_PIPELINE:?}"
