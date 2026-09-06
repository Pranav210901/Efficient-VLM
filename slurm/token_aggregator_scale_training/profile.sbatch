#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/token_aggregator_scale_training/slurm/%x_%j.out
#SBATCH --error=logs/token_aggregator_scale_training/slurm/%x_%j.err
set -euo pipefail
cd "${TOKEN_SCALE_TRAIN_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.token_aggregator_scale_training profile

