#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/07_final_evaluation/zero_shot/logs/%x_%A_%a.out
#SBATCH --error=artifacts/07_final_evaluation/zero_shot/logs/%x_%A_%a.err
set -euo pipefail
ROOT="${FINAL_ZERO_SHOT_ROOT:?FINAL_ZERO_SHOT_ROOT is required}"
cd "$ROOT"
exec scripts/final_zero_shot_python.sh \
  -m src.alignment_v3.final_zero_shot evaluate \
  --pipeline configs/final_zero_shot/pipeline.yaml
