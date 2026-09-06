#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/mixed_data_training/slurm/%x_%A_%a.out
#SBATCH --error=logs/mixed_data_training/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/mixed_data_training/common.sh
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.mixed_data_training \
  evaluate --pipeline configs/mixed_data_training/pipeline.yaml
