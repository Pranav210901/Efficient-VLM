#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/mixed_data_training/slurm/%x_%j.out
#SBATCH --error=logs/mixed_data_training/slurm/%x_%j.err
set -euo pipefail
source slurm/mixed_data_training/common.sh
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.mixed_data_training \
  teacher-cache --pipeline configs/mixed_data_training/pipeline.yaml
