#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/mixed_data_training/slurm/%x_%j.out
#SBATCH --error=logs/mixed_data_training/slurm/%x_%j.err
set -euo pipefail
source slurm/mixed_data_training/common.sh
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.mixed_data_training \
  report --pipeline configs/mixed_data_training/pipeline.yaml
