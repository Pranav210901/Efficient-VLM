#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=cc3m_touch
#SBATCH --output=logs/cc3m_scale/retention/%x_%j.out
#SBATCH --error=logs/cc3m_scale/retention/%x_%j.err
set -euo pipefail
source slurm/cc3m_scale/common.sh
mkdir -p logs/cc3m_scale/retention
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_acquisition \
  retention-refresh --pipeline "${CC3M_PIPELINE}"

