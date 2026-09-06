#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=cc3m_finalise
#SBATCH --output=logs/cc3m_scale/slurm/%x_%j.out
#SBATCH --error=logs/cc3m_scale/slurm/%x_%j.err
set -euo pipefail
source slurm/cc3m_scale/common.sh
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_acquisition \
  finalise --pipeline "${CC3M_PIPELINE}"
