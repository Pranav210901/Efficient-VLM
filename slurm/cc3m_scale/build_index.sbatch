#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=cc3m_index
#SBATCH --output=logs/cc3m_scale/training/%x_%j.out
#SBATCH --error=logs/cc3m_scale/training/%x_%j.err
set -euo pipefail
source slurm/cc3m_scale/training_common.sh
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_training \
  build-index --pipeline "${CC3M_PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_training \
  validate --pipeline "${CC3M_PIPELINE}"

