#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cc3m_scale/slurm/%x_%A_%a.out
#SBATCH --error=logs/cc3m_scale/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/cc3m_scale/common.sh
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_acquisition \
  "${CC3M_COMMAND:?CC3M_COMMAND is required}" --pipeline "${CC3M_PIPELINE}"

