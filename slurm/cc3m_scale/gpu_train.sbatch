#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/cc3m_scale/training/%x_%A_%a.out
#SBATCH --error=logs/cc3m_scale/training/%x_%A_%a.err
set -euo pipefail
source slurm/cc3m_scale/training_common.sh
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_training \
  train --pipeline "${CC3M_PIPELINE}"

