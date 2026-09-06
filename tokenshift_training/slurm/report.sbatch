#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=tokenshift_training/logs/slurm/%x_%j.out
#SBATCH --error=tokenshift_training/logs/slurm/%x_%j.err
set -euo pipefail
cd "${TOKENSHIFT_TRAINING_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m tokenshift_training.runner report

