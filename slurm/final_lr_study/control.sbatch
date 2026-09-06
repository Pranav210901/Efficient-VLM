#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/final_lr_study/slurm/%x_%j.out
#SBATCH --error=logs/final_lr_study/slurm/%x_%j.err
set -euo pipefail
cd "${FINAL_LR_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.final_lr_study "${FINAL_LR_COMMAND:?}"
