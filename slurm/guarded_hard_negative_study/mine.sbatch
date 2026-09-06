#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/guarded_hard_negative_study/slurm/%x_%j.out
#SBATCH --error=logs/guarded_hard_negative_study/slurm/%x_%j.err
set -euo pipefail
cd "${GHN_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.guarded_hard_negative_study mine
