#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/guarded_hard_negative_study/slurm/%x_%A_%a.out
#SBATCH --error=logs/guarded_hard_negative_study/slurm/%x_%A_%a.err
set -euo pipefail
cd "${GHN_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
seeds=(42 43 44)
seed="${seeds[${SLURM_ARRAY_TASK_ID:?}]}"
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.guarded_hard_negative_study encode --seed "$seed"
