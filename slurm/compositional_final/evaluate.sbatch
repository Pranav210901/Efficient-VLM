#!/usr/bin/env bash
#SBATCH --job-name=comp_final
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=artifacts/07_final_evaluation/compositional/logs/%x_%A_%a.out
#SBATCH --error=artifacts/07_final_evaluation/compositional/logs/%x_%A_%a.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final evaluate --index "${SLURM_ARRAY_TASK_ID:?}"
