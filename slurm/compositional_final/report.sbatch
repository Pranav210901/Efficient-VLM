#!/usr/bin/env bash
#SBATCH --job-name=comp_report
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --constraint=fs_weka
#SBATCH --output=artifacts/07_final_evaluation/compositional/logs/%x_%j.out
#SBATCH --error=artifacts/07_final_evaluation/compositional/logs/%x_%j.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final report
