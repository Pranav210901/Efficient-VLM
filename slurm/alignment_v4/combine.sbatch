#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v4/slurm/%x_%A.out
#SBATCH --error=logs/alignment_v4/slurm/%x_%A.err
set -euo pipefail
ALIGNMENT_V4_ROOT="${ALIGNMENT_V4_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
cd "${ALIGNMENT_V4_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" scripts/combine_alignment_v4_results.py
"${ALIGNMENT_V3_PYTHON}" scripts/render_alignment_v4_notebook.py

