#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/probe1_error_decomposition/slurm/%x_%j.out
#SBATCH --error=logs/probe1_error_decomposition/slurm/%x_%j.err
set -euo pipefail
cd "${PROBE1_ROOT:?}"
export HF_HOME="${HF_HOME:-${PROBE1_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${PROBE1_ROOT}/.cache/torch}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-${PROBE1_ROOT}/.cache/open_clip}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.probe1_error_decomposition prepare-gallery
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.probe1_error_decomposition freeze-ambiguity
