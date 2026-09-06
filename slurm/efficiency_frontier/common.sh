#!/usr/bin/env bash
set -euo pipefail

EFFICIENCY_FRONTIER_ROOT="${EFFICIENCY_FRONTIER_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
EFFICIENCY_FRONTIER_PIPELINE="${EFFICIENCY_FRONTIER_PIPELINE:-configs/efficiency_frontier/pipeline.yaml}"
cd "${EFFICIENCY_FRONTIER_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export HF_HOME="${HF_HOME:-${EFFICIENCY_FRONTIER_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${EFFICIENCY_FRONTIER_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/efficiency_frontier/slurm

run_efficiency_frontier_stage() {
  local command="${EFFICIENCY_FRONTIER_COMMAND:?EFFICIENCY_FRONTIER_COMMAND is required}"
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier "${command}" \
    --pipeline "${EFFICIENCY_FRONTIER_PIPELINE}"
}
