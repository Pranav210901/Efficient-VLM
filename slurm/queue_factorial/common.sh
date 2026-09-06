#!/usr/bin/env bash
set -euo pipefail

QUEUE_FACTORIAL_ROOT="${QUEUE_FACTORIAL_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
QUEUE_FACTORIAL_PIPELINE="${QUEUE_FACTORIAL_PIPELINE:-configs/queue_factorial/pipeline.yaml}"
cd "${QUEUE_FACTORIAL_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${QUEUE_FACTORIAL_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${QUEUE_FACTORIAL_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/queue_factorial/slurm

run_queue_factorial_stage() {
  local command="${QUEUE_FACTORIAL_COMMAND:?QUEUE_FACTORIAL_COMMAND is required}"
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_factorial "${command}" \
    --pipeline "${QUEUE_FACTORIAL_PIPELINE}"
}

