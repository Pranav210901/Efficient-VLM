#!/usr/bin/env bash
set -euo pipefail
cd "${MIXED_DATA_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export HF_HOME="${HF_HOME:-${MIXED_DATA_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${MIXED_DATA_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/mixed_data_training/slurm
