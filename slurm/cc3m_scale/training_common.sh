#!/usr/bin/env bash
set -euo pipefail
CC3M_ROOT="${CC3M_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
CC3M_PIPELINE="${CC3M_PIPELINE:-configs/cc3m_scale/pipeline.yaml}"
cd "${CC3M_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export HF_HOME="${HF_HOME:-${CC3M_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CC3M_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/cc3m_scale/training

