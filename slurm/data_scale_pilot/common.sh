#!/usr/bin/env bash
set -euo pipefail
DATA_SCALE_ROOT="${DATA_SCALE_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
DATA_SCALE_PIPELINE="${DATA_SCALE_PIPELINE:-configs/data_scale_pilot/pipeline.yaml}"
cd "${DATA_SCALE_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export HF_HOME="${HF_HOME:-${DATA_SCALE_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${DATA_SCALE_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/data_scale_pilot/slurm
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.data_scale_pilot \
  "${DATA_SCALE_COMMAND:?DATA_SCALE_COMMAND is required}" \
  --pipeline "${DATA_SCALE_PIPELINE}"
