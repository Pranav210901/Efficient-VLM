#!/usr/bin/env bash
set -euo pipefail

ALIGNMENT_V4_ROOT="${ALIGNMENT_V4_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
ALIGNMENT_V4_PIPELINE="${ALIGNMENT_V4_PIPELINE:?ALIGNMENT_V4_PIPELINE is required}"
cd "${ALIGNMENT_V4_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
ALIGNMENT_V4_PYTHON="${ALIGNMENT_V3_PYTHON}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${ALIGNMENT_V4_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ALIGNMENT_V4_ROOT}/.cache/torch}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-${ALIGNMENT_V4_ROOT}/.cache/open_clip}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${OPEN_CLIP_CACHE_DIR}" logs/alignment_v4/slurm

alignment_v4_output_root() {
  "${ALIGNMENT_V4_PYTHON}" - "${ALIGNMENT_V4_PIPELINE}" <<'PY'
import sys
from src.utils.config import load_config
print(load_config(sys.argv[1])["output_root"])
PY
}

alignment_v4_diagnostics() {
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "array_index=${SLURM_ARRAY_TASK_ID:-none}"
  echo "command=${ALIGNMENT_V4_COMMAND:-unset}"
  echo "pipeline=${ALIGNMENT_V4_PIPELINE}"
  echo "python=${ALIGNMENT_V4_PYTHON}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  "${ALIGNMENT_V4_PYTHON}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")'
}

run_alignment_v4_stage() {
  local command="${ALIGNMENT_V4_COMMAND:?ALIGNMENT_V4_COMMAND is required}"
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local output
  output="$(alignment_v4_output_root)"
  local status_root="${output}/slurm_status/${command}/${task}"
  mkdir -p "${status_root}"
  printf '{"status":"RUNNING","command":"%s","job_id":"%s","array_index":"%s"}\n' \
    "${command}" "${SLURM_JOB_ID:-none}" "${task}" > "${status_root}/status.json"
  set +e
  "${ALIGNMENT_V4_PYTHON}" -m src.alignment_v3.runner "${command}" \
    --pipeline "${ALIGNMENT_V4_PIPELINE}"
  local exit_code=$?
  set -e
  if [[ ${exit_code} -eq 0 ]]; then
    printf '{"status":"COMPLETED","command":"%s","exit_code":0}\n' \
      "${command}" > "${status_root}/status.json"
  else
    printf '{"status":"FAILED","command":"%s","exit_code":%s}\n' \
      "${command}" "${exit_code}" > "${status_root}/status.json"
  fi
  return "${exit_code}"
}

