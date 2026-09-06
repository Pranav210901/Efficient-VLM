#!/usr/bin/env bash
set -euo pipefail

ALIGNMENT_V3_ROOT="${ALIGNMENT_V3_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
ALIGNMENT_V3_PIPELINE="${ALIGNMENT_V3_PIPELINE:-configs/alignment_v3_recovery/pipeline.yaml}"
cd "${ALIGNMENT_V3_ROOT}"
source "${ALIGNMENT_V3_ROOT}/scripts/alignment_v3_env.sh"
alignment_v3_resolve_python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${ALIGNMENT_V3_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ALIGNMENT_V3_ROOT}/.cache/torch}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-${ALIGNMENT_V3_ROOT}/.cache/open_clip}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALIGNMENT_V3_OUTPUT_ROOT="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
import os
from src.utils.config import load_config
print(load_config(os.environ["ALIGNMENT_V3_PIPELINE"])["output_root"])
PY
)"
export ALIGNMENT_V3_OUTPUT_ROOT
mkdir -p "${ALIGNMENT_V3_OUTPUT_ROOT}/slurm_status" logs/alignment_v3_recovery/slurm

alignment_v3_recovery_diagnostics() {
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "array_index=${SLURM_ARRAY_TASK_ID:-none}"
  echo "command=${ALIGNMENT_V3_COMMAND:-unset}"
  echo "pipeline=${ALIGNMENT_V3_PIPELINE}"
  echo "output_root=${ALIGNMENT_V3_OUTPUT_ROOT}"
  echo "python=${ALIGNMENT_V3_PYTHON}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  "${ALIGNMENT_V3_PYTHON}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")'
}

run_alignment_v3_recovery_stage() {
  local command="${ALIGNMENT_V3_COMMAND:?ALIGNMENT_V3_COMMAND is required}"
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local status_root="${ALIGNMENT_V3_OUTPUT_ROOT}/slurm_status/${command}/${task}"
  mkdir -p "${status_root}"
  printf '{"status":"RUNNING","command":"%s","job_id":"%s","array_index":"%s"}\n' \
    "${command}" "${SLURM_JOB_ID:-none}" "${task}" > "${status_root}/status.json"
  set +e
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner \
    "${command}" --pipeline "${ALIGNMENT_V3_PIPELINE}"
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

