#!/usr/bin/env bash
set -euo pipefail

ALIGNMENT_V2_ROOT="${ALIGNMENT_V2_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
ALIGNMENT_V2_PIPELINE="${ALIGNMENT_V2_PIPELINE:-configs/alignment_v2/pipeline.yaml}"
cd "${ALIGNMENT_V2_ROOT}"
source "${ALIGNMENT_V2_ROOT}/scripts/alignment_v2_env.sh"
alignment_v2_resolve_python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${ALIGNMENT_V2_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ALIGNMENT_V2_ROOT}/.cache/torch}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-${ALIGNMENT_V2_ROOT}/.cache/open_clip}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${OPEN_CLIP_CACHE_DIR}"

alignment_v2_diagnostics() {
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "array_index=${SLURM_ARRAY_TASK_ID:-none}"
  echo "pipeline=${ALIGNMENT_V2_PIPELINE}"
  echo "python=${ALIGNMENT_V2_PYTHON}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  "${ALIGNMENT_V2_PYTHON}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")'
}

run_alignment_v2_stage() {
  local stage="$1"
  shift
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local status_root="results/alignment_v2/slurm_status/${stage}/${task}"
  mkdir -p "${status_root}"
  printf '{"status":"RUNNING","stage":"%s","job_id":"%s","array_index":"%s"}\n' \
    "${stage}" "${SLURM_JOB_ID:-none}" "${task}" > "${status_root}/status.json"
  set +e
  "$@"
  local exit_code=$?
  set -e
  if [[ ${exit_code} -eq 0 ]]; then
    printf '{"status":"COMPLETED","stage":"%s","exit_code":0}\n' "${stage}" > "${status_root}/status.json"
  else
    printf '{"status":"FAILED","stage":"%s","exit_code":%s}\n' "${stage}" "${exit_code}" > "${status_root}/status.json"
  fi
  return "${exit_code}"
}

run_alignment_v2_notebook_stage() {
  local stage="$1"
  local notebook_selector="${2:-${stage}}"
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local job_id="${SLURM_JOB_ID:-manual}"
  local output="results/alignment_v2/notebooks/${stage}/${job_id}_${task}.ipynb"
  local notebook_command=(
    "${ALIGNMENT_V2_PYTHON}"
    scripts/run_alignment_v2_notebook_cell.py
    --stage "${notebook_selector}"
    --output "${output}"
  )
  if [[ "${task}" != "main" ]]; then
    notebook_command+=(--index "${task}")
  fi

  # Result mapping is an observational side effect. A notebook-rendering
  # problem must be visible in the log but must not invalidate a completed
  # training/evaluation task or block its Slurm dependants.
  set +e
  "${notebook_command[@]}"
  local notebook_exit=$?
  set -e
  if [[ ${notebook_exit} -ne 0 ]]; then
    echo "WARNING: result notebook stage ${stage} failed with exit ${notebook_exit}" >&2
  fi
  return 0
}
