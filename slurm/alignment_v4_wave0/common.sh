#!/usr/bin/env bash
set -euo pipefail

ALIGNMENT_WAVE0_ROOT="${ALIGNMENT_WAVE0_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
ALIGNMENT_WAVE0_PIPELINE="${ALIGNMENT_WAVE0_PIPELINE:-configs/alignment_v4_wave0/pipeline.yaml}"
cd "${ALIGNMENT_WAVE0_ROOT}"
if [[ -n "${ALIGNMENT_V3_PYTHON:-}" && -x "${ALIGNMENT_V3_PYTHON}" ]]; then
  echo "alignment-v3: using explicitly supplied Python ${ALIGNMENT_V3_PYTHON}" >&2
else
  source scripts/alignment_v3_env.sh
  alignment_v3_resolve_python
fi
ALIGNMENT_WAVE0_PYTHON="${ALIGNMENT_V3_PYTHON}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${ALIGNMENT_WAVE0_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ALIGNMENT_WAVE0_ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" logs/alignment_v4_wave0/slurm

run_alignment_wave0_stage() {
  local command="${ALIGNMENT_WAVE0_COMMAND:?ALIGNMENT_WAVE0_COMMAND is required}"
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local status_root="results/alignment_v4_wave0/slurm_status/${command}/${task}"
  mkdir -p "${status_root}"
  printf '{"status":"RUNNING","command":"%s","job_id":"%s","array_index":"%s"}\n' \
    "${command}" "${SLURM_JOB_ID:-none}" "${task}" > "${status_root}/status.json"
  echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${task} command=${command}"
  echo "python=${ALIGNMENT_WAVE0_PYTHON} cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"
  echo "partition_profile=${ALIGNMENT_WAVE0_PARTITION:-unspecified} precision=${ALIGNMENT_WAVE0_PRECISION:-config_default} workers=${ALIGNMENT_WAVE0_NUM_WORKERS:-config_default}"
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  set +e
  "${ALIGNMENT_WAVE0_PYTHON}" -m src.alignment_v3.runner "${command}" \
    --pipeline "${ALIGNMENT_WAVE0_PIPELINE}"
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
