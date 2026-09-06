#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v4_wave0/slurm/%x_%j.out
#SBATCH --error=logs/alignment_v4_wave0/slurm/%x_%j.err
set -euo pipefail

ROOT="${ALIGNMENT_WAVE0_ROOT:?ALIGNMENT_WAVE0_ROOT is required}"
PIPELINE="${ALIGNMENT_WAVE0_PIPELINE:?ALIGNMENT_WAVE0_PIPELINE is required}"
PYTHON="${ALIGNMENT_V3_PYTHON:?ALIGNMENT_V3_PYTHON is required}"
PARALLEL_TASKS="${ALIGNMENT_WAVE0_PARALLEL_TASKS:?ALIGNMENT_WAVE0_PARALLEL_TASKS is required}"
cd "${ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALIGNMENT_WAVE0_PARTITION=teaching
export ALIGNMENT_WAVE0_PRECISION=bf16
export ALIGNMENT_WAVE0_NUM_WORKERS=10
mkdir -p "${HF_HOME}" "${TORCH_HOME}" logs/alignment_v4_wave0/slurm

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#GPU_IDS[@]} < PARALLEL_TASKS )); then
  echo "Expected ${PARALLEL_TASKS} allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

ACTIVE_PIDS=()
forward_signal() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill -USR1 "${pid}" 2>/dev/null || true
  done
}
trap forward_signal USR1

run_parallel_stage() {
  local command="$1"
  local index gpu status_root stdout_path stderr_path pid
  local -a pids=()
  local failed=0
  ACTIVE_PIDS=()
  for ((index=0; index<PARALLEL_TASKS; index++)); do
    gpu="${GPU_IDS[$index]}"
    status_root="results/alignment_v4_wave0/slurm_status/${command}/${index}"
    stdout_path="logs/alignment_v4_wave0/slurm/a4w0_rescreen_confirm_${command}_${SLURM_JOB_ID}_${index}.out"
    stderr_path="logs/alignment_v4_wave0/slurm/a4w0_rescreen_confirm_${command}_${SLURM_JOB_ID}_${index}.err"
    mkdir -p "${status_root}"
    printf '{"status":"RUNNING","command":"%s","job_id":"%s","index":%s,"gpu_slot":"%s"}\n' \
      "${command}" "${SLURM_JOB_ID}" "${index}" "${gpu}" > "${status_root}/status.json"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      echo "host=$(hostname) job=${SLURM_JOB_ID} index=${index} command=${command} gpu=${gpu}"
      "${PYTHON}" -m src.alignment_v3.runner "${command}" \
        --pipeline "${PIPELINE}" --index "${index}"
    ) >"${stdout_path}" 2>"${stderr_path}" &
    pid=$!
    pids+=("${pid}")
    ACTIVE_PIDS+=("${pid}")
  done
  for ((index=0; index<PARALLEL_TASKS; index++)); do
    command_pid="${pids[$index]}"
    status_root="results/alignment_v4_wave0/slurm_status/${command}/${index}"
    if wait "${command_pid}"; then
      printf '{"status":"COMPLETED","command":"%s","exit_code":0}\n' \
        "${command}" > "${status_root}/status.json"
    else
      exit_code=$?
      printf '{"status":"FAILED","command":"%s","exit_code":%s}\n' \
        "${command}" "${exit_code}" > "${status_root}/status.json"
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  return "${failed}"
}

"${PYTHON}" -m src.alignment_v3.runner ledger-pending-stage \
  --pipeline "${PIPELINE}" --ledger-stage wave0-rescreen-confirmation

run_parallel_stage train-wave0-rescreen-confirmation
run_parallel_stage eval-wave0-rescreen-confirmation

report_status="results/alignment_v4_wave0/slurm_status/report-wave0-rescreen-confirmation/main"
mkdir -p "${report_status}"
printf '{"status":"RUNNING","command":"report-wave0-rescreen-confirmation","job_id":"%s"}\n' \
  "${SLURM_JOB_ID}" > "${report_status}/status.json"
"${PYTHON}" -m src.alignment_v3.runner report-wave0-rescreen-confirmation \
  --pipeline "${PIPELINE}"
printf '{"status":"COMPLETED","command":"report-wave0-rescreen-confirmation","exit_code":0}\n' \
  > "${report_status}/status.json"
