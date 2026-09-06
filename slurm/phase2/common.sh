#!/usr/bin/env bash
set -euo pipefail

PHASE2_ROOT="${PHASE2_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
cd "${PHASE2_ROOT}"
source "${PHASE2_ROOT}/scripts/phase2_env.sh"
phase2_resolve_python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

phase2_diagnostics() {
  local config="$1"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
  echo "active_config=${config}"
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
  "${PHASE2_PYTHON}" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("detected_gpus=" + str(torch.cuda.device_count()))'
  git rev-parse HEAD 2>/dev/null || echo "git_commit=unavailable"
}

run_phase2_stage() {
  local stage="$1" config="$2"; shift 2
  local task="${SLURM_ARRAY_TASK_ID:-main}"
  local run_dir="results/phase2/slurm_status/${stage}/${task}"
  mkdir -p "${run_dir}"
  cp "${config}" "${run_dir}/run_config.yaml"
  printf 'epoch,metric,value\n' > "${run_dir}/metrics_by_epoch.csv"
  "${PHASE2_PYTHON}" -c "from pathlib import Path; from src.phase2.status import environment_payload; from src.phase15.io_utils import atomic_json; atomic_json(environment_payload(Path('.').resolve()), Path('${run_dir}')/'environment.json')"
  "${PHASE2_PYTHON}" -c "from src.phase2.status import write_status; write_status('${run_dir}', 'RUNNING', stage='${stage}')"
  set +e
  "$@"
  local exit_code=$?
  set -e
  if [[ ${exit_code} -eq 0 ]]; then
    "${PHASE2_PYTHON}" -c "from src.phase15.io_utils import atomic_json; from src.phase2.status import complete_stage; complete_stage('${run_dir}', '${stage}', {'status':'COMPLETED','exit_code':0})"
  else
    printf 'Stage %s exited with code %s. See the Slurm stderr log for the complete Python or shell traceback.\n' "${stage}" "${exit_code}" > "${run_dir}/traceback.txt"
    "${PHASE2_PYTHON}" -c "from src.phase2.status import write_status; write_status('${run_dir}', 'FAILED', stage='${stage}', exit_code=${exit_code})"
  fi
  return "${exit_code}"
}
