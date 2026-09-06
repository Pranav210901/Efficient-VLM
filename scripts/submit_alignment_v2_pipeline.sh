#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/alignment_v2_env.sh"
alignment_v2_resolve_python

MODE="parallel"
PIPELINE="configs/alignment_v2/pipeline.yaml"
MAX_CONCURRENT=8
PARTITION="teaching"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --max-total-gpus) MAX_CONCURRENT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${MODE}" != "parallel" && "${MODE}" != "sequential" ]]; then
  echo "--mode must be parallel or sequential" >&2
  exit 2
fi
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-total-gpus must be a positive integer" >&2
  exit 2
fi
if [[ -z "${PARTITION}" ]]; then
  echo "--partition must not be empty" >&2
  exit 2
fi

export ALIGNMENT_V2_ROOT="${ROOT}"
export ALIGNMENT_V2_PIPELINE="${PIPELINE}"
echo "alignment-v2 python=${ALIGNMENT_V2_PYTHON}"
validate_timeout="${ALIGNMENT_V2_VALIDATE_TIMEOUT_SECONDS:-180}"
echo "alignment-v2: validating ${PIPELINE} (timeout ${validate_timeout}s)"
timeout --foreground "${validate_timeout}" \
  "${ALIGNMENT_V2_PYTHON}" -m src.alignment_v2.runner validate --pipeline "${PIPELINE}"

reference_jobs="$(( $(wc -l < results/alignment_v2/manifests/reference_jobs.csv) - 1 ))"
unimodal_jobs="$(( $(wc -l < results/alignment_v2/manifests/unimodal_jobs.csv) - 1 ))"
if (( reference_jobs < 1 || unimodal_jobs < 1 )); then
  echo "Validation produced an empty Slurm manifest." >&2
  exit 2
fi
mkdir -p \
  logs/alignment_v2/00_validate \
  logs/alignment_v2/00b_prefetch \
  logs/alignment_v2/01_reference_array \
  logs/alignment_v2/02_train_unimodal_array \
  logs/alignment_v2/03_evaluate_unimodal_array \
  logs/alignment_v2/04_report

submit() {
  local dry_id="$1"
  local submit_timeout="${ALIGNMENT_V2_SBATCH_TIMEOUT_SECONDS:-60}"
  local job_script
  shift
  job_script="${*: -1}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "${dry_id}"
  else
    echo "alignment-v2: submitting ${job_script} (timeout ${submit_timeout}s)" >&2
    local submission
    if submission="$(timeout --foreground "${submit_timeout}" sbatch --parsable "$@")"; then
      echo "alignment-v2: accepted ${job_script} as job ${submission}" >&2
      echo "${submission}"
    else
      local exit_code=$?
      echo "alignment-v2: sbatch did not complete for ${job_script} (exit ${exit_code})." >&2
      echo "Check 'squeue --me' before retrying in case Slurm accepted the job." >&2
      return "${exit_code}"
    fi
  fi
}

validate_id="$(submit 900001 --partition="${PARTITION}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/00_validate.sbatch)"
prefetch_id="$(submit 900002 --partition="${PARTITION}" --dependency="afterok:${validate_id}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/00b_prefetch.sbatch)"
if [[ "${MODE}" == "parallel" ]]; then
  reference_limit=1
  if (( MAX_CONCURRENT == 1 )); then
    train_limit=1
    train_dependency="afterok:${reference_id:-900003}"
  else
    train_limit=$((MAX_CONCURRENT - reference_limit))
    train_dependency="afterok:${prefetch_id}"
  fi
  reference_id="$(submit 900003 --partition="${PARTITION}" --dependency="afterok:${prefetch_id}" --array="0-$((reference_jobs - 1))%${reference_limit}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/01_reference_array.sbatch)"
  if (( MAX_CONCURRENT == 1 )); then
    train_dependency="afterok:${reference_id}"
  fi
  train_id="$(submit 900004 --partition="${PARTITION}" --dependency="${train_dependency}" --array="0-$((unimodal_jobs - 1))%${train_limit}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/02_train_unimodal_array.sbatch)"
else
  reference_id="$(submit 900003 --partition="${PARTITION}" --dependency="afterok:${prefetch_id}" --array="0-$((reference_jobs - 1))%1" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/01_reference_array.sbatch)"
  train_id="$(submit 900004 --partition="${PARTITION}" --dependency="afterok:${reference_id}" --array="0-$((unimodal_jobs - 1))%1" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/02_train_unimodal_array.sbatch)"
fi
evaluate_limit="${MAX_CONCURRENT}"
[[ "${MODE}" == "sequential" ]] && evaluate_limit=1
evaluate_id="$(submit 900005 --partition="${PARTITION}" --dependency="afterok:${train_id}" --array="0-$((unimodal_jobs - 1))%${evaluate_limit}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/03_evaluate_unimodal_array.sbatch)"
if [[ "${MODE}" == "parallel" ]]; then
  report_dependency="afterok:${reference_id}:${evaluate_id}"
else
  report_dependency="afterok:${evaluate_id}"
fi
report_id="$(submit 900006 --partition="${PARTITION}" --dependency="${report_dependency}" --export=ALL,ALIGNMENT_V2_ROOT="${ROOT}",ALIGNMENT_V2_PIPELINE="${PIPELINE}" slurm/alignment_v2/04_report.sbatch)"

echo "alignment-v2 submitted mode=${MODE} partition=${PARTITION} max_total_gpus=${MAX_CONCURRENT}"
echo "validate=${validate_id} prefetch=${prefetch_id} reference=${reference_id} train=${train_id} evaluate=${evaluate_id} report=${report_id}"
