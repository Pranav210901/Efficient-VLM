#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=6
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "OpenCLIP Wave 1 is restricted to teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-6]$ ]] || {
  echo "--max-total-gpus must be in [1,6]." >&2
  exit 2
}

PIPELINE=configs/alignment_wave1/pipeline_openclip.yaml
mkdir -p logs/alignment_wave1/slurm results/alignment_wave1/openclip
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "97${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
cpu() {
  local name="$1" command="$2" dependency="$3"
  local args=(
    --partition="${PARTITION}" --job-name="${name}"
    --output="logs/alignment_wave1/slurm/%x_%j.out"
    --error="logs/alignment_wave1/slurm/%x_%j.err"
  )
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${PIPELINE},ALIGNMENT_V4_COMMAND=${command}" slurm/alignment_v4/cpu_stage.sbatch)
  submit "${args[@]}"
}
gpu() {
  local name="$1" command="$2" dependency="$3" count="$4" concurrency="$5"
  local args=(
    --partition="${PARTITION}" --job-name="${name}"
    --cpus-per-task=12 --mem=96G --time=08:00:00
    --output="logs/alignment_wave1/slurm/%x_%A_%a.out"
    --error="logs/alignment_wave1/slurm/%x_%A_%a.err"
  )
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  (( count <= 1 )) || args+=(--array="0-$((count - 1))%${concurrency}")
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${PIPELINE},ALIGNMENT_V4_COMMAND=${command},ALIGNMENT_WAVE0_NUM_WORKERS=12" slurm/alignment_v4/gpu_stage.sbatch)
  submit "${args[@]}"
}

prefetch="$(cpu w1_opc_prefetch prefetch "")"
cache="$(gpu w1_opc_cache teacher-cache "afterok:${prefetch}" 1 1)"
train="$(gpu w1_opc_train train-sensitivity "afterok:${cache}" 6 "${MAX_GPUS}")"
evaluation="$(gpu w1_opc_eval eval-sensitivity "afterok:${train}" 6 "${MAX_GPUS}")"
report="$(cpu w1_opc_report report-wave1 "afterok:${evaluation}")"
comparison="$(cpu w1_teacher_compare report-wave1-teachers "afterok:${report}")"

echo "OpenCLIP Wave 1: prefetch=${prefetch} cache=${cache} train=${train} eval=${evaluation} report=${report} comparison=${comparison}"
echo "Six training jobs are included: fresh baseline/distilled x seeds 42/43/44."
echo "No Flickr30k test evaluation is included."
