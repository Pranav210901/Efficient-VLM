#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION="teaching"
MAX_GPUS=8
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "Wave 1 is restricted to teaching/native-BF16 hardware." >&2; exit 2;
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2; exit 2;
}

mkdir -p logs/alignment_wave1/slurm results/alignment_wave1
submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "99${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
gpu() {
  local name="$1" command="$2" dependency="$3" pipeline="$4" count="$5" limit="$6"
  local args=(
    --partition="${PARTITION}" --job-name="${name}"
    --cpus-per-task=12 --mem=96G --time=08:00:00
    --output="logs/alignment_wave1/slurm/%x_%A_%a.out"
    --error="logs/alignment_wave1/slurm/%x_%A_%a.err"
  )
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  (( count <= 1 )) || args+=(--array="0-$((count - 1))%${limit}")
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${pipeline},ALIGNMENT_V4_COMMAND=${command},ALIGNMENT_WAVE0_NUM_WORKERS=12" slurm/alignment_v4/gpu_stage.sbatch)
  submit "${args[@]}"
}
cpu() {
  local name="$1" command="$2" dependency="$3" pipeline="$4"
  local args=(
    --partition="${PARTITION}" --job-name="${name}"
    --output="logs/alignment_wave1/slurm/%x_%j.out"
    --error="logs/alignment_wave1/slurm/%x_%j.err"
  )
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${pipeline},ALIGNMENT_V4_COMMAND=${command}" slurm/alignment_v4/cpu_stage.sbatch)
  submit "${args[@]}"
}

# Validate on the submit node so a bad recipe never enters Slurm.
for teacher in mobileclip2 siglip2; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate \
    --pipeline "configs/alignment_wave1/pipeline_${teacher}.yaml"
done

# Split the global concurrency budget evenly across the two teacher arms.
per_teacher=$((MAX_GPUS / 2))
(( per_teacher >= 1 )) || per_teacher=1
declare -a reports=()
for teacher in mobileclip2 siglip2; do
  pipeline="configs/alignment_wave1/pipeline_${teacher}.yaml"
  prefix="w1_${teacher:0:3}"
  prefetch="$(cpu "${prefix}_prefetch" prefetch "" "${pipeline}")"
  cache="$(gpu "${prefix}_cache" teacher-cache "afterok:${prefetch}" "${pipeline}" 1 1)"
  train="$(gpu "${prefix}_train" train-sensitivity "afterok:${cache}" "${pipeline}" 6 "${per_teacher}")"
  eval_job="$(gpu "${prefix}_eval" eval-sensitivity "afterok:${train}" "${pipeline}" 6 "${per_teacher}")"
  report="$(cpu "${prefix}_report" report-wave1 "afterok:${eval_job}" "${pipeline}")"
  reports+=("${report}")
  echo "${teacher}: prefetch=${prefetch} cache=${cache} train=${train} eval=${eval_job} report=${report}"
done

echo "Wave 1 queued as one dependency graph. No Flickr30k test evaluation is included."
echo "Reports: ${reports[*]}"
