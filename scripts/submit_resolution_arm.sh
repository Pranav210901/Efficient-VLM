#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=8
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
  echo "Resolution arm is restricted to teaching/native-BF16 hardware." >&2; exit 2;
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2; exit 2;
}

PIPELINE=configs/resolution_arm/pipeline.yaml
mkdir -p logs/resolution_arm/slurm results/resolution_arm
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_arm validate --pipeline "${PIPELINE}"

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "98${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
gpu() {
  local name="$1" kind="$2" command="$3" dependency="$4" count="$5" concurrency="$6"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  (( count <= 1 )) || args+=(--array="0-$((count - 1))%${concurrency}")
  args+=(--export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_KIND=${kind},RESOLUTION_ARM_COMMAND=${command}" slurm/resolution_arm/gpu_stage.sbatch)
  submit "${args[@]}"
}

train="$(gpu res_train runner train-resolution "" 6 "${MAX_GPUS}")"
# Retrieval evaluation and profiling deliberately use separate allocations.
eval_job="$(gpu res_eval module evaluate "afterok:${train}" 6 "${MAX_GPUS}")"
profile="$(gpu res_profile module profile "afterok:${train}" 4 "${MAX_GPUS}")"
report="$(submit --partition="${PARTITION}" --job-name=res_report \
  --dependency="afterok:${eval_job}:${profile}" \
  --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_COMMAND=report" \
  slurm/resolution_arm/cpu_stage.sbatch)"

echo "Resolution arm: train=${train} eval=${eval_job} profile=${profile} report=${report}"
echo "Flickr30k TEST is not referenced by this graph."
