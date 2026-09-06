#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=3
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
  echo "224px combination is restricted to teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-3]$ ]] || {
  echo "--max-total-gpus must be in [1,3]." >&2
  exit 2
}

PIPELINE=configs/resolution_distillation_224/pipeline.yaml
mkdir -p logs/resolution_distillation_224/slurm \
  results/resolution_distillation_224/manifests
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_distillation validate \
  --pipeline "${PIPELINE}"

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "95${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
gpu() {
  local name="$1" kind="$2" command="$3" dependency="$4"
  local args=(--partition="${PARTITION}" --job-name="${name}" --array="0-2%${MAX_GPUS}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(
    --export="ALL,RESOLUTION_DISTILLATION_ROOT=${ROOT},RESOLUTION_DISTILLATION_PIPELINE=${PIPELINE},RESOLUTION_DISTILLATION_KIND=${kind},RESOLUTION_DISTILLATION_COMMAND=${command}"
    slurm/resolution_distillation_224/gpu_stage.sbatch
  )
  submit "${args[@]}"
}

train="$(gpu rd224_train runner train-sensitivity "")"
evaluation="$(gpu rd224_eval module evaluate "afterok:${train}")"
report="$(submit \
  --partition="${PARTITION}" \
  --job-name=rd224_report \
  --dependency="afterok:${evaluation}" \
  --export="ALL,RESOLUTION_DISTILLATION_ROOT=${ROOT},RESOLUTION_DISTILLATION_PIPELINE=${PIPELINE},RESOLUTION_DISTILLATION_COMMAND=report" \
  slurm/resolution_distillation_224/cpu_stage.sbatch)"

echo "224px MobileCLIP2 combination: train=${train} eval=${evaluation} report=${report}"
echo "Exactly three distilled seeds are included. Flickr30k TEST remains sealed."
