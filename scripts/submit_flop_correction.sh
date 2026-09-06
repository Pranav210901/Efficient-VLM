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
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "FLOP correction is restricted to the frontier hardware profile." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

mkdir -p logs/efficiency_frontier/slurm logs/resolution_arm/slurm
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.flop_correction snapshot

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "94${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
frontier() {
  local name="$1" command="$2" count="$3"
  submit \
    --partition="${PARTITION}" \
    --job-name="${name}" \
    --array="0-$((count - 1))%${MAX_GPUS}" \
    --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=${command}" \
    slurm/efficiency_frontier/gpu_profile.sbatch
}
resolution() {
  submit \
    --partition="${PARTITION}" \
    --job-name=flop_resolution \
    --array="0-3%${MAX_GPUS}" \
    --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=configs/resolution_arm/pipeline.yaml,RESOLUTION_ARM_KIND=module,RESOLUTION_ARM_COMMAND=profile" \
    slurm/resolution_arm/gpu_stage.sbatch
}

stress="$(frontier flop_stress profile 9)"
dynamic="$(frontier flop_dynamic profile-dynamic 9)"
fusion="$(submit \
  --partition="${PARTITION}" \
  --job-name=flop_fusion \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=profile-mobileclip-fusion" \
  slurm/efficiency_frontier/gpu_profile.sbatch)"
res="$(resolution)"

frontier_report="$(submit \
  --partition="${PARTITION}" \
  --job-name=flop_frontier_report \
  --dependency="afterok:${stress}:${dynamic}:${fusion}" \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=aggregate" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
dynamic_report="$(submit \
  --partition="${PARTITION}" \
  --job-name=flop_dynamic_report \
  --dependency="afterok:${frontier_report}" \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=report-dynamic" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
resolution_report="$(submit \
  --partition="${PARTITION}" \
  --job-name=flop_resolution_report \
  --dependency="afterok:${res}" \
  --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=configs/resolution_arm/pipeline.yaml,RESOLUTION_ARM_COMMAND=report" \
  slurm/resolution_arm/cpu_stage.sbatch)"
audit="$(submit \
  --partition="${PARTITION}" \
  --job-name=flop_audit \
  --dependency="afterok:${dynamic_report}:${resolution_report}" \
  --wrap="cd '${ROOT}' && source scripts/alignment_v3_env.sh && alignment_v3_resolve_python && \"\${ALIGNMENT_V3_PYTHON}\" -m src.alignment_v3.flop_correction report")"

echo "FLOP correction: stress=${stress} dynamic=${dynamic} fusion=${fusion} resolution=${res}"
echo "Reports: frontier=${frontier_report} dynamic=${dynamic_report} resolution=${resolution_report} audit=${audit}"
echo "No retrieval evaluation or training is included."
