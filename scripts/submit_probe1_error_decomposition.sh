#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=4
DRY_RUN=false
RESUME=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "Probe 1 requires teaching/native-BF16 RTX PRO 6000 Blackwell." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-4]$ ]] || {
  echo "--max-total-gpus must be in [1,4]." >&2
  exit 2
}

mkdir -p logs/probe1_error_decomposition/slurm results/probe1_error_decomposition
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.probe1_error_decomposition validate

submit() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "$((900000 + RANDOM))"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

prepare="$(submit --partition="${PARTITION}" --job-name=p1_prepare \
  --export="ALL,PROBE1_ROOT=${ROOT}" \
  slurm/probe1_error_decomposition/prepare.sbatch)"
extract="$(submit --partition="${PARTITION}" --job-name=p1_extract \
  --array="0-3%${MAX_GPUS}" --dependency="afterok:${prepare}" \
  --export="ALL,PROBE1_ROOT=${ROOT}" \
  slurm/probe1_error_decomposition/extract.sbatch)"
rank="$(submit --partition="${PARTITION}" --job-name=p1_rank \
  --dependency="afterok:${extract}" \
  --export="ALL,PROBE1_ROOT=${ROOT},PROBE1_COMMAND=rank" \
  slurm/probe1_error_decomposition/cpu.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=p1_report \
  --dependency="afterok:${rank}" \
  --export="ALL,PROBE1_ROOT=${ROOT},PROBE1_COMMAND=report" \
  slurm/probe1_error_decomposition/cpu.sbatch)"

echo "Probe 1: prepare=${prepare} extract=${extract} rank=${rank} report=${report}"
echo "Resume requested=${RESUME}; the frozen ambiguity phase is hash-verified and idempotent."
echo "Four inference-only GPU tasks: C4 seeds 42/43/44 and matched OpenCLIP validation control."
echo "Flickr TEST is structurally sealed; no training or checkpoint selection is performed."
