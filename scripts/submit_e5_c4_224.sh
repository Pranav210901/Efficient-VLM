#!/usr/bin/env bash
set -euo pipefail
echo "BLOCKED: unconditional E5 C4 submission is superseded by the preregistered profile-first text aggregation study." >&2
echo "Use scripts/submit_text_aggregation_study.sh with an explicit --cpu-partition." >&2
exit 3
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
[[ "${PARTITION}" == teaching ]] || {
  echo "E5 C4 requires teaching/native-BF16 RTX PRO 6000 Blackwell hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

PIPELINE=configs/e5_c4_224/pipeline.yaml
mkdir -p logs/e5_c4_224/slurm results/e5_c4_224/manifests
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_distillation_trajectory validate --pipeline "${PIPELINE}"

submit() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "97${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

# Exactly three independent training tasks; each requests one GPU and may
# start as soon as an individual GPU is available.
train="$(submit --partition="${PARTITION}" --job-name=e5c4_train \
  --array="0-2%3" \
  --export="ALL,E5_C4_ROOT=${ROOT},E5_C4_PIPELINE=${PIPELINE}" \
  slurm/e5_c4_224/train.sbatch)"
verify="$(submit --partition="${PARTITION}" --job-name=e5c4_verify \
  --dependency="afterok:${train}" \
  --export="ALL,E5_C4_ROOT=${ROOT},E5_C4_PIPELINE=${PIPELINE},E5_C4_COMMAND=verify-snapshots" \
  slurm/e5_c4_224/cpu.sbatch)"
evaluation="$(submit --partition="${PARTITION}" --job-name=e5c4_eval \
  --array="0-71%${MAX_GPUS}" --dependency="afterok:${verify}" \
  --export="ALL,E5_C4_ROOT=${ROOT},E5_C4_PIPELINE=${PIPELINE}" \
  slurm/e5_c4_224/evaluate.sbatch)"
profile="$(submit --partition="${PARTITION}" --job-name=e5c4_profile \
  --dependency="afterok:${train}" \
  --export="ALL,E5_C4_ROOT=${ROOT},E5_C4_PIPELINE=${PIPELINE}" \
  slurm/e5_c4_224/profile.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=e5c4_report \
  --dependency="afterok:${evaluation}:${profile}" \
  --export="ALL,E5_C4_ROOT=${ROOT},E5_C4_PIPELINE=${PIPELINE},E5_C4_COMMAND=report" \
  slurm/e5_c4_224/cpu.sbatch)"

echo "E5 C4 224: train=${train} verify=${verify} eval=${evaluation} profile=${profile} report=${report}"
echo "Three training runs (seeds 42/43/44), 72 validation-only epoch evaluations, and one trained latency profile."
echo "Flickr30k TEST is sealed and absent from this graph."
