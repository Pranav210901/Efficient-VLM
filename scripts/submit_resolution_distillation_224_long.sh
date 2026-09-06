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
  echo "24-epoch trajectory is restricted to teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

PIPELINE=configs/resolution_distillation_224_long/pipeline.yaml
mkdir -p logs/resolution_distillation_224_long/slurm \
  results/resolution_distillation_224_long/manifests
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.resolution_distillation_trajectory validate \
  --pipeline "${PIPELINE}"

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "93${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

train="$(submit \
  --partition="${PARTITION}" \
  --job-name=rd224l_train \
  --array="0-2%3" \
  --export="ALL,RD224_LONG_ROOT=${ROOT},RD224_LONG_PIPELINE=${PIPELINE}" \
  slurm/resolution_distillation_224_long/train.sbatch)"
verify="$(submit \
  --partition="${PARTITION}" \
  --job-name=rd224l_verify \
  --dependency="afterok:${train}" \
  --export="ALL,RD224_LONG_ROOT=${ROOT},RD224_LONG_PIPELINE=${PIPELINE},RD224_LONG_COMMAND=verify-snapshots" \
  slurm/resolution_distillation_224_long/cpu.sbatch)"
evaluation="$(submit \
  --partition="${PARTITION}" \
  --job-name=rd224l_eval \
  --array="0-71%${MAX_GPUS}" \
  --dependency="afterok:${verify}" \
  --export="ALL,RD224_LONG_ROOT=${ROOT},RD224_LONG_PIPELINE=${PIPELINE}" \
  slurm/resolution_distillation_224_long/evaluate.sbatch)"
report="$(submit \
  --partition="${PARTITION}" \
  --job-name=rd224l_report \
  --dependency="afterok:${evaluation}" \
  --export="ALL,RD224_LONG_ROOT=${ROOT},RD224_LONG_PIPELINE=${PIPELINE},RD224_LONG_COMMAND=report" \
  slurm/resolution_distillation_224_long/cpu.sbatch)"

echo "24-epoch trajectory: train=${train} verify=${verify} eval=${evaluation} report=${report}"
echo "Training: 3 seeds in parallel. Evaluation: 72 immutable epoch checkpoints."
echo "Flickr30k TEST is not referenced anywhere in this graph."
