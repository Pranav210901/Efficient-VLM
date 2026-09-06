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
  echo "Token-projection training requires teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

mkdir -p logs/token_projection_training/slurm \
  results/token_projection_training/manifests \
  results/token_projection_training/report
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.token_projection_training validate

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "97${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

train="$(submit --partition="${PARTITION}" --job-name=tp_train \
  --array="0-8%${MAX_GPUS}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT}" \
  slurm/token_projection_training/train.sbatch)"
verify="$(submit --partition="${PARTITION}" --job-name=tp_verify \
  --dependency="afterok:${train}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT},TOKEN_TRAIN_COMMAND=verify" \
  slurm/token_projection_training/cpu.sbatch)"
evaluation="$(submit --partition="${PARTITION}" --job-name=tp_eval \
  --array="0-215%${MAX_GPUS}" --dependency="afterok:${verify}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT}" \
  slurm/token_projection_training/evaluate.sbatch)"
profile="$(submit --partition="${PARTITION}" --job-name=tp_profile \
  --array="0-2%1" --dependency="afterok:${train}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT}" \
  slurm/token_projection_training/profile.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=tp_report \
  --dependency="afterok:${evaluation}:${profile}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT},TOKEN_TRAIN_COMMAND=report" \
  slurm/token_projection_training/cpu.sbatch)"

echo "Token projection: train=${train} verify=${verify} eval=${evaluation} profile=${profile} report=${report}"
echo "Nine training runs; 216 per-epoch validation evaluations; three serial quiet-GPU profiles."
echo "Flickr30k TEST is sealed and absent from this graph."
