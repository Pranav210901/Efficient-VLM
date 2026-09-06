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
  echo "Scale training requires teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}
mkdir -p logs/token_aggregator_scale_training/slurm \
  results/token_aggregator_scale_training/manifests \
  results/token_aggregator_scale_training/report
"${ALIGNMENT_V3_PYTHON}" \
  -m src.alignment_v3.token_aggregator_scale_training validate

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "96${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
train="$(submit --partition="${PARTITION}" --job-name=tas_train \
  --array="0-11%${MAX_GPUS}" \
  --export="ALL,TOKEN_SCALE_TRAIN_ROOT=${ROOT}" \
  slurm/token_aggregator_scale_training/train.sbatch)"
verify="$(submit --partition="${PARTITION}" --job-name=tas_verify \
  --dependency="afterok:${train}" \
  --export="ALL,TOKEN_SCALE_TRAIN_ROOT=${ROOT},TOKEN_SCALE_TRAIN_COMMAND=verify" \
  slurm/token_aggregator_scale_training/cpu.sbatch)"
evaluation="$(submit --partition="${PARTITION}" --job-name=tas_eval \
  --array="0-287%${MAX_GPUS}" --dependency="afterok:${verify}" \
  --export="ALL,TOKEN_SCALE_TRAIN_ROOT=${ROOT}" \
  slurm/token_aggregator_scale_training/evaluate.sbatch)"
profile="$(submit --partition="${PARTITION}" --job-name=tas_profile \
  --dependency="afterok:${train}" \
  --export="ALL,TOKEN_SCALE_TRAIN_ROOT=${ROOT}" \
  slurm/token_aggregator_scale_training/profile.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=tas_report \
  --dependency="afterok:${evaluation}:${profile}" \
  --export="ALL,TOKEN_SCALE_TRAIN_ROOT=${ROOT},TOKEN_SCALE_TRAIN_COMMAND=report" \
  slurm/token_aggregator_scale_training/cpu.sbatch)"
echo "Token scale training: train=${train} verify=${verify} eval=${evaluation} profile=${profile} report=${report}"
echo "12 training runs, 288 epoch evaluations, one same-allocation trained profile."
echo "C1 is reused; Flickr30k TEST is sealed and absent."
