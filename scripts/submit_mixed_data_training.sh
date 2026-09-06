#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
DRY_RUN=false
MAX_GPUS=6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${MAX_GPUS}" =~ ^[1-6]$ ]] || {
  echo "--max-total-gpus must be in [1,6]." >&2
  exit 2
}
mkdir -p logs/mixed_data_training/slurm \
  results/mixed_data_training/{manifests,report} \
  checkpoints/mixed_data_training \
  data/cc3m_v1_1/teacher_cache
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.mixed_data_training \
  validate --pipeline configs/mixed_data_training/pipeline.yaml

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "98${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

cache="$(submit --job-name=md_cch \
  --export="ALL,MIXED_DATA_ROOT=${ROOT}" \
  slurm/mixed_data_training/teacher_cache.sbatch)"
train="$(submit --job-name=md_train \
  --dependency="afterok:${cache}" --array="0-5%${MAX_GPUS}" \
  --export="ALL,MIXED_DATA_ROOT=${ROOT}" \
  slurm/mixed_data_training/train.sbatch)"
evaluate="$(submit --job-name=md_eval \
  --dependency="afterok:${train}" --array="0-143%${MAX_GPUS}" \
  --export="ALL,MIXED_DATA_ROOT=${ROOT}" \
  slurm/mixed_data_training/evaluate.sbatch)"
report="$(submit --job-name=md_report \
  --dependency="afterok:${evaluate}" \
  --export="ALL,MIXED_DATA_ROOT=${ROOT}" \
  slurm/mixed_data_training/report.sbatch)"

echo "Mixed-data package: teacher-cache=${cache} train=${train} eval=${evaluate} report=${report}"
echo "Flickr30k TEST jobs: 0"
