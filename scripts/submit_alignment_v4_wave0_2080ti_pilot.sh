#!/usr/bin/env bash
set -euo pipefail
echo "BLOCKED: queue-size 1024 is an accepted completed result and must not be retrained." >&2
exit 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
source scripts/alignment_v4_wave0_partition.sh
alignment_v4_wave0_partition_profile 2080ti 1

PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY-RUN: one 2080ti BF16 pilot, global follow-up index 4 (queue=1024)."
  echo "DRY-RUN: timing report compares epoch_seconds with teaching 56-61s."
  exit 0
fi

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
  --pipeline "${PIPELINE}" --ledger-stage wave0-queue-ablation --index 0

train_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name=a4w0_2080_pilot_train --array=4 \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
eval_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name=a4w0_2080_pilot_eval --array=4 \
    --dependency="afterok:${train_id}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
report_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name=a4w0_2080_pilot_report --dependency="afterok:${eval_id}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=report-wave0-2080ti-pilot" \
    "${ALIGNMENT_WAVE0_CPU_SBATCH}"
)"
echo "2080 Ti BF16 pilot: train=${train_id} eval=${eval_id} report=${report_id}"
echo "Do not launch the remaining 2080 Ti jobs until the timing report is reviewed."
