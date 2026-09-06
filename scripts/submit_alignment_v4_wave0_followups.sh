#!/usr/bin/env bash
set -euo pipefail

echo "BLOCKED: the 13 follow-ups must not run on one partition." >&2
echo "Use submit_alignment_v4_wave0_lr_controls.sh for teaching C/D," >&2
echo "then submit_alignment_v4_wave0_2080ti_pilot.sh for one timing pilot." >&2
exit 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
source scripts/alignment_v4_wave0_partition.sh

PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
PARTITION="teaching"
MAX_GPUS=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
alignment_v4_wave0_partition_profile "${PARTITION}" "${MAX_GPUS}"
PARTITION="${ALIGNMENT_WAVE0_PARTITION}"
MAX_GPUS="${ALIGNMENT_WAVE0_MAX_GPUS}"

# C/D remain covered by the original pre-run prediction. F and the pair
# re-screen deliberately have no prediction, as approved by the researcher.
"${ALIGNMENT_V3_PYTHON}" - "${PIPELINE}" <<'PY'
import json
import sys
from pathlib import Path

from src.alignment_v3.runner import (
    load_pipeline,
    wave0_followup_jobs,
    wave0_lr_control_jobs,
    wave0_queue_ablation_jobs,
    wave0_rescreen_jobs,
)

_, pipeline = load_pipeline(sys.argv[1])
path = Path(pipeline["wave0"]["predictions_path"])
if not path.is_file():
    raise SystemExit(f"BLOCKED: missing prediction file {path}")
payload = json.loads(path.read_text())
predictions = payload.get("predictions", [])
if any(not str(value.get("statement", "")).strip() for value in predictions):
    raise SystemExit("BLOCKED: every prediction statement must be non-empty")
if "wave0_winner" not in {str(value.get("id")) for value in predictions}:
    raise SystemExit("BLOCKED: wave0_winner prediction is required for C/D")
if pipeline["wave0"].get("lr_control_status") != "PROVISIONAL_PENDING_LR_CONTROL":
    raise SystemExit("BLOCKED: recipe is not PROVISIONAL_PENDING_LR_CONTROL")
counts = (
    len(wave0_lr_control_jobs(pipeline)),
    len(wave0_queue_ablation_jobs(pipeline)),
    len(wave0_rescreen_jobs(pipeline)),
    len(wave0_followup_jobs(pipeline)),
)
if counts != (4, 3, 6, 13):
    raise SystemExit(f"BLOCKED: expected C/D/F/re-screen counts (4,3,6,13), got {counts}")
print(f"Prediction/config gate: PASS ({path}); counts={counts}")
PY

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY-RUN: partition=${PARTITION} precision=${ALIGNMENT_WAVE0_PRECISION} workers=${ALIGNMENT_WAVE0_NUM_WORKERS}"
  echo "DRY-RUN: GPU resources cpus=${ALIGNMENT_WAVE0_GPU_CPUS} mem=${ALIGNMENT_WAVE0_GPU_MEM}"
  echo "DRY-RUN: one train array 0-12%${MAX_GPUS} (C/D=4, F=3, re-screen=6)"
  echo "DRY-RUN: one dependent eval array 0-12%${MAX_GPUS}"
  echo "DRY-RUN: dependent queue-dose and pair-ranking CPU reports"
  echo "No recipe-selection or Wave 1 job would be submitted."
  exit 0
fi

for index in 0 1 2 3; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-lr-control --index "${index}"
done
for index in 0 1 2; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-queue-ablation --index "${index}"
done
for index in 0 1 2 3 4 5; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-rescreen --index "${index}"
done

train_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --job-name=a4w0_followup_train --array="0-12%${MAX_GPUS}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
eval_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --job-name=a4w0_followup_eval --array="0-12%${MAX_GPUS}" \
    --dependency="afterok:${train_id}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
queue_report_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
    --job-name=a4w0_queue_report --dependency="afterok:${eval_id}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=report-wave0-queue-ablation" \
    "${ALIGNMENT_WAVE0_CPU_SBATCH}"
)"
rescreen_report_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
    --job-name=a4w0_rescreen_report --dependency="afterok:${eval_id}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=report-wave0-rescreen" \
    "${ALIGNMENT_WAVE0_CPU_SBATCH}"
)"

echo "Prepared follow-ups: train=${train_id} eval=${eval_id}"
echo "Reports: queue=${queue_report_id} re-screen=${rescreen_report_id}"
echo "Recipe remains PROVISIONAL_PENDING_LR_CONTROL; Wave 1 was not submitted."
