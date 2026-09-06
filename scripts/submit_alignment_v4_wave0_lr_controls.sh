#!/usr/bin/env bash
set -euo pipefail

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
if [[ "${PARTITION}" != "teaching" ]]; then
  echo "BLOCKED: C/D controls must run on teaching with native BF16." >&2
  exit 2
fi

# Keep the original submit-time prediction gate. These controls test the
# pre-registered wave0_winner prediction; no post-result prediction is added.
"${ALIGNMENT_V3_PYTHON}" - "${PIPELINE}" <<'PY'
import json
import sys
from pathlib import Path

from src.utils.config import load_config

pipeline = load_config(sys.argv[1])
path = Path(pipeline["wave0"]["predictions_path"])
if not path.is_file():
    raise SystemExit(f"BLOCKED: missing prediction file {path}")
payload = json.loads(path.read_text())
values = payload.get("predictions", [])
if any(not str(value.get("statement", "")).strip() for value in values):
    raise SystemExit("BLOCKED: every prediction statement must be non-empty")
if "wave0_winner" not in {str(value.get("id")) for value in values}:
    raise SystemExit("BLOCKED: wave0_winner prediction is required")
if pipeline["wave0"].get("lr_control_status") != "PROVISIONAL_PENDING_LR_CONTROL":
    raise SystemExit("BLOCKED: recipe is not PROVISIONAL_PENDING_LR_CONTROL")
print(f"Prediction gate: PASS ({path})")
PY

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY-RUN partition=${PARTITION} precision=${ALIGNMENT_WAVE0_PRECISION} workers=${ALIGNMENT_WAVE0_NUM_WORKERS}"
  echo "DRY-RUN sbatch --partition=${PARTITION} --array=0-3%${MAX_GPUS} train-wave0-lr-control"
  echo "DRY-RUN sbatch --partition=${PARTITION} --array=0-3%${MAX_GPUS} eval-wave0-lr-control afterok:<train_job>"
  echo "No recipe-selection or Wave 1 job would be submitted."
  exit 0
fi

for index in 0 1 2 3; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-lr-control --index "${index}"
done

train_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --job-name=a4w0_lr_control_train --array="0-3%${MAX_GPUS}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-lr-control" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
eval_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition="${PARTITION}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --job-name=a4w0_lr_control_eval --array="0-3%${MAX_GPUS}" \
    --dependency="afterok:${train_id}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-lr-control" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"

echo "Submitted C/D only: train=${train_id} eval=${eval_id}"
echo "Recipe remains PROVISIONAL_PENDING_LR_CONTROL; Wave 1 was not submitted."
