#!/usr/bin/env bash
set -euo pipefail
echo "BLOCKED: accepted F runs must not be retrained. Use submit_alignment_v4_wave0_rescreen.sh after recipe lock." >&2
exit 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
source scripts/alignment_v4_wave0_partition.sh

MAX_GPUS=16
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
alignment_v4_wave0_partition_profile 2080ti "${MAX_GPUS}"
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
PILOT_REPORT="results/alignment_v4_wave0/wave0-queue-ablation/2080ti_pilot/report.json"

"${ALIGNMENT_V3_PYTHON}" - "${PILOT_REPORT}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"BLOCKED: missing reviewed timing pilot {path}")
payload = json.loads(path.read_text())
if not payload.get("within_four_x", False):
    raise SystemExit(f"BLOCKED: 2080 Ti pilot status is {payload.get('status')}")
print(
    "Timing gate: PASS; "
    f"mean={payload['mean_seconds_per_epoch']:.2f}s "
    f"ratio_vs_61s={payload['conservative_ratio_vs_61_seconds']:.2f}x"
)
PY

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY-RUN: 2080ti BF16 indices 5-12 (two queue doses + six pair re-screens)."
  exit 0
fi

for index in 1 2; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-queue-ablation --index "${index}"
done
for index in 0 1 2 3 4 5; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-rescreen --index "${index}"
done

train_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name=a4w0_2080_bulk_train --array="5-12%${ALIGNMENT_WAVE0_MAX_GPUS}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
eval_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name=a4w0_2080_bulk_eval --array="5-12%${ALIGNMENT_WAVE0_MAX_GPUS}" \
    --dependency="afterok:${train_id}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-followup" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
for command in report-wave0-queue-ablation report-wave0-rescreen; do
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=2080ti \
    --job-name="a4w0_${command#report-wave0-}" --dependency="afterok:${eval_id}" \
    --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=${command}" \
    "${ALIGNMENT_WAVE0_CPU_SBATCH}"
done
echo "2080 Ti BF16 bulk: train=${train_id} eval=${eval_id}"
echo "Recipe remains provisional; Wave 1 was not submitted."
