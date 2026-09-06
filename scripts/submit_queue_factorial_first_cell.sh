#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -n "${ALIGNMENT_V3_SYSTEM_PYTHON:-}" ]]; then
  PYTHON="${ALIGNMENT_V3_SYSTEM_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${ROOT}/.venv-aisurrey/bin/python"
fi
echo "Queue factorial submit Python: ${PYTHON}"
"${PYTHON}" -m src.alignment_v3.queue_factorial validate
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
path = Path("results/queue_factorial/smoke/report.json")
if not path.is_file():
    raise SystemExit("BLOCKED: timing_v2 smoke report is missing.")
report = json.loads(path.read_text())
if (
    report.get("status") != "COMPLETE"
    or report.get("classification") != "NON_OOM"
    or report.get("instrumentation_version") != "timing_v2"
):
    raise SystemExit(f"BLOCKED: timing_v2 smoke gate did not pass: {report}")
print("timing_v2 smoke gate: PASS")
PY
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: first cell only: b1024/capacity65536/both/seed42, 12 epochs."
  exit 0
fi
train_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --gres=gpu:nvidia_rtxpro6000:1 --job-name=qf_first_cell \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=first-cell" \
  slurm/queue_factorial/gpu_stage.sbatch)"
gate_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${train_id}" --job-name=qf_first_gate \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=gate-report" \
  slurm/queue_factorial/cpu_stage.sbatch)"
echo "Queue-factorial first cell: train=${train_id} gate=${gate_id}"

