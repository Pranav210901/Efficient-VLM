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
    raise SystemExit("BLOCKED: queue-factorial smoke report is missing.")
report = json.loads(path.read_text())
if (
    report.get("status") != "COMPLETE"
    or report.get("classification") != "NON_OOM"
    or report.get("instrumentation_version") != "timing_v2"
):
    raise SystemExit(f"BLOCKED: smoke did not close NON_OOM: {report}")
print("Smoke gate: PASS; inspect headroom in", path)
PY
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
path = Path("results/queue_factorial/gate/report.json")
if not path.is_file():
    raise SystemExit("BLOCKED: first-cell gate report is missing.")
report = json.loads(path.read_text())
if report.get("verdict") != "PASS" or report.get("remaining_cells_unblocked") is not True:
    raise SystemExit(
        "BLOCKED: first-cell gate must be PASS with remaining cells unblocked; "
        f"observed {report.get('verdict')!r}."
    )
print(
    "First-cell gate: PASS; image_ratio=",
    report["image"]["ratio"],
    "text_ratio=",
    report["text"]["ratio"],
)
PY

read -r max_gpus eligible_nodes < <(
  sinfo -h -p teaching -N -o '%N|%G|%c' |
  awk -F'|' '
    /gpu:nvidia_rtxpro6000:/ {
      gpus=0
      if (match($2, /gpu:nvidia_rtxpro6000:([0-9]+)/, a)) gpus=a[1]+0
      cpu_tasks=int(($3+0)/12)
      slots=(gpus < cpu_tasks ? gpus : cpu_tasks)
      total+=slots
      nodes+=1
    }
    END { print total+0, nodes+0 }
  '
)
if [[ "${max_gpus}" -lt 1 || "${eligible_nodes}" -lt 1 ]]; then
  echo "BLOCKED: could not find teaching RTX PRO 6000 capacity compatible with 12 CPUs/task." >&2
  exit 2
fi
echo "Teaching RTX PRO 6000 nodes discovered: ${eligible_nodes}"
echo "Maximum concurrent tasks after GPU and 12-CPU/task limits: ${max_gpus}"
remaining_count="$("${PYTHON}" - <<'PY'
from src.alignment_v3.queue_factorial import load_pipeline, remaining_jobs
p = load_pipeline("configs/queue_factorial/pipeline.yaml")
print(len(remaining_jobs(p)))
PY
)"
eval_count="$("${PYTHON}" - <<'PY'
from src.alignment_v3.queue_factorial import load_pipeline, pending_jobs
p = load_pipeline("configs/queue_factorial/pipeline.yaml")
print(len(pending_jobs(p)))
PY
)"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: ${remaining_count} remaining training tasks at concurrency ${max_gpus}; then ${eval_count} evaluations."
  exit 0
fi
last=$((remaining_count - 1))
eval_last=$((eval_count - 1))
train_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --gres=gpu:nvidia_rtxpro6000:1 \
  --array="0-${last}%${max_gpus}" --job-name=qf_train \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=remaining" \
  slurm/queue_factorial/gpu_stage.sbatch)"
eval_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --gres=gpu:nvidia_rtxpro6000:1 \
  --array="0-${eval_last}%${max_gpus}" --dependency="afterok:${train_id}" \
  --job-name=qf_eval \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=eval" \
  slurm/queue_factorial/gpu_stage.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${eval_id}" --job-name=qf_report \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=report" \
  slurm/queue_factorial/cpu_stage.sbatch)"
echo "Queue factorial: train=${train_id} eval=${eval_id} report=${report_id}"
