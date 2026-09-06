#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v4_wave0_partition.sh
alignment_v4_wave0_partition_profile teaching 6
ALIGNMENT_WAVE0_GPU_MEM="96G"
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
SUBMIT_PYTHON="${ROOT}/.venv/bin/python"
ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${ALIGNMENT_V3_PYTHON}" ]]; then
  echo "BLOCKED: expected Python is not executable: ${ALIGNMENT_V3_PYTHON}" >&2
  exit 2
fi
export ALIGNMENT_V3_PYTHON

"${SUBMIT_PYTHON}" - <<'PY'
import json
from pathlib import Path
p = Path("results/alignment_v4_wave0/selection/recipe.json")
v = json.loads(p.read_text()) if p.is_file() else {}
if v.get("status") != "SELECTED_WAVE0_RECIPE":
    raise SystemExit("BLOCKED: Wave 0 recipe is not locked.")
r = v.get("recipe", {})
expected = ("infonce_no_queue", None, 1024, 3.0)
actual = (r.get("loss_type"), r.get("captions_per_image"), r.get("batch_size"), r.get("sweep_multiplier"))
if actual != expected:
    raise SystemExit(f"BLOCKED: locked recipe mismatch: {actual}")
PY
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: six teaching/RTX PRO 6000/native-BF16 pair re-screen jobs."
  exit 0
fi
for index in 0 1 2 3 4 5; do
  "${SUBMIT_PYTHON}" -m src.alignment_v3.runner ledger-pending \
    --pipeline "${PIPELINE}" --ledger-stage wave0-rescreen --index "${index}"
done
train_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-5%6" --job-name=a4w0_rescreen_train \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-rescreen" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"
eval_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-5%6" --dependency="afterok:${train_id}" --job-name=a4w0_rescreen_eval \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-rescreen" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"
report_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${eval_id}" --job-name=a4w0_rescreen_report \
  --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=report-wave0-rescreen" \
  "${ALIGNMENT_WAVE0_CPU_SBATCH}")"
echo "Pair re-screen: train=${train_id} eval=${eval_id} report=${report_id}"
