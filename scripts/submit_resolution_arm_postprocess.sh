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
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "Resolution post-processing requires teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

PIPELINE=configs/resolution_arm/pipeline.yaml
mkdir -p logs/resolution_arm/slurm

# Hard gate: recovery must never silently retrain or proceed from an incomplete
# checkpoint set.
"${ALIGNMENT_V3_PYTHON}" - <<'PY'
from pathlib import Path
required = ("best.pt", "config.yaml", "fingerprint.json", "run_summary.json")
root = Path("checkpoints/resolution_arm/resolution")
runs = [
    f"resolution_{resolution}__seed_{seed}"
    for resolution in (224, 192)
    for seed in (42, 43, 44)
]
failures = {
    run: [name for name in required if not (root / run / name).is_file()]
    for run in runs
}
failures = {run: missing for run, missing in failures.items() if missing}
if failures:
    raise SystemExit(f"incomplete resolution checkpoints; refusing recovery: {failures}")
print("Resolution checkpoint gate: PASS (6/6); post-processing only")
PY

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "96${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}
gpu() {
  local name="$1" command="$2" count="$3"
  submit \
    --partition="${PARTITION}" \
    --job-name="${name}" \
    --array="0-$((count - 1))%${MAX_GPUS}" \
    --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_KIND=module,RESOLUTION_ARM_COMMAND=${command}" \
    slurm/resolution_arm/gpu_stage.sbatch
}

evaluation="$(gpu res_eval_recovery evaluate 6)"
profile="$(gpu res_profile_recovery profile 4)"
report="$(submit \
  --partition="${PARTITION}" \
  --job-name=res_report_recovery \
  --dependency="afterok:${evaluation}:${profile}" \
  --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_COMMAND=report" \
  slurm/resolution_arm/cpu_stage.sbatch)"

echo "Resolution post-processing only: eval=${evaluation} profile=${profile} report=${report}"
echo "No training jobs were submitted by this graph."
