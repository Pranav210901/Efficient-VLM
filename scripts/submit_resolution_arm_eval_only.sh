#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=6
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
  echo "Resolution evaluation requires teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-6]$ ]] || {
  echo "--max-total-gpus must be in [1,6]." >&2
  exit 2
}

PIPELINE=configs/resolution_arm/pipeline.yaml
mkdir -p logs/resolution_arm/slurm

# All training and the first recovery's four quiet-GPU profiles must already
# be complete. This launcher cannot retrain or reprofile.
"${ALIGNMENT_V3_PYTHON}" - <<'PY'
from pathlib import Path
checkpoint_root = Path("checkpoints/resolution_arm/resolution")
profile_root = Path("results/resolution_arm/profiling/per_run")
checkpoint_runs = [
    f"resolution_{resolution}__seed_{seed}"
    for resolution in (224, 192)
    for seed in (42, 43, 44)
]
profiles = [
    profile_root / f"resolution_{resolution}" / f"seed_{seed}" / "profile_dynamic_padding.json"
    for resolution in (224, 192)
    for seed in (42, 43)
]
missing_checkpoints = [
    run for run in checkpoint_runs
    if not (checkpoint_root / run / "best.pt").is_file()
]
missing_profiles = [str(path) for path in profiles if not path.is_file()]
if missing_checkpoints or missing_profiles:
    raise SystemExit(
        "evaluation-only gate failed; wait for profiles to finish: "
        f"checkpoints={missing_checkpoints}, profiles={missing_profiles}"
    )
print("Resolution evaluation-only gate: PASS (6 checkpoints, 4 profiles)")
PY

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "95${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

evaluation="$(submit \
  --partition="${PARTITION}" \
  --job-name=res_eval_final \
  --array="0-5%${MAX_GPUS}" \
  --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_KIND=module,RESOLUTION_ARM_COMMAND=evaluate" \
  slurm/resolution_arm/gpu_stage.sbatch)"
report="$(submit \
  --partition="${PARTITION}" \
  --job-name=res_report_final \
  --dependency="afterok:${evaluation}" \
  --export="ALL,RESOLUTION_ARM_ROOT=${ROOT},RESOLUTION_ARM_PIPELINE=${PIPELINE},RESOLUTION_ARM_COMMAND=report" \
  slurm/resolution_arm/cpu_stage.sbatch)"

echo "Resolution evaluation-only recovery: eval=${evaluation} report=${report}"
echo "No training or profiling jobs were submitted."
