#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/alignment_v3_env.sh
# The direct AISurrey venv works on the submit node. Recent successful
# Blackwell jobs (through 2026-07-30) used .venv/bin/python; that environment
# is therefore the evidence-backed compute interpreter. Keep the two roles
# deliberately separate because /bin/python differs across hosts.
TEXT_AGG_SUBMIT_PYTHON="${ROOT}/.venv-aisurrey/bin/python"
ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
ALIGNMENT_V3_SKIP_IMPORT_PROBE=1
export ALIGNMENT_V3_PYTHON ALIGNMENT_V3_SKIP_IMPORT_PROBE
[[ -x "${TEXT_AGG_SUBMIT_PYTHON}" ]] || {
  echo "Missing submit-node Python: ${TEXT_AGG_SUBMIT_PYTHON}" >&2
  exit 2
}
echo "text-aggregation: submit validation Python ${TEXT_AGG_SUBMIT_PYTHON}" >&2
echo "text-aggregation: compute Python ${ALIGNMENT_V3_PYTHON}" >&2

GPU_PARTITION=teaching
CPU_PARTITION=""
MAX_GPUS=8
DRY_RUN=false
PROFILE_ONLY=false
PROFILE_JOB_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-partition) GPU_PARTITION="$2"; shift 2 ;;
    --cpu-partition) CPU_PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --profile-only) PROFILE_ONLY=true; shift ;;
    --profile-job-id) PROFILE_JOB_ID="$2"; shift 2 ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$GPU_PARTITION" == teaching ]] || { echo "Profiling/training requires teaching RTX PRO 6000 Blackwell." >&2; exit 2; }
# AI Surrey exposes no separate CPU-only partition and teaching allocations
# require a GPU before CPUs can be granted. Gate/audit/report jobs therefore
# request one GPU even though their code remains CPU-only.
[[ -n "$CPU_PARTITION" ]] || CPU_PARTITION="$GPU_PARTITION"
[[ "$MAX_GPUS" =~ ^[1-8]$ ]] || { echo "--max-total-gpus must be 1-8" >&2; exit 2; }

mkdir -p logs/text_aggregation_study/slurm results/text_aggregation_study/{manifests,profiling,gates/minilm,gates/e5,report}
echo "text-aggregation: validating frozen design..." >&2
timeout --foreground 180 "${TEXT_AGG_SUBMIT_PYTHON}" \
  -m src.alignment_v3.text_aggregation_study validate
echo "text-aggregation: validation PASS; submitting dependency graph..." >&2

submit() {
  if [[ "$DRY_RUN" == true ]]; then echo "DRY-RUN sbatch $*" >&2; echo "98${RANDOM}"; else timeout --foreground 60 sbatch --parsable "$@"; fi
}
export_base="ALL,TEXT_AGG_ROOT=${ROOT}"
if [[ -n "$PROFILE_JOB_ID" ]]; then
  [[ "$PROFILE_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "--profile-job-id must be numeric" >&2
    exit 2
  }
  profile="$PROFILE_JOB_ID"
  echo "text-aggregation: reusing completed profile job ${profile}" >&2
else
  profile="$(submit --partition="$GPU_PARTITION" --job-name=ta_profile --export="$export_base" slurm/text_aggregation_study/profile.sbatch)"
fi
if [[ "$PROFILE_ONLY" == true ]]; then
  echo "Text aggregation profile-only: profile=${profile}"
  echo "After it completes successfully, rerun with --profile-job-id ${profile}."
  exit 0
fi
mgate="$(submit --partition="$CPU_PARTITION" --job-name=ta_m_gate --dependency="afterok:${profile}" --export="$export_base,TEXT_AGG_ENCODER=minilm" slurm/text_aggregation_study/gate.sbatch)"
egate="$(submit --partition="$CPU_PARTITION" --job-name=ta_e_gate --dependency="afterok:${profile}" --export="$export_base,TEXT_AGG_ENCODER=e5" slurm/text_aggregation_study/gate.sbatch)"

submit_branch() {
  local branch="$1" dependency="$2" short="$3"
  local train verify eval prof report
  train="$(submit --partition="$GPU_PARTITION" --job-name="ta_${short}_tr" --array="0-2%3" --dependency="afterok:${dependency}" --export="$export_base,TEXT_AGG_BRANCH=${branch}" slurm/text_aggregation_study/train.sbatch)"
  verify="$(submit --partition="$CPU_PARTITION" --job-name="ta_${short}_ck" --dependency="afterok:${train}" --export="$export_base,TEXT_AGG_BRANCH=${branch},TEXT_AGG_COMMAND=verify-snapshots" slurm/text_aggregation_study/report.sbatch)"
  eval="$(submit --partition="$GPU_PARTITION" --job-name="ta_${short}_ev" --array="0-71%${MAX_GPUS}" --dependency="afterok:${verify}" --export="$export_base,TEXT_AGG_BRANCH=${branch}" slurm/text_aggregation_study/evaluate.sbatch)"
  prof="$(submit --partition="$GPU_PARTITION" --job-name="ta_${short}_pf" --dependency="afterok:${train}" --export="$export_base,TEXT_AGG_BRANCH=${branch}" slurm/text_aggregation_study/profile_trained.sbatch)"
  report="$(submit --partition="$CPU_PARTITION" --job-name="ta_${short}_rp" --dependency="afterok:${eval}:${prof}" --export="$export_base,TEXT_AGG_BRANCH=${branch},TEXT_AGG_COMMAND=report" slurm/text_aggregation_study/report.sbatch)"
  echo "${branch}: train=${train} verify=${verify} eval=${eval} profile=${prof} report=${report}"
}

echo "profile=${profile} minilm_gate=${mgate} e5_gate=${egate}"
submit_branch minilm_aggregation "$mgate" ma
submit_branch e5_baseline "$egate" e0
submit_branch e5_aggregation "$egate" ea
echo "Flickr TEST remains sealed. Failed gates intentionally block only their own dependent branch."
