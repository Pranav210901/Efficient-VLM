#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PARTITION=teaching
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == teaching ]] || {
  echo 'The latency amendment requires teaching/native-BF16 RTX PRO 6000 hardware.' >&2
  exit 2
}
mkdir -p latency_amendment/{logs/slurm,results/manifests,results/report}
"${ALIGNMENT_V3_PYTHON}" -m latency_amendment.runner validate
if [[ "${DRY_RUN}" == true ]]; then
  echo "DRY-RUN sbatch --partition=${PARTITION} --job-name=fs_lat_amend latency_amendment/slurm/profile.sbatch"
  exit 0
fi
job_id="$(timeout --foreground 60 sbatch --parsable \
  --partition="${PARTITION}" \
  --job-name=fs_lat_amend \
  --export="ALL,LATENCY_AMENDMENT_ROOT=${ROOT}" \
  latency_amendment/slurm/profile.sbatch)"
echo "FreezeShift latency amendment submitted: ${job_id}"
echo 'One profiling GPU job; no training, evaluation, or checkpoint writes.'

