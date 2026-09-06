#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PARTITION=teaching
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == teaching ]] || {
  echo 'FreezeShift recovery requires teaching/native-BF16 RTX PRO 6000 hardware.' >&2
  exit 2
}
[[ -f freezeshift/results/manifests/reporting_amendment.json ]] || {
  echo 'Missing reporting amendment receipt.' >&2
  exit 2
}
job_id="$(timeout --foreground 60 sbatch --parsable \
  --partition="${PARTITION}" \
  --job-name=fs_recover \
  --export="ALL,FREEZESHIFT_ROOT=${ROOT}" \
  freezeshift/slurm/resume_reporting.sbatch)"
echo "FreezeShift reporting recovery job: ${job_id}"
echo 'Runs arm reports, profiling and final report only. No training or epoch evaluation.'
