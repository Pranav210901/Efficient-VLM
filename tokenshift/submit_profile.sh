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
  echo 'TokenShift requires teaching/native-BF16 RTX PRO 6000 hardware.' >&2
  exit 2
}
mkdir -p tokenshift/{logs/slurm,results/manifests,results/profile}
"${ALIGNMENT_V3_PYTHON}" -m tokenshift.runner validate
"${ALIGNMENT_V3_PYTHON}" -m tokenshift.runner smoke
if [[ "${DRY_RUN}" == true ]]; then
  echo "DRY-RUN sbatch --partition=${PARTITION} --job-name=ts_profile tokenshift/slurm/profile.sbatch"
  exit 0
fi
job_id="$(timeout --foreground 60 sbatch --parsable \
  --partition="${PARTITION}" \
  --job-name=ts_profile \
  --export="ALL,TOKENSHIFT_ROOT=${ROOT}" \
  tokenshift/slurm/profile.sbatch)"
echo "TokenShift profile submitted: ${job_id}"
echo 'One GPU job only; no training is submitted. Flickr test remains sealed.'

