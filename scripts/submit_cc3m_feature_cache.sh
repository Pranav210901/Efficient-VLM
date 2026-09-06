#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PIPELINE="configs/cc3m_scale/pipeline.yaml"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: one teaching RTX PRO 6000 job; deterministic post-training cache only."
  exit 0
fi
if [[ ! -f results/cc3m_scale/report.json ]]; then
  echo "BLOCKED: both CC3M arms must complete before feature caching." >&2
  exit 2
fi
job_id="$(timeout --foreground 60 sbatch --parsable \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/feature_cache.sbatch)"
echo "CC3M post-training feature cache: ${job_id}"
