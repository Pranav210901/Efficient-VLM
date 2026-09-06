#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/efficiency_frontier/pipeline.yaml"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier validate --pipeline "${PIPELINE}" >/dev/null
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: 9 dynamic-padding profiling tasks; no training or retrieval evaluation."
  echo "DRY-RUN: report writes separate dynamic-padding artifacts and preserves profile.json."
  exit 0
fi
profile_id="$(timeout --foreground 60 sbatch --parsable \
  --array=0-8 --job-name=ef_dynpad \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=profile-dynamic" \
  slurm/efficiency_frontier/gpu_profile.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${profile_id}" --job-name=ef_dynpad_report \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=report-dynamic" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
echo "Dynamic-padding re-profile: profile=${profile_id} report=${report_id}"
