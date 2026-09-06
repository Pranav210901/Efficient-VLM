#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/efficiency_frontier/pipeline.yaml"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier validate \
  --pipeline "${PIPELINE}" >/dev/null
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: one paired MobileCLIP2 unfused/fused profiling task."
  echo "DRY-RUN: no training, retrieval evaluation, or frontier overwrite."
  exit 0
fi
job_id="$(timeout --foreground 60 sbatch --parsable \
  --job-name=ef_mc_fusion \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=profile-mobileclip-fusion" \
  slurm/efficiency_frontier/gpu_profile.sbatch)"
echo "MobileCLIP2 fusion re-profile: job=${job_id}"
echo "After completion: cat results/efficiency_frontier/controls/mobileclip_fusion_profile.json"
