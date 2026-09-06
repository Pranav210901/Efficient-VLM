#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/efficiency_frontier/pipeline.yaml"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier validate \
  --pipeline "${PIPELINE}" >/dev/null
[[ -f results/efficiency_frontier/controls/mobileclip_fusion_profile.json ]] || {
  echo "BLOCKED: run the MobileCLIP fusion profile first." >&2
  exit 2
}
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: one fused MobileCLIP2 retrieval evaluation."
  echo "DRY-RUN: Flickr30k test + contaminated COCO validation; no training."
  exit 0
fi
job_id="$(timeout --foreground 60 sbatch --parsable \
  --job-name=ef_mc_fused_eval \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=evaluate-mobileclip-fusion" \
  slurm/efficiency_frontier/gpu_profile.sbatch)"
echo "MobileCLIP2 fused retrieval evaluation: job=${job_id}"
echo "After completion: cat results/efficiency_frontier/controls/mobileclip_fusion_evaluation.json"
