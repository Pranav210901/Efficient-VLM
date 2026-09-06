#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/efficiency_frontier/pipeline.yaml"
validation="$("${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier validate --pipeline "${PIPELINE}")"
echo "${validation}"
if grep -q 'READY_WITH_EXPECTED_MISSING_V4_SEEDS' <<<"${validation}"; then
  echo "BLOCKED: first complete historical-v4 seeds 43/44 with scripts/submit_efficiency_frontier_v4_missing.sh" >&2
  exit 3
fi

read -r max_tasks eligible_nodes < <(
  sinfo -h -p teaching -N -o '%N|%G|%c' |
  awk -F'|' '
    /gpu:nvidia_rtxpro6000:/ {
      gpus=0
      if (match($2, /gpu:nvidia_rtxpro6000:([0-9]+)/, a)) gpus=a[1]+0
      cpu_tasks=int(($3+0)/12)
      slots=(gpus < cpu_tasks ? gpus : cpu_tasks)
      total+=slots
      nodes+=1
    }
    END { print total+0, nodes+0 }
  '
)
if [[ "${max_tasks}" -lt 1 ]]; then
  echo "BLOCKED: no teaching RTX PRO 6000 slots compatible with 12 CPUs/task." >&2
  exit 2
fi
echo "Teaching RTX PRO 6000 eligible nodes: ${eligible_nodes}"
echo "Maximum supported concurrent 1-GPU/12-CPU tasks: ${max_tasks}"
concurrency="${max_tasks}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: provenance gate, 12 evaluation jobs and 9 profiling jobs."
  echo "DRY-RUN: evaluation and profiling use separate allocations; concurrency=${concurrency}."
  echo "DRY-RUN: aggregate/report depends on both arrays."
  exit 0
fi

manifest_id="$(timeout --foreground 60 sbatch --parsable \
  --job-name=ef_manifest \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=manifest" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
eval_id="$(timeout --foreground 60 sbatch --parsable \
  --array="0-11%${concurrency}" --dependency="afterok:${manifest_id}" --job-name=ef_eval \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=evaluate" \
  slurm/efficiency_frontier/gpu_eval.sbatch)"
profile_id="$(timeout --foreground 60 sbatch --parsable \
  --array="0-8%${concurrency}" --dependency="afterok:${manifest_id}" --job-name=ef_profile \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=profile" \
  slurm/efficiency_frontier/gpu_profile.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${eval_id}:${profile_id}" --job-name=ef_report \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=aggregate" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
echo "Efficiency frontier: manifest=${manifest_id} eval=${eval_id} profile=${profile_id} report=${report_id}"
