#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
mkdir -p artifacts/07_final_evaluation/compositional/{logs,manifests,per_run,report}
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final validate
evaluation="$(sbatch --parsable --array=0-8 slurm/compositional_final/evaluate.sbatch)"
report="$(sbatch --parsable --dependency="afterok:${evaluation}" slurm/compositional_final/report.sbatch)"
echo "compositional final: evaluation=${evaluation} report=${report}"
