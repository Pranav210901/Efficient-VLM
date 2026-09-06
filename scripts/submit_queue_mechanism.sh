#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
mkdir -p logs/queue_mechanism/slurm results/queue_mechanism/{manifests,runs,report}
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism validate

stage="${1:-}"
case "${stage}" in
  capacity) indices="0-5" ;;
  mechanism) indices="6-14" ;;
  *)
    echo "Usage: bash scripts/submit_queue_mechanism.sh {capacity|mechanism}" >&2
    exit 2
    ;;
esac

array_job="$(sbatch --parsable --array="${indices}" slurm/queue_mechanism/train.sbatch)"
report_job="$(sbatch --parsable --dependency="afterany:${array_job}" slurm/queue_mechanism/report.sbatch)"
echo "queue mechanism ${stage}: array=${array_job} report=${report_job}"
