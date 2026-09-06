#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/text_aggregation_study/slurm results/text_aggregation_study/diagnostics/m_t1_repeated_latency
export ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
export ALIGNMENT_V3_SKIP_IMPORT_PROBE=1
job="$(timeout --foreground 60 sbatch --parsable \
  --partition=teaching \
  --job-name=ta_m1_repeat \
  --export=ALL,TEXT_AGG_ROOT="$ROOT",ALIGNMENT_V3_PYTHON="$ALIGNMENT_V3_PYTHON",ALIGNMENT_V3_SKIP_IMPORT_PROBE=1 \
  slurm/text_aggregation_study/repeated_latency.sbatch)"
echo "M_T1 paired repeated latency diagnostic: ${job}"
echo "Report: ${ROOT}/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/report.md"
