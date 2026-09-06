#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/text_aggregation_study/slurm results/text_aggregation_study/diagnostics/e5_latency
export ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
export ALIGNMENT_V3_SKIP_IMPORT_PROBE=1
export TEXT_AGG_ROOT="$ROOT"
job="$(timeout --foreground 60 sbatch --parsable \
  --partition=teaching \
  --job-name=ta_e5_diag \
  --export=ALL,TEXT_AGG_ROOT="$ROOT",ALIGNMENT_V3_PYTHON="$ALIGNMENT_V3_PYTHON",ALIGNMENT_V3_SKIP_IMPORT_PROBE=1 \
  slurm/text_aggregation_study/e5_latency_diagnostic.sbatch)"
echo "E5 latency diagnostic: ${job}"
echo "Report: results/text_aggregation_study/diagnostics/e5_latency/report.md"
