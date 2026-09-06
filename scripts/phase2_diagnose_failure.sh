#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/phase2_env.sh"
phase2_resolve_python
PYTHON="$PHASE2_PYTHON"
VALUE="${1:-$ROOT/results/phase2/slurm/submission_manifest.json}"
if [[ "$VALUE" =~ ^[0-9]+$ ]]; then
  sacct -j "$VALUE" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
  find "$ROOT/logs/phase2" -type f -name "*${VALUE}*" -print -exec tail -n 80 {} \;
else
  test -f "$VALUE" || { echo "Missing manifest: $VALUE" >&2; exit 2; }
  "$PYTHON" -m src.phase2.cli.collect_failures
  cat "$ROOT/results/phase2/slurm/failure_report.md"
  find "$ROOT/results/phase2" -name traceback.txt -print -exec tail -n 80 {} \;
fi
echo "Resume recommendation: bash scripts/phase2_resume.sh"
