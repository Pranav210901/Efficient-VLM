#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/phase2_env.sh"
phase2_resolve_python
PYTHON="$PHASE2_PYTHON"
MANIFEST="${1:-$ROOT/results/phase2/slurm/submission_manifest.json}"
if [[ ! -f "$MANIFEST" ]]; then echo "Missing pipeline manifest: $MANIFEST" >&2; exit 2; fi
"$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print("stage\tjob_id\tfailure_job_id\tstatus_path\tlog_path")
for r in p.get("stages",[]): print("\t".join(str(r.get(k,"")) for k in ("stage","job_id","failure_job_id","status_path","log_path")))
PY
IDS="$("$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(','.join(str(r['job_id']) for r in p.get('stages',[]) if str(r['job_id']).isdigit()))
PY
)"
if [[ -n "$IDS" ]]; then
  squeue -j "$IDS" -o '%.18i %.40j %.10T %.10M %.6D %R' 2>/dev/null || true
  sacct -j "$IDS" --format=JobID,JobName,State,ExitCode,Elapsed 2>/dev/null || true
fi
"$PYTHON" -m src.phase2.cli.collect_failures
