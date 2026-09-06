#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/phase2_env.sh"
phase2_resolve_python
PYTHON="$PHASE2_PYTHON"
MANIFEST="${1:-$ROOT/results/phase2/slurm/submission_manifest.json}"
test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
FIRST="$("$PYTHON" - "$MANIFEST" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1])); root=Path(sys.argv[2])
for row in p.get('stages',[]):
 d=root/row.get('status_path','')
 statuses=list(d.glob('*/status.json')) if d.exists() else []
 if not statuses or any(json.load(open(s)).get('status')!='COMPLETED' for s in statuses):
  print(row['stage']); break
PY
)"
if [[ -z "$FIRST" ]]; then echo "All recorded Phase 2 stages are complete."; exit 0; fi
echo "Resuming dependency chain from $FIRST"
exec bash "$ROOT/scripts/submit_phase2_pipeline.sh" --resume-from "$FIRST" --skip-completed --max-total-gpus 8
