#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/phase2_env.sh"
phase2_resolve_python
exec "$PHASE2_PYTHON" -m src.phase2.pipeline --project-root "$ROOT" "$@"
