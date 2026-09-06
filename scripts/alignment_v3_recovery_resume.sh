#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec bash scripts/submit_alignment_v3_recovery_pipeline.sh --resume "$@"

