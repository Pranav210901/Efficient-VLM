#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec bash scripts/submit_alignment_v4_pipeline.sh --resume "$@"

