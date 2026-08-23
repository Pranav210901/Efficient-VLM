#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ALIGNMENT_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "No project Python found at ${PYTHON}. Create one with:" >&2
  echo "  python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt" >&2
  echo "or set ALIGNMENT_PYTHON to an interpreter with the pinned dependencies." >&2
  exit 2
fi
exec "${PYTHON}" "$@"
