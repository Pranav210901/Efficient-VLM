#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for VENV_ROOT in \
  "${ROOT}/.venv-aisurrey" \
  "${ROOT}/quarantine/github_submission_cleanup_20260810/original_paths/.venv-aisurrey" \
  "${ROOT}/.venv" \
  "${ROOT}/quarantine/github_submission_cleanup_20260810/original_paths/.venv"; do
  [[ -f "${VENV_ROOT}/pyvenv.cfg" ]] || continue
  VERSION="$(sed -n 's/^version = \([0-9]*\.[0-9]*\).*/\1/p' "${VENV_ROOT}/pyvenv.cfg" | head -1)"
  SITE_PACKAGES="${VENV_ROOT}/lib/python${VERSION}/site-packages"
  [[ -d "${SITE_PACKAGES}" ]] || continue
  BASE_PYTHON="$(sed -n 's/^executable = //p' "${VENV_ROOT}/pyvenv.cfg" | head -1)"
  [[ -x "${BASE_PYTHON}" ]] || BASE_PYTHON="$(command -v "python${VERSION}" || true)"
  [[ -x "${BASE_PYTHON}" ]] || continue
  export VIRTUAL_ENV="${VENV_ROOT}"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${BASE_PYTHON}" "$@"
done

echo "No complete project Python environment was found (including quarantine)." >&2
exit 2
