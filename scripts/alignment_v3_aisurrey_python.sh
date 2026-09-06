#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${ROOT}/.venv-aisurrey"
CONFIG="${VENV_ROOT}/pyvenv.cfg"

[[ -f "${CONFIG}" ]] || {
  echo "Missing ${CONFIG}" >&2
  exit 2
}

BASE_PYTHON="$(sed -n 's/^executable = //p' "${CONFIG}" | head -1)"
if [[ -z "${BASE_PYTHON}" || ! -x "${BASE_PYTHON}" ]]; then
  BASE_PYTHON="$(command -v python3.11 || true)"
fi
[[ -n "${BASE_PYTHON}" && -x "${BASE_PYTHON}" ]] || {
  echo "AISurrey Python 3.11 is unavailable on this host." >&2
  exit 2
}

SITE_PACKAGES="${VENV_ROOT}/lib/python3.11/site-packages"
[[ -d "${SITE_PACKAGES}" ]] || {
  echo "Missing AISurrey site-packages: ${SITE_PACKAGES}" >&2
  exit 2
}

export VIRTUAL_ENV="${VENV_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BASE_PYTHON}" "$@"

