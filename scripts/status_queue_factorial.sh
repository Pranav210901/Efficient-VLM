#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -n "${ALIGNMENT_V3_SYSTEM_PYTHON:-}" ]]; then
  python_bin="${ALIGNMENT_V3_SYSTEM_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  python_bin="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  python_bin="${ROOT}/.venv/bin/python"
else
  python_bin="${ROOT}/.venv-aisurrey/bin/python"
fi
"${python_bin}" -m src.alignment_v3.queue_factorial validate
find results/queue_factorial -name metrics.json -print 2>/dev/null | sort
squeue -u "${USER}" -o '%.18i %.12P %.30j %.2t %.10M %.30R' | rg 'qf_|JOBID' || true
