#!/usr/bin/env bash

alignment_v3_python_is_usable() {
  local python_path="$1"
  [[ -x "${python_path}" ]] || return 1
  timeout --foreground "${ALIGNMENT_V3_IMPORT_TIMEOUT_SECONDS:-90}" "${python_path}" - <<'PY' >/dev/null 2>&1
import importlib
for module in ("torch", "yaml", "open_clip", "timm", "transformers", "pandas", "PIL", "jupyter_client"):
    importlib.import_module(module)
PY
}

alignment_v3_python_diagnose() {
  local python_path="$1"
  echo "  ${python_path}:" >&2
  if [[ ! -x "${python_path}" ]]; then
    echo "    FAIL: not executable or not present" >&2
    return
  fi
  timeout --foreground "${ALIGNMENT_V3_IMPORT_TIMEOUT_SECONDS:-90}" "${python_path}" - <<'PY' 2>&1 | sed 's/^/    /' >&2
import importlib
import sys

print(f"executable={sys.executable}")
failed = False
for name in ("torch", "yaml", "open_clip", "timm", "transformers", "pandas", "PIL", "jupyter_client"):
    try:
        importlib.import_module(name)
        print(f"OK {name}")
    except BaseException as exc:
        failed = True
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
raise SystemExit(1 if failed else 0)
PY
}

alignment_v3_resolve_python() {
  local root="${ALIGNMENT_V3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local candidate="${ALIGNMENT_V3_PYTHON:-${ALIGNMENT_V3_SYSTEM_PYTHON:-}}"
  local values=(
    "${candidate}"
    "${root}/.venv/bin/python"
    "${root}/scripts/alignment_v3_aisurrey_python.sh"
    "${root}/.venv-aisurrey/bin/python"
  )
  local value
  if [[ "${ALIGNMENT_V3_SKIP_IMPORT_PROBE:-0}" == "1" && -n "${candidate}" && -x "${candidate}" ]]; then
    ALIGNMENT_V3_PYTHON="${candidate}"
    export ALIGNMENT_V3_PYTHON
    echo "alignment-v3: trusted explicit Python ${ALIGNMENT_V3_PYTHON}; runtime validation remains enabled" >&2
    return 0
  fi
  for value in "${values[@]}"; do
    [[ -n "${value}" ]] || continue
    if alignment_v3_python_is_usable "${value}"; then
      ALIGNMENT_V3_PYTHON="${value}"
      export ALIGNMENT_V3_PYTHON
      echo "alignment-v3: selected Python ${ALIGNMENT_V3_PYTHON}" >&2
      return 0
    fi
  done
  echo "No Python environment provides the complete Alignment v3 stack." >&2
  echo "Set ALIGNMENT_V3_SYSTEM_PYTHON to an environment containing torch, timm, transformers, open_clip, pandas, Pillow and jupyter_client." >&2
  echo "Candidate diagnostics:" >&2
  for value in "${values[@]}"; do
    [[ -n "${value}" ]] || continue
    alignment_v3_python_diagnose "${value}" || true
  done
  return 2
}
