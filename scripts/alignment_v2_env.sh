#!/usr/bin/env bash

alignment_v2_python_is_usable() {
  local python_path="$1"
  local import_timeout="${ALIGNMENT_V2_IMPORT_TIMEOUT_SECONDS:-60}"
  [[ -x "${python_path}" ]] || return 1
  echo "alignment-v2: checking Python environment ${python_path} (timeout ${import_timeout}s)" >&2
  timeout --foreground "${import_timeout}" "${python_path}" - <<'PY' >/dev/null 2>&1
import importlib

required = (
    "torch",
    "yaml",
    "open_clip",
    "timm",
    "transformers",
    "pandas",
    "PIL",
)
for module in required:
    importlib.import_module(module)
PY
}

alignment_v2_explain_python_failure() {
  local python_path="$1"
  local import_timeout="${ALIGNMENT_V2_IMPORT_TIMEOUT_SECONDS:-60}"
  if [[ ! -x "${python_path}" ]]; then
    echo "  ${python_path}: executable not found" >&2
    return 0
  fi
  echo "  ${python_path}:" >&2
  timeout --foreground "${import_timeout}" "${python_path}" - <<'PY' >&2 || true
import importlib

required = ("torch", "yaml", "open_clip", "timm", "transformers", "pandas", "PIL")
failed = False
for module in required:
    try:
        importlib.import_module(module)
    except Exception as exc:
        failed = True
        print(f"    FAIL {module}: {type(exc).__name__}: {exc}")
if not failed:
    print("    core imports succeeded")
PY
}

alignment_v2_resolve_python() {
  local root="${ALIGNMENT_V2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local candidate="${ALIGNMENT_V2_PYTHON:-${ALIGNMENT_V2_SYSTEM_PYTHON:-}}"
  local requested_candidate="${candidate}"

  if [[ -n "${candidate}" ]] && alignment_v2_python_is_usable "${candidate}"; then
    ALIGNMENT_V2_PYTHON="${candidate}"
  elif alignment_v2_python_is_usable "${root}/.venv/bin/python"; then
    ALIGNMENT_V2_PYTHON="${root}/.venv/bin/python"
  elif alignment_v2_python_is_usable "${root}/.venv-aisurrey/bin/python"; then
    ALIGNMENT_V2_PYTHON="${root}/.venv-aisurrey/bin/python"
  else
    if type module >/dev/null 2>&1; then
      module load miniconda >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
      local conda_base
      conda_base="$(conda info --base 2>/dev/null || true)"
      if [[ -n "${conda_base}" ]] && [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${ALIGNMENT_V2_CONDA_ENV:-vlm}" >/dev/null 2>&1 || true
      fi
    fi
    candidate="$(command -v python 2>/dev/null || true)"
    if [[ -n "${candidate}" ]] && alignment_v2_python_is_usable "${candidate}"; then
      ALIGNMENT_V2_PYTHON="${candidate}"
    else
      echo "Could not find a complete Alignment v2 Python environment." >&2
      echo "Required core imports: torch, PyYAML, open_clip, timm," >&2
      echo "transformers, pandas, and Pillow." >&2
      if [[ -n "${requested_candidate}" ]]; then
        alignment_v2_explain_python_failure "${requested_candidate}"
      fi
      if [[ "${requested_candidate}" != "${root}/.venv/bin/python" ]]; then
        alignment_v2_explain_python_failure "${root}/.venv/bin/python"
      fi
      if [[ "${requested_candidate}" != "${root}/.venv-aisurrey/bin/python" ]]; then
        alignment_v2_explain_python_failure "${root}/.venv-aisurrey/bin/python"
      fi
      if [[ -n "${candidate}" ]] && [[ "${candidate}" != "${requested_candidate}" ]] && [[ "${candidate}" != "${root}/.venv/bin/python" ]]; then
        alignment_v2_explain_python_failure "${candidate}"
      fi
      echo "Notebook rendering dependencies are checked separately and never block experiments." >&2
      echo "Set ALIGNMENT_V2_SYSTEM_PYTHON or ALIGNMENT_V2_CONDA_ENV." >&2
      return 2
    fi
  fi
  echo "alignment-v2: selected Python ${ALIGNMENT_V2_PYTHON}" >&2
  export ALIGNMENT_V2_PYTHON
}
