#!/usr/bin/env bash

# Resolve a Python environment that can actually import the Phase 2 stack.
# The shared .venv points at /bin/python, which is not the same Python on every
# AISurrey node, so it must be tested rather than trusted by path alone.
phase2_resolve_python() {
  local root="${PHASE2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local candidate="${PHASE2_SYSTEM_PYTHON:-}"

  if [[ -n "$candidate" ]] && [[ -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    PHASE2_PYTHON="$candidate"
  elif [[ -x "$root/.venv/bin/python" ]] && "$root/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
    PHASE2_PYTHON="$root/.venv/bin/python"
  else
    if type module >/dev/null 2>&1; then
      module load miniconda >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
      local conda_base
      conda_base="$(conda info --base 2>/dev/null || true)"
      if [[ -n "$conda_base" ]] && [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "${PHASE2_CONDA_ENV:-vlm}" >/dev/null 2>&1 || true
      fi
    fi
    candidate="$(command -v python 2>/dev/null || true)"
    if [[ -n "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
      PHASE2_PYTHON="$candidate"
    else
      echo "Could not find a Python environment that imports torch." >&2
      echo "On AISurrey, load/activate the project environment first:" >&2
      echo "  module load miniconda" >&2
      echo "  conda activate ${PHASE2_CONDA_ENV:-vlm}" >&2
      echo "Or set PHASE2_SYSTEM_PYTHON=/absolute/path/to/python." >&2
      return 2
    fi
  fi

  export PHASE2_PYTHON
}
