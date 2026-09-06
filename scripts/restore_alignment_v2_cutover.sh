#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
quarantine_root="${project_root}/quarantine/alignment_v2_cutover_20260723"
mode="${1:---dry-run}"

if [[ "${mode}" != "--dry-run" && "${mode}" != "--execute" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

if [[ ! -d "${quarantine_root}" ]]; then
  echo "Quarantine not found: ${quarantine_root}" >&2
  exit 1
fi

restore_path() {
  local source_path="$1"
  local destination_path="$2"

  [[ -e "${source_path}" ]] || return 0
  if [[ -e "${destination_path}" ]]; then
    echo "REFUSE existing destination: ${destination_path}" >&2
    return 1
  fi

  echo "RESTORE ${source_path} -> ${destination_path}"
  if [[ "${mode}" == "--execute" ]]; then
    mkdir -p "$(dirname "${destination_path}")"
    mv "${source_path}" "${destination_path}"
  fi
}

restore_children() {
  local source_directory="$1"
  local destination_directory="$2"
  local child

  [[ -d "${source_directory}" ]] || return 0
  shopt -s nullglob dotglob
  for child in "${source_directory}"/*; do
    restore_path "${child}" "${destination_directory}/$(basename "${child}")"
  done
  shopt -u nullglob dotglob
}

# Legacy checkpoints are merged beside the active alignment_v2 directory.
restore_children "${quarantine_root}/legacy_checkpoints" \
  "${project_root}/checkpoints"

restore_children "${quarantine_root}/legacy_results" \
  "${project_root}/results"
restore_children "${quarantine_root}/legacy_logs" \
  "${project_root}/logs"

restore_path "${quarantine_root}/legacy_data/train.csv" \
  "${project_root}/data/train.csv"
restore_path "${quarantine_root}/legacy_data/val.csv" \
  "${project_root}/data/val.csv"
restore_path "${quarantine_root}/legacy_data/multitask" \
  "${project_root}/data/multitask"
restore_path "${quarantine_root}/legacy_data/coco/train2017.zip" \
  "${project_root}/data/coco/train2017.zip"
restore_path "${quarantine_root}/legacy_data/coco/val2017.zip" \
  "${project_root}/data/coco/val2017.zip"
restore_path "${quarantine_root}/legacy_data/coco/annotations_trainval2017.zip" \
  "${project_root}/data/coco/annotations_trainval2017.zip"
restore_children "${quarantine_root}/legacy_data/coco/annotations" \
  "${project_root}/data/coco/annotations"

restore_path "${quarantine_root}/legacy_environment/.venv-aisurrey" \
  "${project_root}/.venv-aisurrey"

restore_path "${quarantine_root}/scratch_and_caches/root_pytest_cache" \
  "${project_root}/.pytest_cache"
restore_path "${quarantine_root}/scratch_and_caches/root_notebook_checkpoints" \
  "${project_root}/.ipynb_checkpoints"
restore_path "${quarantine_root}/scratch_and_caches/notebooks_checkpoints" \
  "${project_root}/notebooks/.ipynb_checkpoints"
restore_path "${quarantine_root}/scratch_and_caches/scripts_checkpoints" \
  "${project_root}/scripts/.ipynb_checkpoints"
restore_path "${quarantine_root}/scratch_and_caches/Untitled.ipynb" \
  "${project_root}/Untitled.ipynb"
restore_path "${quarantine_root}/scratch_and_caches/src___pycache__" \
  "${project_root}/src/__pycache__"
restore_path "${quarantine_root}/scratch_and_caches/scripts___pycache__" \
  "${project_root}/scripts/__pycache__"
restore_path "${quarantine_root}/scratch_and_caches/tests___pycache__" \
  "${project_root}/tests/__pycache__"

if [[ "${mode}" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with --execute to restore."
else
  echo "Restore complete. Existing active paths were not overwritten."
fi
