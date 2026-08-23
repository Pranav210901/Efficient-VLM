#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COCO_DIR="${1:-data/coco}"
mkdir -p "${COCO_DIR}"

fetch() {
  local url="$1"
  local file="$2"
  if [[ -f "${COCO_DIR}/${file}" ]]; then
    echo "Already downloaded: ${COCO_DIR}/${file}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c -P "${COCO_DIR}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "${COCO_DIR}/${file}" "${url}"
  else
    echo "Need wget or curl to download COCO." >&2
    exit 1
  fi
}

fetch "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
fetch "http://images.cocodataset.org/zips/val2017.zip" "val2017.zip"
fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"

unzip -n "${COCO_DIR}/train2017.zip" -d "${COCO_DIR}"
unzip -n "${COCO_DIR}/val2017.zip" -d "${COCO_DIR}"
unzip -n "${COCO_DIR}/annotations_trainval2017.zip" -d "${COCO_DIR}"

python scripts/prepare_coco.py --coco_dir "${COCO_DIR}"
