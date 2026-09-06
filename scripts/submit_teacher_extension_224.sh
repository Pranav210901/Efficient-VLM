#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
MAX_GPUS=8
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "Teacher extension is restricted to teaching/native-BF16 hardware." >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be in [1,8]." >&2
  exit 2
}

mkdir -p logs/teacher_extension_224/slurm results/teacher_extension_224
pipelines=(
  configs/teacher_extension_224/pipeline_siglip2.yaml
  configs/teacher_extension_224/pipeline_openclip.yaml
  configs/teacher_extension_224/pipeline_multi.yaml
)
for pipeline in "${pipelines[@]}"; do
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${pipeline}"
  "${ALIGNMENT_V3_PYTHON}" \
    -m src.alignment_v3.resolution_distillation_trajectory validate \
    --pipeline "${pipeline}"
done
"${ALIGNMENT_V3_PYTHON}" - <<'PY'
import open_clip
from src.alignment_v3.references import REFERENCE_CHECKPOINTS
expected = ("ViT-B-32-quickgelu", "openai")
actual = REFERENCE_CHECKPOINTS["openclip_vit_b32_quickgelu_openai"]
if actual != expected:
    raise SystemExit(f"OpenCLIP registry mismatch: {actual} != {expected}")
if open_clip.__version__ != "3.3.0":
    raise SystemExit(f"OpenCLIP version mismatch: {open_clip.__version__} != 3.3.0")
print(f"OpenCLIP teacher registry verified: {actual}, open_clip={open_clip.__version__}")
PY

submit() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "98${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

cache="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_cache \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT}" \
  slurm/teacher_extension_224/cache.sbatch)"
train="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_train \
  --array="0-8%${MAX_GPUS}" \
  --dependency="afterok:${cache}" \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT}" \
  slurm/teacher_extension_224/train.sbatch)"
verify="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_verify \
  --dependency="afterok:${train}" \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT},TEACHER_EXTENSION_COMMAND=verify" \
  slurm/teacher_extension_224/cpu.sbatch)"
evaluation="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_eval \
  --array="0-215%${MAX_GPUS}" \
  --dependency="afterok:${verify}" \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT}" \
  slurm/teacher_extension_224/evaluate.sbatch)"
profile="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_profile \
  --array="0-2%1" \
  --dependency="afterok:${train}" \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT}" \
  slurm/teacher_extension_224/profile.sbatch)"
report="$(submit \
  --partition="${PARTITION}" \
  --job-name=te224_report \
  --dependency="afterok:${evaluation}:${profile}" \
  --export="ALL,TEACHER_EXTENSION_ROOT=${ROOT},TEACHER_EXTENSION_COMMAND=report" \
  slurm/teacher_extension_224/cpu.sbatch)"

echo "Teacher extension: cache=${cache} train=${train} verify=${verify} eval=${evaluation} profile=${profile} report=${report}"
echo "Nine training runs: SigLIP2, OpenCLIP, and equal-weight MobileCLIP2+SigLIP2, seeds 42/43/44."
echo "Flickr30k TEST is not referenced anywhere in this graph."

