#!/usr/bin/env bash
set -euo pipefail
CC3M_ROOT="${CC3M_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
CC3M_PIPELINE="${CC3M_PIPELINE:-configs/cc3m_scale/pipeline.yaml}"
cd "${CC3M_ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export PYTHONUNBUFFERED=1
mkdir -p logs/cc3m_scale/slurm

