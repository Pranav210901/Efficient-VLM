#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
mkdir -p logs/token_projection_training/slurm \
  results/token_projection_training/freeze

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.token_projection_freeze paired
job="$(timeout --foreground 60 sbatch --parsable \
  --partition=teaching \
  --job-name=tp_freeze_profile \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT}" \
  slurm/token_projection_training/freeze_profile.sbatch)"
echo "Controlled freeze profile: job=${job}"
echo "Profiles trained B, trained CLS-only, and OpenCLIP sequentially in one allocation."
echo "No training and no Flickr30k test evaluation."
