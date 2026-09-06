#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
mkdir -p logs/token_aggregator_scale_profile \
  results/token_aggregator_scale_profile
job="$(timeout --foreground 60 sbatch --parsable \
  --partition=teaching \
  --job-name=token_scale_profile \
  --export="ALL,TOKEN_SCALE_ROOT=${ROOT}" \
  slurm/token_aggregator_scale_profile/profile.sbatch)"
echo "Token-aggregator scale profiling only: job=${job}"
echo "Profiles CLS and C1-C5 sequentially in one quiet allocation."
echo "No training and no Flickr30k test evaluation."
