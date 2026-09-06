#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
mkdir -p logs/token_projection_profile results/token_projection_profile

[[ ! -e results/token_projection_profile/training_started ]] || {
  echo "Refusing to profile: unexpected training marker exists." >&2
  exit 2
}

job_id="$(timeout --foreground 60 sbatch --parsable \
  --partition=teaching \
  --job-name=token_pool_profile \
  --export="ALL,TOKEN_PROFILE_ROOT=${ROOT}" \
  slurm/token_projection_profile/profile.sbatch)"
echo "Token-projection profiling only: job=${job_id}"
echo "No training or Flickr30k test evaluation is included."

