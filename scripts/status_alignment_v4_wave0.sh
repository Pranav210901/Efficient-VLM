#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
echo "===== Wave 0 OOM smoke results ====="
if [[ -f results/alignment_v4_wave0/oom_smoke/summary.csv ]]; then
  column -s, -t < results/alignment_v4_wave0/oom_smoke/summary.csv || \
    cat results/alignment_v4_wave0/oom_smoke/summary.csv
else
  echo "No completed OOM summary yet."
fi
echo "===== Wave 0 Slurm jobs ====="
squeue -u "${USER}" -o "%.18i %.9P %.28j %.2t %.10M %.30R"

