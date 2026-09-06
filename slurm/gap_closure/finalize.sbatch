#!/usr/bin/env bash
#SBATCH --job-name=gap_finalize
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/gap_closure/%x_%j.out
#SBATCH --error=logs/gap_closure/%x_%j.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
bash scripts/run_final_bootstrap.sh
bash scripts/gap_closure_status.sh
