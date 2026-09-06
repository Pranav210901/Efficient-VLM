#!/usr/bin/env bash
#SBATCH --job-name=p2_failure
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/99_report_failure/%j.out
#SBATCH --error=logs/phase2/99_report_failure/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/reporting.yaml"
phase2_diagnostics "$CONFIG"
"$PHASE2_PYTHON" -m src.phase2.cli.collect_failures --stage "${FAILED_STAGE:-unknown}" --job-id "${FAILED_JOB_ID:-unknown}"
