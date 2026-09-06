#!/usr/bin/env bash
#SBATCH --job-name=p2_tokens
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/01_validate_token_interfaces/%j.out
#SBATCH --error=logs/phase2/01_validate_token_interfaces/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/token_interfaces.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 01_validate_token_interfaces "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.validate_token_interfaces --config "$CONFIG"
