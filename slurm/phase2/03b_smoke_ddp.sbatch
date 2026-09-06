#!/usr/bin/env bash
#SBATCH --job-name=p2_ddp_smoke
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/03b_smoke_ddp/%j.out
#SBATCH --error=logs/phase2/03b_smoke_ddp/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/smoke.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 03b_smoke_ddp "$CONFIG" "$PHASE2_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" -m src.phase2.cli.smoke_test --config "$CONFIG"
