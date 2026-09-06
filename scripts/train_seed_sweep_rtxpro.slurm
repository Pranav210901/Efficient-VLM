#!/usr/bin/env bash
#SBATCH --job-name=blf_seed_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SWEEP_GPU_COUNT="${SWEEP_GPU_COUNT:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p logs checkpoints/seed_sweep results

if [[ ! -f data/val_all_captions.csv ]]; then
  "$PYTHON_BIN" scripts/prepare_coco.py \
    --split val \
    --all_val_captions \
    --output_val_csv data/val_all_captions.csv
fi

GPU_IDS=()
for ((gpu=0; gpu<SWEEP_GPU_COUNT; gpu++)); do
  GPU_IDS+=("$gpu")
done

echo "Project: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Allocated CUDA devices: ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"
echo "Launching one training process per logical GPU: ${GPU_IDS[*]}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$PYTHON_BIN" scripts/run_seed_sweep.py \
  --sweep_config configs/seed_sweep.yaml \
  --gpus "${GPU_IDS[@]}" \
  --resume
