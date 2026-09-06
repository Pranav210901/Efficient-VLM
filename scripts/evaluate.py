from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data import build_dataloaders
from src.training.evaluate import evaluate_model
from src.training.train import build_model_from_config
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BLF VLM checkpoint.")
    parser.add_argument("--config", default="configs/experiment_blf.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    device = get_device(str(config.get("device", "auto")))
    _, val_loader = build_dataloaders(config)
    model = build_model_from_config(config).to(device)
    load_checkpoint(PROJECT_ROOT / args.checkpoint, model, map_location=device)
    metrics = evaluate_model(model, val_loader, device, k_values=list(config.get("evaluation", {}).get("k_values", [1, 5, 10])))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
