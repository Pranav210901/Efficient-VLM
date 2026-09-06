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
from src.training.evaluate import extract_embeddings
from src.training.train import build_model_from_config
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract validation embeddings from a trained checkpoint.")
    parser.add_argument("--config", default="configs/experiment_blf.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output_dir", default="embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    device = get_device(str(config.get("device", "auto")))
    _, val_loader = build_dataloaders(config)
    model = build_model_from_config(config).to(device)
    load_checkpoint(PROJECT_ROOT / args.checkpoint, model, map_location=device)
    payload = extract_embeddings(model, val_loader, device)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload["image_embeds"], out_dir / "image_embeds.pt")
    torch.save(payload["text_embeds"], out_dir / "text_embeds.pt")
    (out_dir / "metadata.txt").write_text(
        "\n".join(f"{path}\t{caption}" for path, caption in zip(payload["image_paths"], payload["captions"], strict=True)) + "\n"
    )
    print(f"Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
