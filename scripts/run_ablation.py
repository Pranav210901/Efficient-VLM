from __future__ import annotations

import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.training.train import train_from_config
from src.utils.config import load_config


ABLATIONS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLF local/global ablations.")
    parser.add_argument("--config", default="configs/experiment_blf.yaml")
    parser.add_argument("--vision_encoder", default=None)
    parser.add_argument("--text_encoder", default=None)
    parser.add_argument("--fusion_type", default=None)
    return parser.parse_args()


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_config = load_config(PROJECT_ROOT / args.config)
    results_path = PROJECT_ROOT / "results" / "ablation_results.csv"
    for use_local, use_global in ABLATIONS:
        config = deepcopy(base_config)
        model_cfg = config["model"]
        if args.vision_encoder:
            model_cfg["vision_encoder"] = args.vision_encoder
        if args.text_encoder:
            model_cfg["text_encoder"] = args.text_encoder
        if args.fusion_type:
            model_cfg["fusion_type"] = args.fusion_type
        model_cfg["use_local_blf"] = use_local
        model_cfg["use_global_blf"] = use_global
        suffix = f"{model_cfg['vision_encoder']}_{model_cfg['text_encoder']}_local{int(use_local)}_global{int(use_global)}"
        config["training"]["save_dir"] = str(Path(config["training"].get("save_dir", "checkpoints")) / suffix)
        metrics = train_from_config(config)
        row = {
            "vision_encoder": model_cfg["vision_encoder"],
            "text_encoder": model_cfg["text_encoder"],
            "fusion_type": model_cfg.get("fusion_type", "concat_mlp"),
            "use_local_blf": use_local,
            "use_global_blf": use_global,
            **metrics,
        }
        append_result(results_path, row)
        print(f"Saved ablation row to {results_path}")


if __name__ == "__main__":
    main()
