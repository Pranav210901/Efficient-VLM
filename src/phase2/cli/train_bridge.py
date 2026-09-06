from __future__ import annotations

import os
from .common import context, parser
from src.phase2.ablations import manifest_row
from src.phase2.trainer import train_bridge


def main() -> None:
    args = parser("Train one selected bridge", "configs/phase2/bridge_training.yaml").parse_args()
    root, config = context(args); index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    pair = manifest_row(root / "results/phase2/manifests/bridge_training_manifest.csv", index)
    print(train_bridge(root, pair, config, resume=args.resume))


if __name__ == "__main__": main()
