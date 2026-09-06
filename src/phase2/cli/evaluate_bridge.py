from __future__ import annotations

import os
from pathlib import Path
from .common import context, parser
from src.phase2.ablations import manifest_row
from src.phase2.bridge_evaluation import evaluate_bridge_checkpoint
from src.phase2.status import complete_stage, managed_stage, valid_completed_run
from src.phase2.config import fingerprint


def main() -> None:
    args = parser("Evaluate one trained bridge", "configs/phase2/bridge_evaluation.yaml").parse_args()
    root, config = context(args); index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    row = manifest_row(root / "results/phase2/manifests/bridge_evaluation_manifest.csv", index)
    training_pair = manifest_row(root / "results/phase2/manifests/bridge_training_manifest.csv", index); pair_id = str(row["pair_id"])
    checkpoint = root / "results/phase2/bridges/checkpoints" / pair_id / "best.pt"
    run_dir = root / "results/phase2/reranking/runs" / pair_id
    if args.resume and valid_completed_run(run_dir, fingerprint({key: value for key, value in config.items() if key != "_config_path"})):
        print({"status": "skipped_complete", "pair_id": pair_id, "output": str(run_dir)}); return
    with managed_stage(run_dir, "bridge_evaluation", config, root):
        summary = evaluate_bridge_checkpoint(root, training_pair, config, checkpoint, run_dir)
        complete_stage(run_dir, "bridge_evaluation", {"status": "COMPLETED", "pair_id": pair_id, "config_fingerprint": fingerprint({key: value for key, value in config.items() if key != "_config_path"}), **summary})
    print({"status": "COMPLETED", "pair_id": pair_id, "output": str(run_dir)})


if __name__ == "__main__": main()
