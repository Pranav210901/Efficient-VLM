from __future__ import annotations

import os, random
import numpy as np
import pandas as pd
import torch

from .common import context, parser
from src.phase15.io_utils import atomic_csv
from src.phase2.ablations import manifest_row
from src.phase2.prerequisites import load_locked_selection, primary_paths
from src.phase2.trainer import train_bridge
from src.phase2.bridge_evaluation import evaluate_bridge_checkpoint
from src.phase2.dense_training import evaluate_dense_checkpoint, train_dense_teacher


def main() -> None:
    args = parser("Run one reliability seed", "configs/phase2/seed_sweep.yaml").parse_args()
    root, config = context(args); index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    row = manifest_row(root / "results/phase2/manifests/seed_sweep_manifest.csv", index); seed = int(row["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run = root / "results/phase2/seed_sweep/runs" / f"{row['method']}__seed_{seed}"; run.mkdir(parents=True, exist_ok=True)
    if args.resume and (run / "results.csv").exists():
        print({"status": "skipped_complete", "run": str(run)}); return
    if row["method"] == "best_bridge":
        pair = primary_paths(load_locked_selection(root))[0]; config["seed"] = seed
        summary = train_bridge(root, pair, config, resume=args.resume, experiment_id=f"best_bridge__seed_{seed}", output_group="seed_sweep")
        evaluation_config = dict(config); evaluation_config["candidate_depths"] = [64]; evaluation_config["tasks"] = ["coco_retrieval", "cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"]
        evaluated = evaluate_bridge_checkpoint(root, pair, evaluation_config, summary["best_checkpoint"], run)
        samples = pd.read_parquet(run / "per_sample_results.parquet")
        retrieval_utility = 1.0 / samples.loc[samples["direction"].notna(), "positive_rank"].astype(float).clip(lower=1.0)
        classification_utility = samples.loc[samples["direction"].isna(), "correct"].astype(float)
        efficiency = pd.read_csv(run / "efficiency.csv")
        values = {"mean_sample_utility": float(pd.concat([retrieval_utility, classification_utility]).mean()), "latency_ms": float(efficiency["latency_ms"].mean()), "peak_memory_bytes": float(efficiency["peak_memory_bytes"].max())}
    else:
        active = dict(config); active.update(config.get("dense_teacher", {})); active.update({"seed": seed, "strategy": "task_and_input", "strategies": ["task_and_input"], "run_name": f"dense_teacher__seed_{seed}"})
        summary = train_dense_teacher(root, active, resume=args.resume)
        values = evaluate_dense_checkpoint(root, active, summary["best_checkpoint"])
    result = pd.DataFrame([{"method": row["method"], "seed": seed, "task": "multitask", "metric": metric, "value": value} for metric, value in values.items()]); atomic_csv(result, run / "results.csv"); print(result.to_dict("records"))


if __name__ == "__main__": main()
