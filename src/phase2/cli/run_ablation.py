from __future__ import annotations

import os
from copy import deepcopy
import pandas as pd

from .common import context, parser
from src.phase15.io_utils import atomic_csv
from src.phase2.ablations import manifest_row
from src.phase2.bridge_evaluation import evaluate_bridge_checkpoint
from src.phase2.prerequisites import load_locked_selection, primary_paths
from src.phase2.trainer import train_bridge


def main() -> None:
    args = parser("Run one focused Phase 2 ablation", "configs/phase2/ablations.yaml").parse_args()
    root, config = context(args); index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    row = manifest_row(root / "results/phase2/manifests/ablation_manifest.csv", index); ablation_id = str(row["ablation_id"])
    pair = next(value for value in load_locked_selection(root)["phase2_pair_shortlist"] if value["pair_id"] == row["pair_id"])
    run_dir = root / "results/phase2/ablations/runs" / ablation_id
    if not bool(row.get("enabled", True)):
        result = pd.DataFrame([{"method": ablation_id, "task": "optional", "metric": "enabled", "value": 0.0, "ablation_id": ablation_id}])
        run_dir.mkdir(parents=True, exist_ok=True); atomic_csv(result, run_dir / "results.csv"); print({"status": "optional_disabled", "ablation_id": ablation_id}); return
    if row["setting"] == "method" and row["value"] in {"dual_encoder", "blf"}:
        source = pd.read_csv(root / "results/phase1_multitask_development/task_results_long.csv")
        if row["value"] == "dual_encoder":
            result = source[source["config_id"].eq(pair["pair_id"])].copy()
        else:
            selection = load_locked_selection(root); prefix = f"{pair['vision_encoder']}__{pair['text_encoder']}__"
            candidates = [value["pair_id"] for value in selection.get("excluded_high_scoring_pairs", []) if value["pair_id"].startswith(prefix) and not value["pair_id"].endswith("__baseline")]
            candidate = candidates[0] if candidates else source.loc[source["variant"].ne("baseline"), "config_id"].iloc[0]
            result = source[source["config_id"].eq(candidate)].copy()
            reason = next((value["exclusion_reason"] for value in selection.get("excluded_high_scoring_pairs", []) if value["pair_id"] == candidate), "BLF reliability was not established")
            result["reliability_status"] = "inconclusive_or_not_tested"; result["comparison_note"] = reason
        result["method"] = str(row["value"]); result["ablation_id"] = ablation_id
        run_dir.mkdir(parents=True, exist_ok=True); atomic_csv(result, run_dir / "results.csv")
        print({"status": "reference", "ablation_id": ablation_id}); return
    canonical_results = root / "results/phase2/reranking/runs" / str(pair["pair_id"]) / "results.csv"
    if (row["setting"] == "method" and row["value"] == "cross_attention") or row["setting"] == "candidate_depth":
        result = pd.read_csv(canonical_results)
        if row["setting"] == "candidate_depth": result = result[result["candidate_depth"].eq(int(row["value"]))]
        result["method"] = ablation_id; result["ablation_id"] = ablation_id; run_dir.mkdir(parents=True, exist_ok=True); atomic_csv(result, run_dir / "results.csv"); print({"status": "reused_canonical_evaluation", "ablation_id": ablation_id}); return
    overridden = deepcopy(config); section = overridden.setdefault("model", {})
    setting, value = str(row["setting"]), row["value"]
    if setting in {"attention_heads", "bridge_dim", "pooling", "layers"}: section[setting] = int(value) if setting in {"attention_heads", "bridge_dim", "layers"} else value
    if setting == "negative_source": overridden.setdefault("training", {})["negative_source"] = value
    if setting == "pair_path":
        # Optional encoder paths were not used to mine a separate negative pool.
        # Use the canonical pool so this ablation changes the frozen path only.
        canonical = primary_paths(load_locked_selection(root))[0]["pair_id"]
        overridden.setdefault("training", {})["negative_config_id"] = canonical
    overridden["ablation"] = row
    summary = train_bridge(root, pair, overridden, resume=args.resume, experiment_id=ablation_id, output_group="ablations")
    evaluated = evaluate_bridge_checkpoint(root, pair, overridden, root / summary["best_checkpoint"], run_dir)
    print({**summary, **evaluated})


if __name__ == "__main__": main()
