from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.phase15.io_utils import atomic_csv
from .prerequisites import load_locked_selection, primary_paths


ABLATIONS = [
    ("dual_encoder_only", "method", "dual_encoder"),
    ("reliable_blf", "method", "blf"),
    ("canonical_cross_attention", "method", "cross_attention"),
    ("random_negatives", "negative_source", "random"),
    ("hard_negatives", "negative_source", "hard"),
    ("heads_2", "attention_heads", 2), ("heads_4", "attention_heads", 4),
    ("dim_256", "bridge_dim", 256), ("dim_512", "bridge_dim", 512),
    ("pool_cls", "pooling", "cls"), ("pool_masked_mean", "pooling", "masked_mean"),
    ("pool_learned", "pooling", "learned"), ("layers_1", "layers", 1), ("layers_2", "layers", 2),
    ("depth_16", "candidate_depth", 16), ("depth_32", "candidate_depth", 32),
    ("depth_64", "candidate_depth", 64), ("depth_128", "candidate_depth", 128),
    ("bidirectional_optional", "bridge_direction", "bidirectional"),
    ("learned_query_optional", "bridge_direction", "learned_query"),
]


def build_phase2_manifests(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    selection = load_locked_selection(root)
    primary = primary_paths(selection)
    output = root / "results/phase2/manifests"
    output.mkdir(parents=True, exist_ok=True)
    bridge = pd.DataFrame([{**pair, "manifest_index": index} for index, pair in enumerate(primary)])
    evaluation = pd.DataFrame([
        {"manifest_index": index, "pair_id": pair["pair_id"], "candidate_depths": "16,32,64,128"}
        for index, pair in enumerate(primary)
    ])
    canonical = primary[0]["pair_id"]
    ablation = pd.DataFrame([
        {"manifest_index": index, "pair_id": canonical, "ablation_id": name, "setting": setting, "value": value, "enabled": not name.endswith("_optional")}
        for index, (name, setting, value) in enumerate(ABLATIONS)
    ])
    optional_rows = []
    for pair in selection["phase2_pair_shortlist"]:
        if pair.get("intended_role") == "optional_ablation":
            optional_rows.append({"manifest_index": len(ablation) + len(optional_rows), "pair_id": pair["pair_id"], "ablation_id": f"optional_path__{pair['pair_id']}", "setting": "pair_path", "value": pair["pair_id"], "enabled": True})
    if optional_rows: ablation = pd.concat([ablation, pd.DataFrame(optional_rows)], ignore_index=True)
    seed_rows = []
    for method in ("best_bridge", "dense_teacher"):
        for seed in (17, 29, 43, 71, 101):
            seed_rows.append({"manifest_index": len(seed_rows), "method": method, "seed": seed})
    frames = {
        "bridge_training_manifest.csv": bridge,
        "bridge_evaluation_manifest.csv": evaluation,
        "ablation_manifest.csv": ablation,
        "seed_sweep_manifest.csv": pd.DataFrame(seed_rows),
    }
    paths = {}
    for name, frame in frames.items():
        paths[name] = atomic_csv(frame, output / name)
    bridge_root = root / "results/phase2/bridges"; bridge_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(bridge, bridge_root / "experiment_manifest.csv")
    ablation_root = root / "results/phase2/ablations"; (ablation_root / "figures").mkdir(parents=True, exist_ok=True)
    atomic_csv(ablation, ablation_root / "manifest.csv")
    for relative in ("bridges/checkpoints", "bridges/logs", "bridges/runs", "dense_teacher/checkpoints", "dense_teacher/logs", "dense_teacher/runs", "figures"):
        (root / "results/phase2" / relative).mkdir(parents=True, exist_ok=True)
    return paths


def manifest_row(path: str | Path, index: int) -> dict:
    frame = pd.read_csv(path)
    matches = frame[frame["manifest_index"].eq(int(index))]
    if len(matches) != 1:
        raise IndexError(f"Manifest index {index} does not identify exactly one row in {path}")
    return matches.iloc[0].to_dict()
