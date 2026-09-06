from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .classification_complementarity import load_classification_predictions
from .io_utils import atomic_csv, atomic_json
from .evidence import resolve_evaluation_root


def oracle_recall(ranks: list[list[int]] | np.ndarray, k_values=(1, 5, 10)) -> dict[str, float]:
    values = np.asarray(ranks)
    if values.ndim != 2: raise ValueError("ranks must be [experts, samples]")
    best = values.min(axis=0)
    return {f"oracle_R@{k}": float((best <= k).mean()) for k in k_values}


def oracle_accuracy(correct: list[list[bool]] | np.ndarray) -> float:
    values = np.asarray(correct, dtype=bool)
    if values.ndim != 2: raise ValueError("correct must be [experts, samples]")
    return float(values.any(axis=0).mean())


def classification_oracle_subset(
    correct_by_configuration: dict[str, Iterable[bool]],
    configurations: Iterable[str] | None = None,
) -> dict[str, Any]:
    names = list(configurations or sorted(correct_by_configuration))
    if not names:
        raise ValueError("At least one configuration is required")
    arrays = [np.asarray(list(correct_by_configuration[name]), dtype=bool) for name in names]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("All classification correctness arrays must have equal non-zero length")
    matrix = np.stack(arrays)
    union = matrix.any(axis=0)
    individual = matrix.mean(axis=1)
    unique: dict[str, int] = {}
    for index, name in enumerate(names):
        unique[name] = int((matrix[index] & (matrix.sum(axis=0) == 1)).sum())
    oracle = float(union.mean())
    best = float(individual.max())
    return {
        "subset_size": len(names),
        "configurations": names,
        "best_individual_accuracy": best,
        "oracle_top1_accuracy": oracle,
        "absolute_oracle_gain": oracle - best,
        "relative_oracle_gain": (oracle - best) / max(best, 1e-12),
        "oracle_correct_count": int(union.sum()),
        "unique_contribution_per_configuration": unique,
        "oracle_mask": union,
        "matrix": matrix,
    }


def _subset_id(scope: str, names: Iterable[str]) -> str:
    joined = "|".join(names)
    return f"{scope}__{hashlib.sha256(joined.encode()).hexdigest()[:12]}"


def _comparison_pools(root: Path, names: list[str]) -> dict[str, tuple[str, ...]]:
    pools: dict[str, tuple[str, ...]] = {"full_pool": tuple(names)}
    parsed = pd.DataFrame(
        [
            {
                "config_id": name,
                "vision": name.split("__")[0] if name.count("__") == 2 else "",
                "text": name.split("__")[1] if name.count("__") == 2 else "",
                "variant": name.split("__")[2] if name.count("__") == 2 else "",
            }
            for name in names
        ]
    )
    vision_path = root / "results/phase15/expert_selection/vision_experts_ranked.csv"
    if not vision_path.exists():
        vision_path = root / "results/phase15/expert_selection/vision_experts.csv"
    text_path = root / "results/phase15/expert_selection/text_experts_ranked.csv"
    if not text_path.exists():
        text_path = root / "results/phase15/expert_selection/text_experts.csv"
    visions = pd.read_csv(vision_path)["vision_encoder"].astype(str).head(3).tolist() if vision_path.exists() else sorted(parsed["vision"].unique())[:3]
    texts = pd.read_csv(text_path)["text_encoder"].astype(str).head(3).tolist() if text_path.exists() else sorted(parsed["text"].unique())[:3]
    vision_pool = parsed[
        parsed["vision"].isin(visions) & parsed["text"].eq("all_minilm_l6_v2") & parsed["variant"].eq("baseline")
    ]["config_id"].tolist()
    text_pool = parsed[
        parsed["text"].isin(texts) & parsed["vision"].eq("dinov2_vits14") & parsed["variant"].eq("baseline")
    ]["config_id"].tolist()
    if len(vision_pool) >= 2:
        pools["selected_vision_pool"] = tuple(vision_pool)
    if len(text_pool) >= 2:
        pools["selected_text_pool"] = tuple(text_pool)
    return pools


def run_classification_oracle_analysis(project_root: str | Path, mode: str = "full", include_triples: bool = True, evaluation_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = pd.read_csv(resolve_evaluation_root(root, evaluation_root) / "evaluation_manifest.csv")
    if "mode" not in manifest:
        manifest["mode"] = manifest["cache"].map(lambda value: "full" if str(value).endswith("__full.pt") else "smoke")
    selected = manifest[
        manifest["status"].eq("complete")
        & manifest["mode"].eq(mode)
        & manifest["task"].isin(["cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"])
    ].copy()
    if selected.empty:
        raise FileNotFoundError("No classification predictions are available for oracle analysis")
    oracle_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    for dataset, group in selected.groupby("task", sort=True):
        tables = {}
        for row in group.sort_values("config_id").itertuples():
            path = Path(str(row.predictions))
            if not path.is_absolute():
                path = root / path
            tables[str(row.config_id)] = load_classification_predictions(path).sort_values("sample_id").reset_index(drop=True)
        names = sorted(tables)
        reference_ids = tables[names[0]]["sample_id"].astype(str).tolist()
        targets = tables[names[0]]["target_index"].to_numpy()
        class_names = tables[names[0]]["target"].astype(str).to_numpy()
        correct: dict[str, np.ndarray] = {}
        for name, table in tables.items():
            if table["sample_id"].astype(str).tolist() != reference_ids or not np.array_equal(table["target_index"].to_numpy(), targets):
                raise ValueError(f"{dataset} predictions are not sample-aligned for {name}")
            correct[name] = table["correct"].astype(bool).to_numpy()
        subsets: list[tuple[str, tuple[str, ...]]] = []
        subsets.extend(("pair", value) for value in itertools.combinations(names, 2))
        if include_triples and len(names) <= 30:
            subsets.extend(("triple", value) for value in itertools.combinations(names, 3))
        subsets.extend(_comparison_pools(root, names).items())
        seen: set[tuple[str, ...]] = set()
        for scope, subset in subsets:
            subset = tuple(subset)
            if subset in seen or len(subset) < 2:
                continue
            seen.add(subset)
            result = classification_oracle_subset(correct, subset)
            subset_id = _subset_id(scope, subset)
            common = {
                "dataset": dataset,
                "subset_id": subset_id,
                "subset_scope": scope,
                "subset_size": len(subset),
                "configurations": json.dumps(list(subset)),
                "best_individual_accuracy": result["best_individual_accuracy"],
                "oracle_top1_accuracy": result["oracle_top1_accuracy"],
                "absolute_oracle_gain": result["absolute_oracle_gain"],
                "relative_oracle_gain": result["relative_oracle_gain"],
                "oracle_correct_count": result["oracle_correct_count"],
                "unique_contribution_per_configuration": json.dumps(result["unique_contribution_per_configuration"], sort_keys=True),
                "interpretation": "diagnostic upper bound — not deployable",
            }
            oracle_rows.append(common)
            oracle_mask = result["oracle_mask"]
            for class_id in sorted(set(int(value) for value in targets)):
                mask = targets == class_id
                per_class_rows.append(
                    {
                        "dataset": dataset,
                        "subset_id": subset_id,
                        "subset_scope": scope,
                        "subset_size": len(subset),
                        "class_id": class_id,
                        "class_name": str(class_names[np.flatnonzero(mask)[0]]),
                        "sample_count": int(mask.sum()),
                        "oracle_correct_count": int(oracle_mask[mask].sum()),
                        "oracle_top1_accuracy": float(oracle_mask[mask].mean()),
                    }
                )
            matrix = result["matrix"]
            full_oracle = float(result["oracle_top1_accuracy"])
            for index, name in enumerate(subset):
                without = np.delete(matrix, index, axis=0).any(axis=0)
                marginal_rows.append(
                    {
                        "dataset": dataset,
                        "subset_id": subset_id,
                        "subset_scope": scope,
                        "configuration": name,
                        "unique_correct_count": int(result["unique_contribution_per_configuration"][name]),
                        "marginal_oracle_loss": full_oracle - float(without.mean()),
                    }
                )
    oracle_frame = pd.DataFrame(oracle_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    marginal_frame = pd.DataFrame(marginal_rows)
    output = root / "results/phase15/oracle"
    paths = {
        "oracle": output / "classification_oracle.csv",
        "per_class": output / "classification_oracle_per_class.csv",
        "marginal": output / "classification_marginal_contributions.csv",
    }
    atomic_csv(oracle_frame, paths["oracle"])
    atomic_csv(per_class_frame, paths["per_class"])
    atomic_csv(marginal_frame, paths["marginal"])
    summary = {
        "datasets": sorted(selected["task"].unique()),
        "oracle_subsets": len(oracle_frame),
        "per_class_rows": len(per_class_frame),
        "marginal_rows": len(marginal_frame),
        "interpretation": "diagnostic upper bound — not deployable",
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    atomic_json(summary, output / "classification_oracle_summary.json")
    return summary
