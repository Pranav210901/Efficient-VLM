"""Assemble leakage-safe multi-task development evidence without recomputation.

COCO val2017 is explicitly the retrieval development/selection split, so its
existing rows are reused. Classification rows must come from the new frozen
development evaluator output. Historical classification test rows are never
copied into this table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.multitask.runner import _write_tables

from .io_utils import atomic_json


CLASSIFICATION_TASKS = {"cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"}


def assemble_development_evidence(project_root: str | Path, *, strict: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    historical = root / "results/phase1_multitask"
    development = root / "results/phase1_multitask_development"
    required = [
        historical / "task_results_long.csv",
        historical / "evaluation_manifest.csv",
        development / "task_results_long.csv",
        development / "evaluation_manifest.csv",
    ]
    missing = [str(value) for value in required if not value.exists()]
    if missing:
        raise FileNotFoundError(f"Development evidence prerequisites are missing: {missing}")
    historical_results = pd.read_csv(required[0])
    historical_manifest = pd.read_csv(required[1])
    development_results = pd.read_csv(required[2])
    development_manifest = pd.read_csv(required[3])
    historical_coco = historical_results[
        historical_results["task"].eq("coco_retrieval")
        & historical_results.get("mode", pd.Series("full", index=historical_results.index)).eq("full")
    ].copy()
    historical_coco["split_role"] = "development"
    historical_coco["dataset_split"] = "coco_val2017_development"
    development_classification = development_results[
        development_results["task"].isin(CLASSIFICATION_TASKS)
        & development_results.get("mode", pd.Series("full", index=development_results.index)).eq("full")
    ].copy()
    if "split_role" not in development_classification or not development_classification["split_role"].eq("development").all():
        raise ValueError("Classification evidence is not labelled as development")
    coco_configs = set(historical_coco["config_id"])
    classification_configs = set(development_classification["config_id"])
    if strict and coco_configs != classification_configs:
        raise ValueError(
            f"Development configuration coverage differs: COCO-only={sorted(coco_configs - classification_configs)}, "
            f"classification-only={sorted(classification_configs - coco_configs)}"
        )
    complete_configs = coco_configs & classification_configs
    combined_results = pd.concat(
        [historical_coco[historical_coco["config_id"].isin(complete_configs)], development_classification[development_classification["config_id"].isin(complete_configs)]],
        ignore_index=True,
    )
    if combined_results.duplicated(["config_id", "task", "metric"]).any():
        raise ValueError("Combined development metrics contain duplicates")
    coco_manifest = historical_manifest[
        historical_manifest["task"].eq("coco_retrieval")
        & historical_manifest["status"].eq("complete")
        & historical_manifest.get("mode", pd.Series("full", index=historical_manifest.index)).eq("full")
        & historical_manifest["config_id"].isin(complete_configs)
    ].copy()
    coco_manifest["split_role"] = "development"
    coco_manifest["dataset_split"] = "coco_val2017_development"
    class_manifest = development_manifest[
        development_manifest["task"].isin(CLASSIFICATION_TASKS)
        & development_manifest["status"].eq("complete")
        & development_manifest["config_id"].isin(complete_configs)
    ].copy()
    combined_manifest = pd.concat([coco_manifest, class_manifest], ignore_index=True)
    expected_tasks = 4
    complete = combined_manifest.groupby("config_id")["task"].nunique()
    if strict and (complete != expected_tasks).any():
        raise ValueError("Every development configuration must cover COCO and all three classification datasets")
    status_frames = []
    for path, tasks in (
        (historical / "task_status.csv", {"coco_retrieval"}),
        (development / "task_status.csv", CLASSIFICATION_TASKS),
    ):
        if path.exists():
            frame = pd.read_csv(path)
            status_frames.append(frame[frame["task"].isin(tasks)])
    statuses = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    paths = _write_tables(development, combined_results.to_dict("records"), statuses.to_dict("records"), combined_manifest.to_dict("records"))
    summary = {
        "status": "complete" if len(complete_configs) and (complete == expected_tasks).all() else "incomplete",
        "configurations": len(complete_configs),
        "tasks": sorted(combined_manifest["task"].unique()),
        "coco_source": "historical COCO val2017 reused as frozen development evidence",
        "classification_source": "deterministic train/trainval/EuroSAT development partitions",
        "historical_classification_test_rows_reused": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    atomic_json(summary, development / "development_evidence_manifest.json")
    return summary
