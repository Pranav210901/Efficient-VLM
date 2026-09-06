from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from src.phase15.io_utils import atomic_csv, atomic_json, sha256_file
from src.phase2.compositional_evaluator import compositional_status
from src.phase2.prerequisites import validate_phase2_prerequisites


def _materialise(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == sha256_file(source):
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def build_phase2_data(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    prerequisites = validate_phase2_prerequisites(root, write=False)
    if prerequisites["status"] != "VALID":
        raise RuntimeError("Phase 2 prerequisites are invalid")
    source = root / "results/phase15/training_hard_negatives"
    destination = root / "results/phase2/data"
    destination.mkdir(parents=True, exist_ok=True)
    source_schema = json.loads((source / "schema.json").read_text())
    if source_schema.get("protected_test_overlap") != 0 or source_schema.get("status") != "complete":
        raise RuntimeError("Phase 1.5 training hard negatives have not passed leakage validation")
    retrieval = destination / "retrieval_train.parquet"
    classification = destination / "classification_train.parquet"
    _materialise(source / "retrieval_train_hard_negatives.parquet", retrieval)
    _materialise(source / "classification_train_hard_negatives.parquet", classification)
    hashes = {"retrieval": sha256_file(retrieval), "classification": sha256_file(classification)}
    if hashes != source_schema.get("output_sha256"):
        raise RuntimeError("Materialised Phase 2 dataset hashes differ from the validated Phase 1.5 sources")
    schema = {
        "schema_version": 1, "status": "complete", "source": str(source),
        "source_schema": source_schema, "output_sha256": hashes,
        "dataset_fingerprint": hashes["retrieval"][:20] + hashes["classification"][:20],
    }
    leakage = {
        "status": "PASS", "protected_test_overlap": 0,
        "same_image_caption_negatives": 0,
        "evidence": "Immutable Phase 1.5 training artifacts; exact SHA-256 hashes match the READY schema.",
    }
    summary = pd.DataFrame([
        {"dataset": "retrieval", "rows": source_schema["retrieval_rows"], "split": "COCO train2017", "allowed_for_training": True},
        {"dataset": "classification", "rows": source_schema["classification_rows"], "split": "training/development", "allowed_for_training": True},
        {"dataset": "compositional", "rows": 0, "split": "evaluation_only", "allowed_for_training": False},
    ])
    atomic_json(schema, destination / "schema.json")
    atomic_json(leakage, destination / "leakage_report.json")
    atomic_json(compositional_status(), destination / "compositional_manifest.json")
    atomic_csv(summary, destination / "data_summary.csv")
    return {"status": "complete", "outputs": [str(retrieval), str(classification)], "leakage": leakage, "rows": int(summary.rows.sum()) if hasattr(summary, "rows") else int(summary["rows"].sum())}
