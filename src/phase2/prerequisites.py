from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.models.text_encoders import TEXT_MODEL_REGISTRY
from src.models.vision_encoders import VISION_MODEL_REGISTRY
from src.phase15.evaluation_protocol import protected_test_ids
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text, sha256_file


def latest_snapshot(root: Path) -> Path:
    snapshots = sorted((root / "results/phase15/frozen_snapshots").glob("phase15_locked_*"))
    if not snapshots:
        raise FileNotFoundError("No frozen Phase 1.5 snapshot is available")
    return snapshots[-1]


def load_locked_selection(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / "results/phase15/expert_selection/selected_experts.yaml"
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("phase2_pair_shortlist"):
        raise ValueError("Canonical selected_experts.yaml is invalid")
    return payload


def primary_paths(selection: dict[str, Any]) -> list[dict[str, Any]]:
    return [value for value in selection["phase2_pair_shortlist"] if value.get("intended_role") == "primary"]


def _check(name: str, passed: bool, details: str, severity: str = "required") -> dict[str, Any]:
    return {"check": name, "severity": severity, "passed": bool(passed), "details": details}


def validate_phase2_prerequisites(project_root: str | Path, write: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    readiness_path = root / "results/phase15/phase2_readiness/readiness_report.json"
    selection_path = root / "results/phase15/expert_selection/selected_experts.yaml"
    training_root = root / "results/phase15/training_hard_negatives"
    snapshot = latest_snapshot(root)
    hashes_path = snapshot / "SHA256SUMS.json"
    rows: list[dict[str, Any]] = []

    readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
    rows.append(_check("phase15_readiness", readiness.get("status") == "READY", str(readiness.get("status"))))
    rows.append(_check("locked_selection_exists", selection_path.exists(), str(selection_path)))
    rows.append(_check("frozen_snapshot_exists", snapshot.exists() and hashes_path.exists(), str(snapshot)))

    expected_hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    mismatches = []
    for relative, expected in expected_hashes.items():
        live = root / relative
        if not live.exists() or sha256_file(live) != expected:
            mismatches.append(relative)
    rows.append(_check("snapshot_fingerprints", not mismatches, "matched" if not mismatches else ", ".join(mismatches)))

    selection = load_locked_selection(root) if selection_path.exists() else {}
    paths = primary_paths(selection) if selection else []
    missing_checkpoints = [value.get("checkpoint_path", "") for value in paths if not Path(value.get("checkpoint_path", "")).exists()]
    rows.append(_check("primary_checkpoints", len(paths) == 4 and not missing_checkpoints, f"paths={len(paths)} missing={missing_checkpoints}"))
    unknown_encoders = [
        value[side]
        for value in paths
        for side, registry in (("vision_encoder", VISION_MODEL_REGISTRY), ("text_encoder", TEXT_MODEL_REGISTRY))
        if value[side] not in registry
    ]
    rows.append(_check("encoder_registry", not unknown_encoders, str(unknown_encoders or "all registered")))

    efficiency = [root / f"results/phase15/efficiency/{name}" for name in ("vision_encoders.csv", "text_encoders.csv", "full_pairs.csv")]
    rows.append(_check("efficiency_tables", all(value.exists() for value in efficiency), str([str(value) for value in efficiency])))
    retrieval = training_root / "retrieval_train_hard_negatives.parquet"
    classification = training_root / "classification_train_hard_negatives.parquet"
    schema_path = training_root / "schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    rows.append(_check("training_hard_negatives", retrieval.exists() and classification.exists() and schema.get("status") == "complete", f"retrieval={retrieval.exists()} classification={classification.exists()} status={schema.get('status')}"))
    rows.append(_check("zero_protected_overlap", schema.get("protected_test_overlap") == 0, str(schema.get("protected_test_overlap"))))
    rows.append(_check("evaluation_protocol", (root / "configs/evaluation_protocol.yaml").exists(), "present"))
    data_paths = [root / "data/coco/train2017", root / "data/multitask/cifar100", root / "data/multitask/oxford-iiit-pet", root / "data/multitask/eurosat"]
    rows.append(_check("training_datasets", all(value.exists() for value in data_paths), str([str(value) for value in data_paths if not value.exists()] or "present")))
    rows.append(_check("protected_id_registry", len(protected_test_ids(root)) > 10000, f"protected_ids={len(protected_test_ids(root))}"))

    failures = [value for value in rows if value["severity"] == "required" and not value["passed"]]
    report = {
        "status": "VALID" if not failures else "INVALID",
        "phase15_readiness": readiness.get("status"),
        "snapshot": str(snapshot),
        "selection_sha256": sha256_file(selection_path) if selection_path.exists() else None,
        "primary_paths": [value.get("pair_id") for value in paths],
        "required_failures": len(failures),
        "checks": rows,
    }
    if write:
        output = root / "results/phase2/prerequisites"
        output.mkdir(parents=True, exist_ok=True)
        atomic_csv(pd.DataFrame(rows), output / "prerequisite_report.csv")
        atomic_json(report, output / "prerequisite_report.json")
        if failures:
            raise RuntimeError(f"Phase 2 prerequisites are invalid: {[value['check'] for value in failures]}")
        atomic_text(selection_path.read_text(), output / "validated_selection.yaml")
    return report
