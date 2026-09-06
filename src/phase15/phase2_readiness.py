"""Evidence-based gate for starting future Phase 2 work.

The checker reports readiness only. It deliberately contains no Phase 2 model,
router, fusion, cross-attention, training, or controller implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.multitask.config import TEXT_ENCODERS, VISION_ENCODERS

from .evaluation_protocol import protected_test_ids as load_protected_test_ids
from .evidence import resolve_evaluation_root
from .io_utils import atomic_csv, atomic_json, atomic_text, sha256_file
from .training_hard_negative_mining import validate_no_protected_ids, validate_training_negative_metadata


SAFE_DEFAULTS = {
    "RUN_PHASE1_TRAINING": False,
    "RUN_SEED_SWEEP": False,
    "RUN_QUALITATIVE_RETRIEVAL": False,
    "RUN_MULTITASK_EVALUATION": False,
    "RUN_PHASE15": False,
    "RUN_DEVELOPMENT_EVALUATION": False,
    "RUN_EFFICIENCY_PROFILING": False,
    "RUN_DIAGNOSTIC_HARD_NEGATIVE_MINING": False,
    "RUN_TRAINING_HARD_NEGATIVE_MINING": False,
    "RUN_MODE": "smoke",
    "RESULTS_ONLY": True,
}


def _read_notebook_defaults(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    defaults: dict[str, Any] = {}
    for key in SAFE_DEFAULTS:
        for line in source.splitlines():
            if line.strip().startswith(key + " ="):
                raw = line.split("=", 1)[1].split("#", 1)[0].strip()
                if raw in {"True", "False"}:
                    defaults[key] = raw == "True"
                else:
                    defaults[key] = raw.strip("\"'")
                break
    return defaults


def _file_check(path: Path, required_columns: set[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, f"missing: {path}", {}
    if required_columns is None:
        return True, "present", {}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}: {exc}", {}
    missing = required_columns - set(frame.columns)
    if missing:
        return False, f"missing columns: {sorted(missing)}", {"rows": len(frame)}
    return (not frame.empty), ("valid" if not frame.empty else "empty"), {"rows": len(frame)}


def _selected_yaml(root: Path) -> tuple[dict[str, Any] | None, str]:
    path = root / "results/phase15/expert_selection/selected_experts.yaml"
    if not path.exists():
        return None, f"missing: {path}"
    try:
        payload = yaml.safe_load(path.read_text())
        required = {"schema_version", "vision_experts", "text_experts", "phase2_pair_shortlist", "selection_protocol"}
        if not isinstance(payload, dict) or required - set(payload):
            return None, f"invalid selected_experts.yaml schema: missing {sorted(required - set(payload or {}))}"
        if payload.get("schema_version") != 1 or not payload.get("phase2_pair_shortlist"):
            return None, "invalid selected_experts.yaml schema/version/shortlist"
        return payload, "valid schema_version=1"
    except Exception as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"


def _read_parquet_metadata(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(
            path,
            columns=["usage", "allowed_for_training", "source_split", "dataset", "task", "configuration_id", "checkpoint_fingerprint", "sample_id", "query_id", "positive_id"],
        )
    except Exception:
        frame = pd.read_parquet(path)
        return frame


def _constraint_check(payload: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
    if payload is None:
        return False, "canonical selection unavailable", {}
    primary = pd.DataFrame([value for value in payload["phase2_pair_shortlist"] if value.get("intended_role") == "primary"])
    if primary.empty:
        return False, "no primary pair paths", {}
    protocol = payload.get("selection_protocol", {})
    constraints = protocol.get("diversity_constraints", {})
    vision_family = {value["canonical_name"]: value.get("architecture_family") for value in payload.get("vision_experts", [])}
    checks = {
        "at_most_two_per_vision": int(primary["vision_encoder"].value_counts().max()) <= int(constraints.get("maximum_paths_per_vision_encoder", 2)),
        "at_most_two_per_text": int(primary["text_encoder"].value_counts().max()) <= int(constraints.get("maximum_paths_per_text_encoder", 2)),
        "two_vision_families": len({vision_family.get(value, value) for value in primary["vision_encoder"]}) >= int(constraints.get("minimum_vision_architecture_families", 2)),
        "two_text_encoders": primary["text_encoder"].nunique() >= int(constraints.get("minimum_distinct_text_encoders", 2)),
    }
    return all(checks.values()), json.dumps(checks, sort_keys=True), checks


def _blf_duplicate_check(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if payload is None:
        return False, "canonical selection unavailable"
    primary = pd.DataFrame([value for value in payload["phase2_pair_shortlist"] if value.get("intended_role") == "primary"])
    for _, group in primary.groupby(["vision_encoder", "text_encoder"]):
        if len(group) > 1 and ((group["variant"] != "baseline") & (group["reliability_status"] != "supported")).any():
            return False, "an inconclusive/unsupported BLF duplicates its baseline"
    return True, "no inconclusive BLF duplicate in primary paths"


def _checkpoint_check(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if payload is None:
        return False, "canonical selection unavailable"
    missing = [value.get("checkpoint_path", "") for value in payload["phase2_pair_shortlist"] if not Path(str(value.get("checkpoint_path", ""))).exists()]
    return (not missing), ("all selected checkpoints exist" if not missing else f"missing checkpoints: {missing}")


def _registry_check(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if payload is None:
        return False, "canonical selection unavailable"
    vision = {value.get("canonical_name") for value in payload.get("vision_experts", [])}
    text = {value.get("canonical_name") for value in payload.get("text_experts", [])}
    missing = sorted((vision - set(VISION_ENCODERS)) | (text - set(TEXT_ENCODERS)))
    return (not missing), ("all selected encoders are registered" if not missing else f"unregistered encoders: {missing}")


def _training_negative_check(root: Path, protected_ids: set[str]) -> tuple[bool, str, bool, str]:
    directory = root / "results/phase15/training_hard_negatives"
    paths = [directory / "retrieval_train_hard_negatives.parquet", directory / "classification_train_hard_negatives.parquet"]
    schema_path = directory / "schema.json"
    if not schema_path.exists():
        return False, f"missing completed training-negative commit marker: {schema_path}", False, "leakage cannot be checked"
    if any(not value.exists() for value in paths):
        return False, f"missing training Parquet outputs: {[str(value) for value in paths if not value.exists()]}", False, "leakage cannot be checked"
    try:
        schema = json.loads(schema_path.read_text())
        hashes = schema.get("output_sha256", {})
        if schema.get("schema_version") != 1 or schema.get("status") != "complete":
            raise ValueError("training-negative commit marker is incomplete or has an unsupported schema")
        expected_hashes = {"retrieval": hashes.get("retrieval"), "classification": hashes.get("classification")}
        if not all(expected_hashes.values()):
            raise ValueError("training-negative commit marker is missing output hashes")
        frames: list[pd.DataFrame] = []
        for label, path in zip(("retrieval", "classification"), paths):
            if sha256_file(path) != expected_hashes[label]:
                raise ValueError(f"{label} training negatives do not match the committed output hash")
            frame = _read_parquet_metadata(path)
            validate_training_negative_metadata(frame)
            validate_no_protected_ids(frame, protected_ids)
            frames.append(frame)
        expected_rows = [int(schema.get("retrieval_rows", -1)), int(schema.get("classification_rows", -1))]
        if [len(frame) for frame in frames] != expected_rows:
            raise ValueError("training-negative row counts do not match the commit marker")
        expected_configs = set(map(str, schema.get("configurations", [])))
        observed_configs = set().union(*(set(frame["configuration_id"].astype(str)) for frame in frames))
        if not expected_configs or observed_configs != expected_configs:
            raise ValueError("training-negative configuration IDs do not match the commit marker")
        return True, "training negatives have train/development metadata", True, "zero protected-ID overlap"
    except Exception as exc:
        return False, f"invalid training negatives: {type(exc).__name__}: {exc}", False, f"leakage validation failed: {exc}"


def _diagnostic_check(root: Path) -> tuple[bool, str]:
    metadata = root / "results/phase15/hard_negatives/diagnostic_metadata.json"
    if not metadata.exists():
        return False, f"missing diagnostic restriction sidecar: {metadata}"
    try:
        payload = json.loads(metadata.read_text())
        valid = payload.get("usage") == "diagnostic_only" and payload.get("allowed_for_training") is False and payload.get("source_split") == "validation_or_test"
        return valid, "diagnostic negatives are explicitly non-trainable" if valid else "diagnostic restriction metadata is invalid"
    except Exception as exc:
        return False, f"unreadable diagnostic metadata: {exc}"


def run_phase2_readiness(
    project_root: str | Path,
    *,
    protected_ids: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    evidence_root = resolve_evaluation_root(root)
    checks: list[dict[str, Any]] = []

    def add(number: int, name: str, passed: bool, details: str, *, severity: str = "required") -> None:
        checks.append({"check_number": number, "check": name, "severity": severity, "passed": bool(passed), "details": details})

    notebook = root / "notebooks/01_experiment_workflow.ipynb"
    try:
        defaults = _read_notebook_defaults(notebook)
        add(1, "safe notebook defaults", defaults == SAFE_DEFAULTS, f"observed={defaults}")
    except Exception as exc:
        add(1, "safe notebook defaults", False, f"notebook unreadable: {exc}")
    file_checks = {
        2: ("canonical retrieval results", root / "results/alignment_matrix_best_clean.csv", {"vision_encoder", "text_encoder", "variant"}),
        3: ("seed-sweep results", root / "results/seed_sweep_results.csv", {"seed", "variant"}),
        4: ("multi-task development outputs", evidence_root / "task_results_long.csv", {"config_id", "task", "metric", "value", "dataset_split"}),
        6: ("retrieval complementarity", root / "results/phase15/complementarity/retrieval_pairwise.csv", {"expert_a", "expert_b"}),
        7: ("classification complementarity", root / "results/phase15/complementarity/classification_pairwise.csv", {"dataset", "configuration_a", "configuration_b"}),
        8: ("retrieval oracle", root / "results/phase15/oracle/retrieval_oracle.csv", {"expert_a", "expert_b"}),
        9: ("classification oracle", root / "results/phase15/oracle/classification_oracle.csv", {"dataset", "subset_id", "oracle_top1_accuracy"}),
        10: ("cross-task matrix", root / "results/phase15/cross_task/task_performance_matrix_normalised.csv", {"config_id"}),
        11: ("cross-task rank analysis", root / "results/phase15/cross_task/configuration_rankings.csv", {"config_id"}),
        12: ("efficiency profiles", root / "results/phase15/efficiency/full_pairs.csv", {"config_id", "latency_ms", "peak_allocated_bytes"}),
        13: ("seed reliability mapping", root / "results/phase15/expert_selection/seed_reliability.csv", {"config_id", "seed_runs", "reliability_status"}),
        20: ("development/final evaluation protocol", root / "results/phase15/methodology/evaluation_protocol.json", None),
    }
    for number, (name, path, columns) in file_checks.items():
        passed, details, _ = _file_check(path, columns)
        if number == 4 and passed:
            try:
                frame = pd.read_csv(path)
                protected_tokens = ("official_test", "complete_dataset")
                if not evidence_root.name.endswith("_development") or frame["dataset_split"].astype(str).str.contains("|".join(protected_tokens), case=False).any():
                    passed = False
                    details = "historical outputs are exploratory; development-split multi-task predictions are not yet present"
            except Exception as exc:
                passed, details = False, str(exc)
        add(number, name, passed, details)
    coverage_path = root / "results/phase15/prediction_validation/coverage_report.csv"
    coverage_ok, coverage_details, _ = _file_check(coverage_path, {"valid", "direction", "configuration_id", "task", "dataset_split"})
    if coverage_ok:
        coverage = pd.read_csv(coverage_path)
        protected_split = coverage["dataset_split"].astype(str).str.contains("official_test|complete_dataset", case=False)
        coverage_ok = bool(evidence_root.name.endswith("_development") and not coverage.empty and coverage["valid"].map(bool).all() and not protected_split.any())
        coverage_details = f"{int(coverage['valid'].sum())}/{len(coverage)} exports valid"
    add(5, "strict prediction coverage", coverage_ok, coverage_details)
    selected, selected_details = _selected_yaml(root)
    add(14, "canonical selected_experts.yaml", selected is not None, selected_details)
    valid_constraints, constraint_details, _ = _constraint_check(selected)
    add(15, "pair-shortlist diversity constraints", valid_constraints, constraint_details)
    no_duplicate, duplicate_details = _blf_duplicate_check(selected)
    add(16, "no inconclusive BLF duplicate", no_duplicate, duplicate_details)
    protected = load_protected_test_ids(root) if protected_ids is None else protected_ids
    training_ok, training_details, leakage_ok, leakage_details = _training_negative_check(root, protected)
    add(17, "training hard negatives", training_ok, training_details)
    add(18, "training-negative leakage validation", leakage_ok, leakage_details)
    diagnostic_ok, diagnostic_details = _diagnostic_check(root)
    add(19, "diagnostic negatives non-trainable", diagnostic_ok, diagnostic_details)
    checkpoint_ok, checkpoint_details = _checkpoint_check(selected)
    add(21, "selected checkpoints", checkpoint_ok, checkpoint_details)
    registry_ok, registry_details = _registry_check(selected)
    add(22, "selected encoder registry coverage", registry_ok, registry_details)
    compositional = root / "results/phase15/compositional/task_status.csv"
    add(23, "optional compositional benchmarks", compositional.exists(), "optional task status is documented" if compositional.exists() else "optional status report missing", severity="warning")
    frame = pd.DataFrame(checks).sort_values("check_number", kind="stable")
    failed_required = frame[(frame["severity"].eq("required")) & (~frame["passed"])]
    failed_warnings = frame[(frame["severity"].eq("warning")) & (~frame["passed"])]
    status = "NOT_READY" if not failed_required.empty else "READY_WITH_WARNINGS" if not failed_warnings.empty else "READY"
    output = root / "results/phase15/phase2_readiness"
    atomic_csv(frame, output / "readiness_checks.csv")
    report = {
        "status": status,
        "ready_to_implement_phase2": status in {"READY", "READY_WITH_WARNINGS"},
        "required_failures": int(len(failed_required)),
        "warnings": int(len(failed_warnings)),
        "checks": frame.to_dict("records"),
        "phase2_implemented": False,
    }
    atomic_json(report, output / "readiness_report.json")
    markdown = [
        "# Phase 2 readiness report",
        "",
        f"**Status: {status}**",
        "",
        "Phase 2 is not implemented in this repository state. It must not begin while any required check fails.",
        "",
        frame.to_markdown(index=False),
    ]
    atomic_text("\n".join(markdown) + "\n", output / "readiness_report.md")
    return report
