from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .audit import run_phase15_audit
from .classification_complementarity import run_classification_complementarity
from .complementarity import retrieval_complementarity
from .compositional_status import run_compositional_status
from .cross_task_analysis import run_cross_task_analysis
from .evaluation_protocol import write_evaluation_protocol_outputs
from .evidence import resolve_evaluation_root
from .io_utils import atomic_csv, atomic_json, atomic_text
from .oracle_analysis import oracle_recall, run_classification_oracle_analysis
from .prediction_validation import run_prediction_validation
from .seed_reliability import run_seed_reliability_mapping


def _manifest_mode(row: pd.Series) -> str | None:
    mode = row.get("mode")
    if mode in {"smoke", "full"}:
        return str(mode)
    cache = row.get("cache")
    if pd.notna(cache):
        name = Path(str(cache)).name
        if name.endswith("__full.pt"):
            return "full"
        if name.endswith("__smoke.pt"):
            return "smoke"
    return None


def full_retrieval_prediction_files(project_root: str | Path, evaluation_root: str | Path | None = None) -> list[tuple[str, Path]]:
    root = Path(project_root).resolve()
    manifest_path = resolve_evaluation_root(root, evaluation_root) / "evaluation_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run Phase 1 multi-task evaluation first: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"config_id", "task", "status", "predictions"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Evaluation manifest is missing columns: {sorted(missing_columns)}")
    manifest = manifest.copy()
    manifest["resolved_mode"] = manifest.apply(_manifest_mode, axis=1)
    selected = manifest[
        manifest["task"].eq("coco_retrieval")
        & manifest["status"].eq("complete")
        & manifest["resolved_mode"].eq("full")
    ]
    duplicates = selected[selected.duplicated("config_id", keep=False)]["config_id"].unique().tolist()
    if duplicates:
        raise ValueError(f"Multiple full COCO prediction records found for: {sorted(duplicates)}")
    files: list[tuple[str, Path]] = []
    for row in selected.sort_values("config_id").itertuples():
        path = Path(str(row.predictions))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Missing full COCO predictions for {row.config_id}: {path}")
        files.append((str(row.config_id), path))
    if not files:
        raise FileNotFoundError(f"No completed full COCO predictions are listed in {manifest_path}")
    return files


def run_phase15(
    project_root: str | Path,
    anchor_text: str = "all_minilm_l6_v2",
    anchor_vision: str = "dinov2_vits14",
    *,
    mode: str = "full",
    strict_validation: bool = False,
    materialize_missing_t2i: bool = False,
    validation_device: str | None = None,
    include_classification_triples: bool = True,
    reuse_prediction_validation: bool = False,
    evaluation_root: str | Path | None = None,
) -> dict[str, object]:
    """Run Phase 1.5 derived analyses without training or changing checkpoints.

    Strict full execution fails on incomplete sample coverage. Defaults remain
    backwards-compatible and do not launch GPU work or t2i materialisation.
    """
    root = Path(project_root).resolve()
    evidence_root = resolve_evaluation_root(root, evaluation_root)
    source = evidence_root / "predictions"
    if not source.exists():
        raise FileNotFoundError(f"Run Phase 1 multi-task evaluation first: {source}")
    output = root / "results/phase15"
    predictions_out = output / "predictions"
    complementarity_out = output / "complementarity"
    oracle_out = output / "oracle"
    for path in (predictions_out, complementarity_out, oracle_out, output / "cross_task", output / "efficiency", output / "hard_negatives", output / "expert_selection"):
        path.mkdir(parents=True, exist_ok=True)
    audit_before = run_phase15_audit(root)
    enhanced_protocol = (root / "configs/evaluation_protocol.yaml").exists()
    methodology = write_evaluation_protocol_outputs(root) if enhanced_protocol else {}
    compositional = run_compositional_status(root) if enhanced_protocol else pd.DataFrame()
    coverage = pd.DataFrame()
    manifest_columns = set(pd.read_csv(evidence_root / "evaluation_manifest.csv", nrows=0).columns)
    saved_coverage = root / "results/phase15/prediction_validation/coverage_report.csv"
    if reuse_prediction_validation and saved_coverage.exists():
        coverage = pd.read_csv(saved_coverage)
        if strict_validation and (coverage.empty or not coverage["valid"].map(bool).all()):
            raise RuntimeError(f"Saved strict prediction coverage is invalid; rerun Step 13: {saved_coverage}")
    elif {"checkpoint", "samples", "predictions"}.issubset(manifest_columns):
        coverage = run_prediction_validation(
            root,
            mode=mode,
            strict=strict_validation,
            materialize_missing_retrieval=materialize_missing_t2i,
            device=validation_device,
            evaluation_root=evidence_root,
        )
    elif strict_validation:
        raise ValueError("Strict prediction validation requires checkpoint, samples, and predictions manifest columns")
    files = full_retrieval_prediction_files(root, evidence_root)
    tables = {}
    for config_id, path in files:
        frame = pd.read_json(path, lines=True); tables[config_id] = frame
        target = predictions_out / path.name
        if not target.exists(): frame.to_json(target, orient="records", lines=True)
    pair_rows = []; oracle_rows = []
    names = sorted(tables)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            merged = tables[left].merge(tables[right], on=["direction", "query_id"], suffixes=("_a", "_b"))
            values = retrieval_complementarity(merged.correct_target_rank_a, merged.correct_target_rank_b)
            pair_rows.append({"expert_a": left, "expert_b": right, **values})
            oracle_rows.append({"expert_a": left, "expert_b": right, **oracle_recall([merged.correct_target_rank_a, merged.correct_target_rank_b])})
    pair_path = complementarity_out / "retrieval_pairwise.csv"
    oracle_path = oracle_out / "retrieval_oracle.csv"
    atomic_csv(pd.DataFrame(pair_rows), pair_path)
    atomic_csv(pd.DataFrame(oracle_rows), oracle_path)
    manifest = pd.read_csv(evidence_root / "evaluation_manifest.csv")
    has_classification = manifest["task"].isin(["cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"]).any()
    classification_complementarity: dict[str, object] | None = None
    classification_oracle: dict[str, object] | None = None
    if has_classification:
        classification_complementarity = run_classification_complementarity(root, mode=mode, evaluation_root=evidence_root)
        classification_oracle = run_classification_oracle_analysis(root, mode=mode, include_triples=include_classification_triples, evaluation_root=evidence_root)
    cross_task: dict[str, object] | None = None
    if (evidence_root / "task_results_long.csv").exists():
        cross_task = run_cross_task_analysis(root, vision_anchor_text=anchor_text, text_anchor_vision=anchor_vision, evaluation_root=evidence_root)
    seed_reliability_rows = 0
    if (root / "results/seed_sweep_results.csv").exists():
        seed_reliability_rows = len(run_seed_reliability_mapping(root))
    summary = {
        "anchor_text": anchor_text,
        "anchor_vision": anchor_vision,
        "mode": mode,
        "evaluation_root": str(evidence_root),
        "strict_validation": strict_validation,
        "prediction_files": len(files),
        "coverage_exports": len(coverage),
        "coverage_valid_exports": int(coverage["valid"].sum()) if not coverage.empty else 0,
        "config_ids": sorted(tables),
        "classification_complementarity": classification_complementarity,
        "classification_oracle": classification_oracle,
        "cross_task": cross_task,
        "seed_reliability_rows": seed_reliability_rows,
        "methodology": methodology,
        "compositional_status": compositional.to_dict("records"),
        "audit_rows_before": len(audit_before),
        "oracle_is_non_deployable": True,
        "historical_classification_status": "exploratory/development; not untouched final-test evidence",
        "outputs": [str(pair_path), str(oracle_path)],
    }
    atomic_json(summary, output / "phase15_summary.json")
    atomic_text(
        "# Phase 1.5 summary\n\n"
        "Oracle values are non-deployable diagnostic upper bounds. Historical classification test outputs are exploratory; future selection must use the frozen development protocol.\n",
        output / "phase15_summary.md",
    )
    run_phase15_audit(root)
    return summary
