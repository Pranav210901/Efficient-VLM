"""Strict, manifest-driven sample prediction and cache coverage validation."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from src.common.configuration_ids import canonical_configuration_id

from .io_utils import atomic_csv, atomic_json, stat_fingerprint
from .evidence import resolve_evaluation_root


CLASSIFICATION_TASKS = ("cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot")
TASK_SPLITS = {
    "coco_retrieval": "coco_val2017_development",
    "cifar100_zeroshot": "cifar100_official_test_exploratory",
    "pets_zeroshot": "oxford_iiit_pet_official_test_exploratory",
    "eurosat_zeroshot": "eurosat_complete_dataset_exploratory",
}
CLASSIFICATION_REQUIRED = {
    "task",
    "sample_id",
    "target_index",
    "target",
    "prediction_index",
    "prediction",
    "correct",
    "confidence",
    "correct_class_score",
    "highest_incorrect_index",
    "highest_incorrect_score",
    "classification_margin",
    "top1_margin",
}
RETRIEVAL_REQUIRED = {
    "direction",
    "query_id",
    "correct_target_rank",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "top_candidate_ids",
    "top_candidate_scores",
}


def _read_cache(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict) or not isinstance(value.get("payload"), dict):
        raise ValueError("cache must contain metadata and payload dictionaries")
    return value["metadata"], value["payload"]


def _path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return rows, f"line {line_number} is not a JSON object"
                rows.append(value)
    except Exception as exc:
        return rows, f"{type(exc).__name__}: {exc}"
    return rows, None


def validate_retrieval_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    direction: str,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    values = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    columns = set().union(*(row.keys() for row in values)) if values else set()
    missing_columns = RETRIEVAL_REQUIRED - columns
    ids = [str(row.get("query_id")) for row in values]
    duplicates = len(ids) - len(set(ids))
    expected = set(str(value) for value in expected_ids) if expected_ids is not None else set(ids)
    missing_ids = expected - set(ids)
    unexpected_ids = set(ids) - expected
    nonfinite = 0
    invalid = 0
    for row in values:
        rank = row.get("correct_target_rank")
        if row.get("direction") != direction or not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            invalid += 1
            continue
        if bool(row.get("recall_at_1")) != (rank <= 1) or bool(row.get("recall_at_5")) != (rank <= 5) or bool(row.get("recall_at_10")) != (rank <= 10):
            invalid += 1
        numeric = list(row.get("top_candidate_scores") or [])
        numeric.extend(row.get(key) for key in ("positive_similarity", "hardest_negative_similarity", "retrieval_margin") if key in row)
        nonfinite += sum(not _finite(value) for value in numeric)
        if {"positive_similarity", "hardest_negative_similarity", "retrieval_margin"}.issubset(row):
            calculated = float(row["positive_similarity"]) - float(row["hardest_negative_similarity"])
            if not math.isclose(calculated, float(row["retrieval_margin"]), rel_tol=1e-5, abs_tol=1e-6):
                invalid += 1
        if row.get("hardest_negative_id") is not None and row.get("target_image_id") is not None:
            if str(row["hardest_negative_id"]) == str(row["target_image_id"]):
                invalid += 1
    return {
        "observed_samples": len(values),
        "missing_count": len(missing_ids),
        "unexpected_count": len(unexpected_ids),
        "duplicate_count": duplicates,
        "nonfinite_score_count": nonfinite,
        "invalid_prediction_count": invalid,
        "schema_status": "valid" if not missing_columns else "invalid",
        "missing_columns": sorted(missing_columns),
        "valid": not missing_columns and not duplicates and not missing_ids and not unexpected_ids and not nonfinite and not invalid,
    }


def validate_classification_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    task: str,
    expected_ids: Iterable[str] | None = None,
    expected_classes: int | None = None,
) -> dict[str, Any]:
    values = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    columns = set().union(*(row.keys() for row in values)) if values else set()
    missing_columns = CLASSIFICATION_REQUIRED - columns
    ids = [str(row.get("sample_id")) for row in values]
    duplicates = len(ids) - len(set(ids))
    expected = set(str(value) for value in expected_ids) if expected_ids is not None else set(ids)
    missing_ids = expected - set(ids)
    unexpected_ids = set(ids) - expected
    nonfinite = 0
    invalid = 0
    observed_labels: set[int] = set()
    for row in values:
        target = row.get("target_index")
        prediction = row.get("prediction_index")
        if not isinstance(target, int) or isinstance(target, bool) or not isinstance(prediction, int) or isinstance(prediction, bool):
            invalid += 1
            continue
        observed_labels.add(target)
        if row.get("task") != task or bool(row.get("correct")) != (prediction == target):
            invalid += 1
        if expected_classes is not None and not (0 <= target < expected_classes and 0 <= prediction < expected_classes):
            invalid += 1
        numeric = [row.get(key) for key in ("confidence", "correct_class_score", "highest_incorrect_score", "classification_margin", "top1_margin")]
        numeric.extend(item.get("score") for item in (row.get("top5") or []) if isinstance(item, dict))
        nonfinite += sum(not _finite(value) for value in numeric)
        if _finite(row.get("correct_class_score")) and _finite(row.get("highest_incorrect_score")) and _finite(row.get("classification_margin")):
            margin = float(row["correct_class_score"]) - float(row["highest_incorrect_score"])
            if not math.isclose(margin, float(row["classification_margin"]), rel_tol=1e-5, abs_tol=1e-6):
                invalid += 1
    return {
        "observed_samples": len(values),
        "missing_count": len(missing_ids),
        "unexpected_count": len(unexpected_ids),
        "duplicate_count": duplicates,
        "nonfinite_score_count": nonfinite,
        "invalid_prediction_count": invalid,
        "observed_target_classes": len(observed_labels),
        "schema_status": "valid" if not missing_columns else "invalid",
        "missing_columns": sorted(missing_columns),
        "valid": not missing_columns and not duplicates and not missing_ids and not unexpected_ids and not nonfinite and not invalid,
    }


def _checkpoint_status(config_id: str, checkpoint: Path, metadata: dict[str, Any]) -> tuple[str, list[str]]:
    issues: list[str] = []
    if not checkpoint.exists():
        return "missing", [f"checkpoint missing: {checkpoint}"]
    try:
        checkpoint_id = canonical_configuration_id(checkpoint.parent.name)
        if checkpoint_id != config_id:
            issues.append(f"checkpoint configuration {checkpoint_id} does not match {config_id}")
    except ValueError as exc:
        issues.append(str(exc))
    cached_path = Path(str(metadata.get("checkpoint", ""))).resolve()
    if cached_path != checkpoint.resolve():
        issues.append("cache checkpoint path differs from the manifest")
    cached_mtime = metadata.get("checkpoint_mtime_ns")
    if cached_mtime is None:
        status = "missing"
        issues.append("cache has no checkpoint fingerprint metadata")
    elif int(cached_mtime) != checkpoint.stat().st_mtime_ns:
        status = "mismatch"
        issues.append("checkpoint mtime differs from the cached fingerprint")
    else:
        status = "matched_legacy_stat"
    return (status if not issues or status == "matched_legacy_stat" and len(issues) == 0 else "mismatch"), issues


def _atomic_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@torch.inference_mode()
def materialize_t2i_predictions(cache_path: str | Path, output_path: str | Path, device: str = "cpu", chunk_size: int = 256, topk: int = 10) -> Path:
    if chunk_size < 1 or topk < 1:
        raise ValueError("chunk_size and topk must be positive")
    _, payload = _read_cache(Path(cache_path))
    images = payload["image_embeddings"].float()
    texts = payload["text_embeddings"].float()
    image_ids = [str(value) for value in payload["image_ids"]]
    owners = [str(value) for value in payload["text_image_ids"]]
    if images.ndim != 2 or texts.ndim != 2 or images.size(1) != texts.size(1):
        raise ValueError("retrieval cache embeddings are invalid")
    if len(image_ids) != images.size(0) or len(owners) != texts.size(0) or len(set(image_ids)) != len(image_ids):
        raise ValueError("retrieval cache IDs do not match embedding rows")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA t2i materialisation requested but no CUDA device is visible")
    gallery = images.to(target)
    id_to_index = {value: index for index, value in enumerate(image_ids)}
    try:
        target_indices = torch.tensor([id_to_index[value] for value in owners], device=target)
    except KeyError as exc:
        raise ValueError(f"caption owner is outside the image gallery: {exc.args[0]}") from exc

    def rows() -> Iterable[dict[str, Any]]:
        for start in range(0, len(owners), chunk_size):
            stop = min(start + chunk_size, len(owners))
            similarity = texts[start:stop].to(target) @ gallery.t()
            local_targets = target_indices[start:stop]
            target_scores = similarity.gather(1, local_targets[:, None]).squeeze(1)
            ranks = similarity.gt(target_scores[:, None]).sum(dim=1) + 1
            masked = similarity.clone()
            masked[torch.arange(stop - start, device=target), local_targets] = -torch.inf
            negative_scores, negative_indices = masked.max(dim=1)
            selected_scores, selected_indices = similarity.topk(min(topk, similarity.size(1)), dim=1)
            for local, query_index in enumerate(range(start, stop)):
                rank = int(ranks[local])
                positive = float(target_scores[local].cpu())
                hardest = float(negative_scores[local].cpu())
                chosen = selected_indices[local].cpu().tolist()
                yield {
                    "direction": "t2i",
                    "query_id": f"caption:{query_index:06d}",
                    "query_index": query_index,
                    "target_image_id": owners[query_index],
                    "correct_target_rank": rank,
                    "recall_at_1": rank <= 1,
                    "recall_at_5": rank <= 5,
                    "recall_at_10": rank <= 10,
                    "positive_similarity": positive,
                    "hardest_negative_similarity": hardest,
                    "hardest_negative_id": image_ids[int(negative_indices[local])],
                    "retrieval_margin": positive - hardest,
                    "top_candidate_ids": [image_ids[int(index)] for index in chosen],
                    "top_candidate_scores": [float(value) for value in selected_scores[local].cpu()],
                }

    destination = Path(output_path)
    _atomic_jsonl(rows(), destination)
    return destination


def _base_record(row: Any, direction: str, source: Path, split: str) -> dict[str, Any]:
    return {
        "configuration_id": str(row.config_id),
        "task": str(row.task),
        "direction": direction,
        "mode": str(row.mode),
        "dataset_split": split,
        "source_path": str(source),
        "exists": source.exists(),
        "readable": False,
        "valid": False,
        "expected_samples": 0,
        "observed_samples": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "duplicate_count": 0,
        "nonfinite_score_count": 0,
        "invalid_prediction_count": 0,
        "class_count": None,
        "expected_class_count": None,
        "checkpoint_fingerprint_status": "missing",
        "checkpoint_stat_fingerprint": None,
        "schema_status": "missing",
        "cache_metadata_status": "invalid",
        "issues": "",
    }


def run_prediction_validation(
    project_root: str | Path,
    *,
    mode: str = "full",
    strict: bool = False,
    materialize_missing_retrieval: bool = False,
    device: str | None = None,
    evaluation_root: str | Path | None = None,
) -> pd.DataFrame:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    root = Path(project_root).resolve()
    evidence_root = resolve_evaluation_root(root, evaluation_root)
    manifest_path = evidence_root / "evaluation_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"config_id", "task", "status", "checkpoint", "cache", "predictions", "samples"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Evaluation manifest is missing columns: {sorted(missing)}")
    if "mode" not in manifest:
        manifest["mode"] = manifest["cache"].map(lambda value: "full" if str(value).endswith("__full.pt") else "smoke")
    selected = manifest[manifest["status"].eq("complete") & manifest["mode"].eq(mode)].copy()
    if selected.duplicated(["config_id", "task"]).any():
        duplicates = selected.loc[selected.duplicated(["config_id", "task"], keep=False), ["config_id", "task"]]
        raise ValueError(f"Duplicate manifest records: {duplicates.to_dict('records')}")
    records: list[dict[str, Any]] = []
    derived = root / "results/phase15/predictions"
    target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    for row in selected.sort_values(["config_id", "task"]).itertuples():
        cache_path = _path(row.cache, root)
        checkpoint = _path(row.checkpoint, root)
        cache_issues: list[str] = []
        try:
            metadata, payload = _read_cache(cache_path)
            cache_readable = True
        except Exception as exc:
            metadata, payload = {}, {}
            cache_readable = False
            cache_issues.append(f"cache unreadable: {type(exc).__name__}: {exc}")
        fingerprint_status, fingerprint_issues = _checkpoint_status(str(row.config_id), checkpoint, metadata) if cache_readable else ("missing", [])
        cache_issues.extend(fingerprint_issues)
        mode_consistent = metadata.get("mode") == mode
        if cache_readable and not mode_consistent:
            cache_issues.append(f"cache mode {metadata.get('mode')!r} does not match {mode!r}")
        cache_metadata_status = "valid" if cache_readable and mode_consistent and fingerprint_status in {"matched", "matched_legacy_stat"} else "invalid"
        split = str(getattr(row, "dataset_split", TASK_SPLITS.get(str(row.task), "unknown")))
        if row.task == "coco_retrieval":
            if cache_readable:
                image_ids = [str(value) for value in payload.get("image_ids", [])]
                owners = [str(value) for value in payload.get("text_image_ids", [])]
                embeddings_finite = bool(torch.isfinite(payload.get("image_embeddings", torch.tensor(float("nan")))).all() and torch.isfinite(payload.get("text_embeddings", torch.tensor(float("nan")))).all())
                mapping_valid = len(image_ids) == len(set(image_ids)) and set(owners).issubset(set(image_ids)) and len(owners) == int(payload.get("text_embeddings", torch.empty(0)).shape[0])
                if not embeddings_finite:
                    cache_issues.append("retrieval cache contains non-finite embeddings")
                if not mapping_valid:
                    cache_issues.append("retrieval valid-caption ownership mapping is incomplete")
            else:
                image_ids, owners, embeddings_finite, mapping_valid = [], [], False, False
            i2t_path = _path(row.predictions, root)
            for direction, source, expected_ids in (
                ("i2t", i2t_path, image_ids),
                (
                    "t2i",
                    _path(getattr(row, "predictions_t2i"), root)
                    if hasattr(row, "predictions_t2i") and pd.notna(getattr(row, "predictions_t2i"))
                    else derived / f"{row.config_id}__coco_t2i__{mode}.jsonl",
                    [f"caption:{index:06d}" for index in range(len(owners))],
                ),
            ):
                if direction == "t2i" and not source.exists() and materialize_missing_retrieval and cache_readable:
                    materialize_t2i_predictions(cache_path, source, device=target_device)
                record = _base_record(row, direction, source, split)
                record["expected_samples"] = len(expected_ids)
                record["checkpoint_fingerprint_status"] = fingerprint_status
                record["checkpoint_stat_fingerprint"] = stat_fingerprint(checkpoint) if checkpoint.exists() else None
                record["cache_metadata_status"] = cache_metadata_status
                issues = list(cache_issues)
                if source.exists():
                    values, error = _load_jsonl(source)
                    record["readable"] = error is None
                    if error:
                        issues.append(error)
                    else:
                        result = validate_retrieval_rows(values, direction=direction, expected_ids=expected_ids)
                        record.update({key: result[key] for key in result if key != "valid"})
                        if result["missing_columns"]:
                            issues.append(f"missing columns: {result['missing_columns']}")
                        record["valid"] = bool(result["valid"] and cache_metadata_status == "valid" and embeddings_finite and mapping_valid)
                else:
                    issues.append("prediction file missing")
                record["issues"] = "; ".join(issues)
                records.append(record)
        elif row.task in CLASSIFICATION_TASKS:
            source = _path(row.predictions, root)
            record = _base_record(row, "classification", source, split)
            expected_ids = [str(value) for value in payload.get("sample_ids", [])] if cache_readable else []
            expected_classes = len(payload.get("class_names", [])) if cache_readable else int(row.classes) if hasattr(row, "classes") and pd.notna(row.classes) else None
            record["expected_samples"] = len(expected_ids) if expected_ids else int(row.samples)
            record["expected_class_count"] = expected_classes
            record["class_count"] = expected_classes
            record["checkpoint_fingerprint_status"] = fingerprint_status
            record["checkpoint_stat_fingerprint"] = stat_fingerprint(checkpoint) if checkpoint.exists() else None
            record["cache_metadata_status"] = cache_metadata_status
            issues = list(cache_issues)
            if cache_readable:
                logits = payload.get("logits")
                targets = payload.get("targets")
                if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor) or logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.numel():
                    issues.append("classification cache logits/targets are invalid")
                elif not torch.isfinite(logits).all():
                    issues.append("classification cache contains non-finite logits")
                if metadata.get("task") != row.task or int(metadata.get("dataset_samples", -1)) != int(row.samples) or int(metadata.get("dataset_classes", -1)) != int(expected_classes or -1):
                    issues.append("classification cache task/sample/class metadata is inconsistent")
            if source.exists():
                values, error = _load_jsonl(source)
                record["readable"] = error is None
                if error:
                    issues.append(error)
                else:
                    result = validate_classification_rows(values, task=str(row.task), expected_ids=expected_ids or None, expected_classes=expected_classes)
                    record.update({key: result[key] for key in result if key not in {"valid", "observed_target_classes"}})
                    if result["missing_columns"]:
                        issues.append(f"missing columns: {result['missing_columns']}")
                    record["valid"] = bool(result["valid"] and cache_metadata_status == "valid" and not issues)
            else:
                issues.append("prediction file missing")
            record["issues"] = "; ".join(issues)
            records.append(record)
    frame = pd.DataFrame(records)
    output = root / "results/phase15/prediction_validation"
    atomic_csv(frame, output / "coverage_report.csv")
    missing_frame = frame.loc[~frame["valid"]].copy() if not frame.empty else frame.copy()
    atomic_csv(missing_frame, output / "missing_predictions.csv")
    summary = {
        "mode": mode,
        "strict": strict,
        "expected_exports": len(frame),
        "valid_exports": int(frame["valid"].sum()) if not frame.empty else 0,
        "missing_or_invalid_exports": int((~frame["valid"]).sum()) if not frame.empty else 0,
        "expected_samples": int(frame["expected_samples"].sum()) if not frame.empty else 0,
        "observed_samples": int(frame["observed_samples"].sum()) if not frame.empty else 0,
        "duplicate_count": int(frame["duplicate_count"].sum()) if not frame.empty else 0,
        "nonfinite_score_count": int(frame["nonfinite_score_count"].sum()) if not frame.empty else 0,
        "invalid_prediction_count": int(frame["invalid_prediction_count"].sum()) if not frame.empty else 0,
        "status": "valid" if not frame.empty and frame["valid"].all() else "invalid",
        "outputs": {
            "coverage": str(output / "coverage_report.csv"),
            "missing": str(output / "missing_predictions.csv"),
        },
    }
    atomic_json(summary, output / "coverage_report.json")
    if strict and summary["status"] != "valid":
        failed = frame.loc[~frame["valid"], ["configuration_id", "task", "direction", "issues"]]
        raise RuntimeError(f"Strict prediction coverage validation failed for {len(failed)} exports; see {output / 'missing_predictions.csv'}")
    return frame
