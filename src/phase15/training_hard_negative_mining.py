"""Phase 2 training-negative mining restricted to train/development splits.

Full execution is intentionally explicit and GPU-heavy. Notebook Run All
never calls it under the committed defaults. Held-out diagnostic negatives in
results/phase15/hard_negatives are neither read as training data nor replaced.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Subset

from src.data.transforms import build_image_transform
from src.multitask.checkpoint_selection import load_frozen_model, resolve_best_checkpoints
from src.multitask.classification_evaluator import IndexedClassificationDataset, evaluate_zeroshot_classification
from src.multitask.config import PROMPT_TEMPLATES, VISION_ENCODERS
from src.multitask.embedding_cache import load_cache, save_cache
from src.multitask.prompt_templates import clean_class_name
from src.multitask.runner import _encode_coco, _read_coco

from .evaluation_protocol import create_eurosat_split_manifest, protected_test_ids
from .hard_negative_mining import enriched_classification_hard_negatives
from .io_utils import atomic_csv, atomic_json, sha256_file, stat_fingerprint


TRAINING_USAGE = "phase2_training"
TRAINING_HARD_NEGATIVES_PER_POSITIVE = 8
CLASSIFICATION_TRAINING_TASKS = ("cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot")


def annotate_training_rows(
    frame: pd.DataFrame,
    *,
    source_split: str,
    dataset: str,
    task: str,
    configuration_id: str,
    checkpoint_fingerprint: str,
    source_checkpoint: str | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if "task" in result and "source_task" not in result:
        result = result.rename(columns={"task": "source_task"})
    values = {
        "usage": TRAINING_USAGE,
        "allowed_for_training": True,
        "source_split": source_split,
        "dataset": dataset,
        "task": task,
        "configuration_id": configuration_id,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "source_checkpoint": source_checkpoint or "not_recorded",
    }
    for position, (column, value) in enumerate(values.items()):
        if column in result:
            result = result.drop(columns=[column])
        result.insert(position, column, value)
    return result


def validate_training_negative_metadata(frame: pd.DataFrame) -> None:
    required = {"usage", "allowed_for_training", "source_split", "dataset", "task", "configuration_id", "checkpoint_fingerprint"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training negatives are missing metadata: {sorted(missing)}")
    if not frame["usage"].eq(TRAINING_USAGE).all() or not frame["allowed_for_training"].map(bool).all():
        raise ValueError("Every training-negative row must be explicitly trainable")
    protected_words = ("official_test", "_test", "validation_or_test", "coco_val", "evaluation_only", "complete_dataset")
    invalid = frame["source_split"].astype(str).str.lower().map(lambda value: any(word in value for word in protected_words))
    if invalid.any():
        raise ValueError(f"Protected/evaluation splits appear in training negatives: {sorted(frame.loc[invalid, 'source_split'].unique())}")


def validate_no_protected_ids(frame: pd.DataFrame, protected_ids: set[str]) -> None:
    id_columns = [column for column in ("sample_id", "query_id", "positive_id") if column in frame]
    leaked: set[str] = set()
    for column in id_columns:
        leaked.update(set(frame[column].dropna().astype(str)) & protected_ids)
    if leaked:
        raise ValueError(f"Protected test IDs entered training negatives: {sorted(leaked)[:10]}")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for resumable training-negative Parquet outputs") from exc
    return pa, pq


def _write_parquet_chunks(chunks: Iterable[pd.DataFrame], path: Path) -> int:
    pa, pq = _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for frame in chunks:
            if frame.empty:
                continue
            validate_training_negative_metadata(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
        if writer is None:
            raise ValueError("No training-negative rows were produced")
        writer.close()
        writer = None
        temporary.replace(path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return rows


def _combine_parquet(shards: list[Path], destination: Path) -> int:
    pa, pq = _require_pyarrow()
    if not shards:
        raise ValueError("At least one Parquet shard is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for shard in shards:
            file = pq.ParquetFile(shard)
            for batch in file.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                rows += table.num_rows
        if writer is None:
            raise ValueError("Parquet shards were empty")
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return rows


def _shard_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _source_fingerprint(paths: Iterable[Path]) -> str:
    values: list[str] = []
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        values.append(f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(values).encode()).hexdigest()[:20]


def _classification_source_fingerprint(root: Path) -> str:
    manifest = create_eurosat_split_manifest(root)
    sources = (
        root / "data/multitask/cifar100/cifar-100-python/train",
        root / "data/multitask/oxford-iiit-pet/oxford-iiit-pet/annotations/trainval.txt",
        root / "data/multitask/eurosat/eurosat/2750",
        manifest,
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Training classification sources are missing: {missing}")
    return _source_fingerprint(sources)


def _shard_identity(
    *,
    configuration_id: str,
    checkpoint_fingerprint: str,
    task: str,
    mode: str,
    count: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "configuration_id": configuration_id,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "task": task,
        "mode": mode,
        "count": int(count),
        "source_fingerprint": source_fingerprint,
    }


def _parquet_rank_extrema(file: Any, column: str) -> tuple[int | None, int | None]:
    try:
        column_index = file.schema.names.index(column)
    except ValueError:
        return None, None
    minima: list[int] = []
    maxima: list[int] = []
    for index in range(file.metadata.num_row_groups):
        statistics = file.metadata.row_group(index).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            return None, None
        minima.append(int(statistics.min))
        maxima.append(int(statistics.max))
    return (min(minima), max(maxima)) if minima else (None, None)


def _valid_training_shard(path: Path, identity: dict[str, Any]) -> int | None:
    """Return a valid shard's row count, including legacy shards without sidecars."""
    if not path.exists():
        return None
    _, pq = _require_pyarrow()
    try:
        file = pq.ParquetFile(path)
        rows = int(file.metadata.num_rows)
        if rows < 1 or file.metadata.num_row_groups < 1:
            return None
        sidecar = _shard_sidecar(path)
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata != {**identity, "rows": rows}:
                return None
            return rows

        required = {"configuration_id", "checkpoint_fingerprint", "task", "usage", "allowed_for_training"}
        if not required.issubset(file.schema.names):
            return None
        first = file.read_row_group(0, columns=sorted(required)).to_pandas()
        expected_values = {
            "configuration_id": identity["configuration_id"],
            "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
            "task": identity["task"],
            "usage": TRAINING_USAGE,
        }
        for column, expected in expected_values.items():
            if first[column].empty or not first[column].astype(str).eq(str(expected)).all():
                return None
        if first["allowed_for_training"].empty or not first["allowed_for_training"].map(bool).all():
            return None
        rank_column = "candidate_rank" if identity["task"] == "retrieval" else "hard_negative_rank"
        minimum_rank, maximum_rank = _parquet_rank_extrema(file, rank_column)
        if minimum_rank != 1 or maximum_rank != int(identity["count"]):
            return None
        atomic_json({**identity, "rows": rows}, sidecar)
        return rows
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _commit_shard_metadata(path: Path, identity: dict[str, Any], rows: int) -> None:
    atomic_json({**identity, "rows": int(rows)}, _shard_sidecar(path))


def _normalise_caption(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()


def _coco_train_caption_csv(root: Path) -> Path:
    destination = root / "data/splits/coco_train_all_captions.csv"
    if destination.exists():
        return destination
    annotation = root / "data/coco/annotations/captions_train2017.json"
    if not annotation.exists():
        raise FileNotFoundError(f"COCO training annotations are missing: {annotation}")
    payload = json.loads(annotation.read_text())
    images = {int(value["id"]): value["file_name"] for value in payload["images"]}
    rows = []
    for value in payload["annotations"]:
        image_id = int(value["image_id"])
        if image_id not in images:
            continue
        relative = f"data/coco/train2017/{images[image_id]}"
        if not (root / relative).is_file():
            continue
        rows.append({"image_path": relative, "caption": str(value["caption"]), "annotation_id": int(value["id"])})
    atomic_csv(pd.DataFrame(rows), destination)
    return destination


@torch.inference_mode()
def _retrieval_negative_chunks(
    payload: dict[str, Any],
    captions: list[str],
    *,
    count: int,
    chunk_size: int,
    device: str,
    metadata: dict[str, Any],
) -> Iterator[pd.DataFrame]:
    images = payload["image_embeddings"].float().to(device)
    texts = payload["text_embeddings"].float().to(device)
    image_ids = [str(value) for value in payload["image_ids"]]
    owners = [str(value) for value in payload["text_image_ids"]]
    image_index = {value: index for index, value in enumerate(image_ids)}
    owner_to_text: dict[str, list[int]] = {}
    caption_to_text: dict[str, list[int]] = {}
    caption_to_owners: dict[str, set[str]] = {}
    for index, (owner, caption) in enumerate(zip(owners, captions)):
        normalised = _normalise_caption(caption)
        owner_to_text.setdefault(owner, []).append(index)
        caption_to_text.setdefault(normalised, []).append(index)
        caption_to_owners.setdefault(normalised, set()).add(owner)
    for start in range(0, len(image_ids), chunk_size):
        stop = min(start + chunk_size, len(image_ids))
        similarity = images[start:stop] @ texts.t()
        for local, query_index in enumerate(range(start, stop)):
            owner = image_ids[query_index]
            invalid = set(owner_to_text.get(owner, []))
            for index in owner_to_text.get(owner, []):
                invalid.update(caption_to_text[_normalise_caption(captions[index])])
            if invalid:
                similarity[local, torch.tensor(sorted(invalid), device=device)] = -torch.inf
        scores, indices = similarity.topk(min(count, similarity.size(1)), dim=1)
        rows = []
        for local, query_index in enumerate(range(start, stop)):
            positive_indices = owner_to_text.get(image_ids[query_index], [])
            if not positive_indices:
                raise RuntimeError(f"COCO image {image_ids[query_index]} has no positive caption")
            positive_caption = captions[positive_indices[0]]
            for rank, negative_index in enumerate(indices[local].tolist(), 1):
                rows.append({**metadata, "direction": "i2t", "query_id": image_ids[query_index], "positive_id": image_ids[query_index], "positive_text": positive_caption, "negative_id": f"caption:{negative_index:06d}", "negative_owner_id": owners[negative_index], "negative_text": captions[negative_index], "dual_encoder_score": float(scores[local, rank - 1].cpu()), "candidate_rank": rank, "valid_caption_exclusion_applied": True, "duplicate_caption_exclusion_applied": True})
        yield pd.DataFrame(rows)
    for start in range(0, len(captions), chunk_size):
        stop = min(start + chunk_size, len(captions))
        similarity = texts[start:stop] @ images.t()
        for local, query_index in enumerate(range(start, stop)):
            invalid_owners = set(caption_to_owners[_normalise_caption(captions[query_index])]) | {owners[query_index]}
            invalid = [image_index[value] for value in invalid_owners if value in image_index]
            similarity[local, torch.tensor(invalid, device=device)] = -torch.inf
        scores, indices = similarity.topk(min(count, similarity.size(1)), dim=1)
        rows = []
        for local, query_index in enumerate(range(start, stop)):
            for rank, negative_index in enumerate(indices[local].tolist(), 1):
                rows.append({**metadata, "direction": "t2i", "query_id": f"caption:{query_index:06d}", "positive_id": owners[query_index], "positive_text": captions[query_index], "negative_id": image_ids[negative_index], "negative_owner_id": image_ids[negative_index], "negative_text": "", "dual_encoder_score": float(scores[local, rank - 1].cpu()), "candidate_rank": rank, "valid_caption_exclusion_applied": True, "duplicate_caption_exclusion_applied": True})
        yield pd.DataFrame(rows)


def _stratified_train_indices(targets: list[int], train_fraction: float, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    values = np.asarray(targets)
    selected: list[int] = []
    for label in sorted(set(targets)):
        indices = np.flatnonzero(values == label)
        indices = indices[rng.permutation(len(indices))]
        selected.extend(int(value) for value in indices[: int(round(len(indices) * train_fraction))])
    return sorted(selected)


def _classification_training_dataset(root: Path, task: str, image_size: int, mode: str):
    from torchvision.datasets import CIFAR100, EuroSAT, OxfordIIITPet

    transform = build_image_transform(image_size=image_size, train=False)
    if task == "cifar100_zeroshot":
        base = CIFAR100(root=root / "data/multitask/cifar100", train=True, transform=transform, download=False)
        indices = _stratified_train_indices(list(base.targets), 0.80, 42)
        dataset_id, split = "cifar100_train", "cifar100_official_train_train_partition"
    elif task == "pets_zeroshot":
        base = OxfordIIITPet(root=root / "data/multitask/oxford-iiit-pet", split="trainval", target_types="category", transform=transform, download=False)
        labels = [int(value) for value in getattr(base, "_labels")]
        indices = _stratified_train_indices(labels, 0.80, 42)
        dataset_id, split = "pets_trainval", "oxford_iiit_pet_trainval_train_partition"
    elif task == "eurosat_zeroshot":
        base = EuroSAT(root=root / "data/multitask/eurosat", transform=transform, download=False)
        manifest = pd.read_csv(create_eurosat_split_manifest(root))
        allowed = set(manifest.loc[manifest["split"].eq("train"), "sample_id"].astype(str))
        wrapped = IndexedClassificationDataset(base, "eurosat_train", root)
        indices = [index for index in range(len(wrapped)) if wrapped._sample_id(index) in allowed]
        dataset_id, split = "eurosat_train", "eurosat_deterministic_stratified_train"
    else:
        raise KeyError(task)
    wrapped = IndexedClassificationDataset(base, dataset_id, root)
    if mode == "smoke":
        indices = indices[: min(64, len(indices))]
    return Subset(wrapped, indices), [clean_class_name(str(value)) for value in base.classes], split


def _selected_pair_ids(root: Path) -> list[str]:
    path = root / "results/phase15/expert_selection/selected_experts.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Run Step 17 selection before training-negative mining: {path}")
    payload = yaml.safe_load(path.read_text())
    selected = [str(value["pair_id"]) for value in payload.get("phase2_pair_shortlist", []) if value.get("intended_role") == "primary"]
    if not selected:
        raise ValueError("selected_experts.yaml contains no primary pair paths")
    return selected


def selected_training_pair_ids(project_root: str | Path, mode: str = "full") -> list[str]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke/full")
    selected = _selected_pair_ids(Path(project_root).resolve())
    return selected[:1] if mode == "smoke" else selected


def _resolve_training_specs(root: Path, config_ids: list[str]):
    variants = tuple(sorted({value.rsplit("__", 1)[-1] for value in config_ids}))
    specs = resolve_best_checkpoints(
        root / "checkpoints",
        list(VISION_ENCODERS),
        variants,
        config_ids=set(config_ids),
        load_epochs=False,
    )
    by_id = {value.config_id: value for value in specs}
    missing = set(config_ids) - set(by_id)
    if missing:
        raise FileNotFoundError(f"Selected training-negative checkpoints are missing: {sorted(missing)}")
    return [by_id[value] for value in config_ids]


def prepare_training_hard_negative_sources(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    coco_csv = _coco_train_caption_csv(root)
    eurosat_manifest = create_eurosat_split_manifest(root)
    return {
        "coco_captions": str(coco_csv),
        "coco_fingerprint": stat_fingerprint(coco_csv),
        "classification_fingerprint": _classification_source_fingerprint(root),
        "eurosat_manifest": str(eurosat_manifest),
    }


def _training_paths(root: Path, configuration_id: str, mode: str) -> tuple[Path, Path, Path]:
    output = root / "results/phase15/training_hard_negatives"
    retrieval = output / "shards" / f"{configuration_id}__retrieval__{mode}.parquet"
    classification = output / "shards" / f"{configuration_id}__classification__{mode}.parquet"
    cache = output / "cache" / f"{configuration_id}__coco_train__{mode}.pt"
    return retrieval, classification, cache


def _training_identities(
    spec: Any,
    mode: str,
    count: int,
    sources: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_fingerprint = stat_fingerprint(spec.checkpoint)
    retrieval = _shard_identity(
        configuration_id=spec.config_id,
        checkpoint_fingerprint=checkpoint_fingerprint,
        task="retrieval",
        mode=mode,
        count=count,
        source_fingerprint=sources["coco_fingerprint"],
    )
    classification = _shard_identity(
        configuration_id=spec.config_id,
        checkpoint_fingerprint=checkpoint_fingerprint,
        task="classification",
        mode=mode,
        count=1,
        source_fingerprint=sources["classification_fingerprint"],
    )
    return retrieval, classification


def training_configuration_status(
    project_root: str | Path,
    configuration_id: str,
    *,
    mode: str = "full",
    count: int = TRAINING_HARD_NEGATIVES_PER_POSITIVE,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    prepared = sources or prepare_training_hard_negative_sources(root)
    spec = _resolve_training_specs(root, [configuration_id])[0]
    retrieval_identity, classification_identity = _training_identities(spec, mode, count, prepared)
    retrieval_path, classification_path, cache_path = _training_paths(root, configuration_id, mode)
    retrieval_rows = _valid_training_shard(retrieval_path, retrieval_identity)
    classification_rows = _valid_training_shard(classification_path, classification_identity)
    return {
        "configuration_id": configuration_id,
        "retrieval_complete": retrieval_rows is not None,
        "classification_complete": classification_rows is not None,
        "complete": retrieval_rows is not None and classification_rows is not None,
        "retrieval_rows": retrieval_rows,
        "classification_rows": classification_rows,
        "retrieval_shard": str(retrieval_path),
        "classification_shard": str(classification_path),
        "cache": str(cache_path),
    }


def _mine_retrieval_for_configuration(
    root: Path,
    spec: Any,
    model: Any,
    config: dict[str, Any],
    *,
    mode: str,
    count: int,
    chunk_size: int,
    device: str,
    sources: dict[str, str],
    identity: dict[str, Any],
    destination: Path,
    cache_path: Path,
) -> int:
    coco_csv = Path(sources["coco_captions"])
    max_images = 32 if mode == "smoke" else None
    cache_metadata = {
        "configuration_id": spec.config_id,
        "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
        "source_split": "coco_train2017",
        "source_fingerprint": sources["coco_fingerprint"],
        "mode": mode,
    }
    retrieval = load_cache(cache_path, expected_metadata=cache_metadata)
    if retrieval is None:
        # Accept the cache format written before source fingerprints were added.
        legacy_metadata = {key: value for key, value in cache_metadata.items() if key != "source_fingerprint"}
        retrieval = load_cache(cache_path, expected_metadata=legacy_metadata)
    if retrieval is None:
        retrieval = _encode_coco(model, config, coco_csv, root, device, max_images)
        save_cache(cache_path, retrieval, cache_metadata)
    _, captions, caption_owners = _read_coco(coco_csv, root, max_images)
    if [str(value) for value in retrieval["text_image_ids"]] != [str(value) for value in caption_owners]:
        raise RuntimeError("COCO train caption ordering does not match its embedding cache")
    metadata = {
        "usage": TRAINING_USAGE,
        "allowed_for_training": True,
        "source_split": "coco_train2017",
        "dataset": "coco",
        "task": "retrieval",
        "configuration_id": spec.config_id,
        "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
        "source_checkpoint": str(spec.checkpoint),
    }
    rows = _write_parquet_chunks(
        _retrieval_negative_chunks(
            retrieval,
            captions,
            count=count,
            chunk_size=chunk_size,
            device=device,
            metadata=metadata,
        ),
        destination,
    )
    _commit_shard_metadata(destination, identity, rows)
    del retrieval
    return rows


def _mine_classification_for_configuration(
    root: Path,
    spec: Any,
    model: Any,
    config: dict[str, Any],
    *,
    mode: str,
    device: str,
    identity: dict[str, Any],
    destination: Path,
) -> int:
    image_size = int(config.get("data", {}).get("image_size", 224))
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    classification_parts: list[pd.DataFrame] = []
    for task in CLASSIFICATION_TRAINING_TASKS:
        dataset, class_names, split = _classification_training_dataset(root, task, image_size, mode)
        evaluated = evaluate_zeroshot_classification(
            model,
            dataset,
            class_names,
            device,
            task.replace("_zeroshot", "_train"),
            batch_size=batch_size,
            num_workers=0,
            max_samples=None,
            templates=PROMPT_TEMPLATES,
        )
        negatives = enriched_classification_hard_negatives(evaluated, task, count=1)
        negatives["prompt_template_ids"] = json.dumps(list(range(len(PROMPT_TEMPLATES))))
        negatives = annotate_training_rows(
            negatives,
            source_split=split,
            dataset=task.replace("_zeroshot", ""),
            task="classification",
            configuration_id=spec.config_id,
            checkpoint_fingerprint=identity["checkpoint_fingerprint"],
            source_checkpoint=str(spec.checkpoint),
        )
        validate_no_protected_ids(negatives, protected_test_ids(root))
        classification_parts.append(negatives)
    classification_frame = pd.concat(classification_parts, ignore_index=True)
    rows = _write_parquet_chunks([classification_frame], destination)
    _commit_shard_metadata(destination, identity, rows)
    return rows


def mine_training_hard_negatives_for_config(
    project_root: str | Path,
    configuration_id: str,
    *,
    mode: str = "full",
    count: int = TRAINING_HARD_NEGATIVES_PER_POSITIVE,
    device: str | None = None,
    chunk_size: int = 128,
    resume: bool = True,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"} or count < 1 or chunk_size < 1:
        raise ValueError("mode must be smoke/full and counts must be positive")
    _require_pyarrow()
    root = Path(project_root).resolve()
    target = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == "full" and not str(target).startswith("cuda"):
        raise RuntimeError("Full training-negative mining requires an explicitly allocated GPU")
    sources = prepare_training_hard_negative_sources(root)
    spec = _resolve_training_specs(root, [configuration_id])[0]
    retrieval_identity, classification_identity = _training_identities(spec, mode, count, sources)
    retrieval_path, classification_path, cache_path = _training_paths(root, configuration_id, mode)
    retrieval_rows = _valid_training_shard(retrieval_path, retrieval_identity) if resume else None
    classification_rows = _valid_training_shard(classification_path, classification_identity) if resume else None
    retrieval_status = "resumed" if retrieval_rows is not None else "pending"
    classification_status = "resumed" if classification_rows is not None else "pending"
    if retrieval_rows is not None and classification_rows is not None:
        print(f"Resuming complete training-negative configuration: {configuration_id}", flush=True)
        return {
            "configuration_id": configuration_id,
            "status": "resumed",
            "retrieval_status": retrieval_status,
            "classification_status": classification_status,
            "retrieval_rows": retrieval_rows,
            "classification_rows": classification_rows,
        }

    print(f"Loading {configuration_id} on {target}", flush=True)
    model, config, _ = load_frozen_model(spec, target)
    try:
        if retrieval_rows is None:
            print(f"Mining retrieval negatives for {configuration_id}", flush=True)
            retrieval_rows = _mine_retrieval_for_configuration(
                root,
                spec,
                model,
                config,
                mode=mode,
                count=count,
                chunk_size=chunk_size,
                device=target,
                sources=sources,
                identity=retrieval_identity,
                destination=retrieval_path,
                cache_path=cache_path,
            )
            retrieval_status = "complete"
        else:
            print(f"Reusing retrieval shard for {configuration_id}: {retrieval_path}", flush=True)
        if classification_rows is None:
            print(f"Mining classification negatives for {configuration_id}", flush=True)
            classification_rows = _mine_classification_for_configuration(
                root,
                spec,
                model,
                config,
                mode=mode,
                device=target,
                identity=classification_identity,
                destination=classification_path,
            )
            classification_status = "complete"
        else:
            print(f"Reusing classification shard for {configuration_id}: {classification_path}", flush=True)
    finally:
        del model
        gc.collect()
        if str(target).startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "configuration_id": configuration_id,
        "status": "complete",
        "retrieval_status": retrieval_status,
        "classification_status": classification_status,
        "retrieval_rows": int(retrieval_rows),
        "classification_rows": int(classification_rows),
    }


def finalize_training_hard_negative_mining(
    project_root: str | Path,
    *,
    mode: str = "full",
    count: int = TRAINING_HARD_NEGATIVES_PER_POSITIVE,
    config_ids: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    selected = config_ids or selected_training_pair_ids(root, mode)
    sources = prepare_training_hard_negative_sources(root)
    specs = _resolve_training_specs(root, selected)
    retrieval_shards: list[Path] = []
    classification_shards: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        retrieval_identity, classification_identity = _training_identities(spec, mode, count, sources)
        retrieval_path, classification_path, _ = _training_paths(root, spec.config_id, mode)
        retrieval_rows = _valid_training_shard(retrieval_path, retrieval_identity)
        classification_rows = _valid_training_shard(classification_path, classification_identity)
        if retrieval_rows is None:
            missing.append(f"{spec.config_id}:retrieval")
        if classification_rows is None:
            missing.append(f"{spec.config_id}:classification")
        if retrieval_rows is None or classification_rows is None:
            continue
        retrieval_shards.append(retrieval_path)
        classification_shards.append(classification_path)
        summary_rows.append(
            {
                "configuration_id": spec.config_id,
                "mode": mode,
                "retrieval_rows": retrieval_rows,
                "classification_rows": classification_rows,
                "checkpoint_fingerprint": retrieval_identity["checkpoint_fingerprint"],
                "status": "complete",
            }
        )
    if missing:
        raise RuntimeError(f"Cannot finalize training-negative mining; missing or stale shards: {missing}")

    output = root / "results/phase15/training_hard_negatives"
    output.mkdir(parents=True, exist_ok=True)
    retrieval_output = output / "retrieval_train_hard_negatives.parquet"
    classification_output = output / "classification_train_hard_negatives.parquet"
    compositional = {
        "training_status": "evaluation_only",
        "allowed_for_training": False,
        "datasets": ["winoground", "sugarcrepe"],
        "reason": "no separate compositional training set is configured",
    }
    with tempfile.TemporaryDirectory(dir=output, prefix=".finalize-") as temporary:
        staging = Path(temporary)
        staged_retrieval = staging / retrieval_output.name
        staged_classification = staging / classification_output.name
        print(f"Combining {len(retrieval_shards)} retrieval shards into staging", flush=True)
        retrieval_rows = _combine_parquet(retrieval_shards, staged_retrieval)
        print(f"Combining {len(classification_shards)} classification shards into staging", flush=True)
        classification_rows = _combine_parquet(classification_shards, staged_classification)
        _, pq = _require_pyarrow()
        classification_ids = pq.read_table(staged_classification, columns=["sample_id"]).to_pandas()
        validate_no_protected_ids(classification_ids, protected_test_ids(root))
        schema = {
            "schema_version": 1,
            "status": "complete",
            "required_row_metadata": [
                "usage",
                "allowed_for_training",
                "source_split",
                "dataset",
                "task",
                "configuration_id",
                "checkpoint_fingerprint",
            ],
            "usage": TRAINING_USAGE,
            "allowed_for_training": True,
            "protected_test_overlap": 0,
            "mode": mode,
            "retrieval_negatives_per_positive": count,
            "configurations": selected,
            "retrieval_rows": retrieval_rows,
            "classification_rows": classification_rows,
        }
        staged_compositional = atomic_json(compositional, staging / "compositional_training_manifest.json")
        staged_summary = atomic_csv(pd.DataFrame(summary_rows), staging / "mining_summary.csv")

        # Invalidate the commit marker before publishing any member of the set.
        # An interruption between replacements therefore cannot pass readiness.
        schema_output = output / "schema.json"
        schema_output.unlink(missing_ok=True)
        os.replace(staged_retrieval, retrieval_output)
        os.replace(staged_classification, classification_output)
        os.replace(staged_compositional, output / staged_compositional.name)
        os.replace(staged_summary, output / staged_summary.name)
        schema["output_sha256"] = {
            "retrieval": sha256_file(retrieval_output),
            "classification": sha256_file(classification_output),
        }
        atomic_json(schema, schema_output)
    return {
        "status": "complete",
        "mode": mode,
        "configurations": selected,
        "retrieval_rows": retrieval_rows,
        "classification_rows": classification_rows,
        "protected_test_overlap": 0,
        "outputs": {
            "retrieval": str(retrieval_output),
            "classification": str(classification_output),
            "compositional": str(output / "compositional_training_manifest.json"),
            "summary": str(output / "mining_summary.csv"),
            "schema": str(output / "schema.json"),
        },
    }


def run_training_hard_negative_mining(
    project_root: str | Path,
    *,
    mode: str = "full",
    count: int = TRAINING_HARD_NEGATIVES_PER_POSITIVE,
    device: str | None = None,
    chunk_size: int = 128,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_ids = selected_training_pair_ids(root, mode)
    worker_results = [
        mine_training_hard_negatives_for_config(
            root,
            configuration_id,
            mode=mode,
            count=count,
            device=device,
            chunk_size=chunk_size,
            resume=resume,
        )
        for configuration_id in config_ids
    ]
    result = finalize_training_hard_negative_mining(root, mode=mode, count=count, config_ids=config_ids)
    result["worker_results"] = worker_results
    return result
