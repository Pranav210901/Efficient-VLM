from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.multitask.config import PROMPT_TEMPLATES
from src.multitask.embedding_cache import load_cache
from src.multitask.runner import _read_coco


HEADLINE_METRICS = {
    "coco_retrieval": ("coco5_i2t_R@1", "coco5_t2i_R@1"),
    "cifar100_zeroshot": ("top1_accuracy",),
    "pets_zeroshot": ("top1_accuracy",),
    "eurosat_zeroshot": ("top1_accuracy",),
}

SOURCE_SPLITS = {
    "coco_retrieval": "coco_val2017",
    "cifar100_zeroshot": "cifar100_test",
    "pets_zeroshot": "oxford_iiit_pet_test",
    "eurosat_zeroshot": "eurosat_complete_dataset",
}

USAGE_RESTRICTION = "diagnostic_only_do_not_use_for_training"
USAGE = "diagnostic_only"


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        compression = "gzip" if path.suffix == ".gz" else None
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _combine_gzip_csvs(sources: list[Path], destination: Path) -> None:
    if not sources:
        raise ValueError("At least one CSV shard is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        expected_header: str | None = None
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
            for source in sources:
                with gzip.open(source, "rt", encoding="utf-8", newline="") as input_file:
                    header = input_file.readline()
                    if expected_header is None:
                        expected_header = header
                        output.write(header)
                    elif header != expected_header:
                        raise ValueError(f"CSV shard columns do not match: {source}")
                    shutil.copyfileobj(input_file, output)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mine_retrieval_hard_negatives(
    similarity: torch.Tensor,
    query_ids: list[str],
    candidate_owner_ids: list[str],
    count: int = 8,
) -> list[dict[str, object]]:
    if similarity.shape != (len(query_ids), len(candidate_owner_ids)):
        raise ValueError("ID counts must match similarity shape")
    if count < 1:
        raise ValueError("count must be positive")
    rows = []
    for query_index, query_id in enumerate(query_ids):
        order = similarity[query_index].argsort(descending=True)
        kept = [
            (int(index), float(similarity[query_index, index]))
            for index in order
            if candidate_owner_ids[int(index)] != query_id
        ][:count]
        for rank, (index, score) in enumerate(kept, 1):
            rows.append(
                {
                    "query_id": query_id,
                    "negative_id": str(index),
                    "negative_owner_id": candidate_owner_ids[index],
                    "hard_negative_rank": rank,
                    "score": score,
                }
            )
    return rows


def mine_classification_hard_negatives(logits: torch.Tensor, targets: torch.Tensor) -> list[dict[str, object]]:
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.numel():
        raise ValueError("logits must be [samples, classes] and targets must be [samples]")
    masked = logits.clone(); masked[torch.arange(len(targets)), targets] = -torch.inf
    scores, labels = masked.max(dim=1)
    return [
        {
            "sample_index": index,
            "true_class": int(targets[index]),
            "negative_class": int(labels[index]),
            "score": float(scores[index]),
        }
        for index in range(len(targets))
    ]


def select_reliable_configs(results: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Select strong multi-task configurations without looking at test errors."""

    if limit < 1:
        raise ValueError("limit must be positive")
    required = {"config_id", "task", "metric", "value"}
    if not required.issubset(results.columns):
        raise ValueError(f"Results are missing columns: {sorted(required - set(results.columns))}")
    selected = results[
        results.apply(lambda row: row["metric"] in HEADLINE_METRICS.get(row["task"], ()), axis=1)
    ].copy()
    if selected.empty:
        raise ValueError("No Phase 1 headline metrics are available for hard-negative model selection")
    selected["metric_key"] = selected["task"].astype(str) + "__" + selected["metric"].astype(str)
    selected["metric_percentile"] = selected.groupby("metric_key")["value"].rank(method="average", pct=True)
    scores = selected.groupby("config_id", as_index=False).agg(
        multitask_performance_score=("metric_percentile", "mean"),
        headline_metrics=("metric_key", "nunique"),
    )
    return scores.sort_values(
        ["multitask_performance_score", "headline_metrics", "config_id"],
        ascending=[False, False, True],
        kind="stable",
    ).head(limit).reset_index(drop=True)


def _owner_indices(values: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        result.setdefault(str(value), []).append(index)
    return result


@torch.inference_mode()
def mine_bidirectional_retrieval(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_ids: list[str],
    text_image_ids: list[str],
    captions: list[str],
    count: int = 8,
    chunk_size: int = 128,
    device: str = "cpu",
) -> pd.DataFrame:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("retrieval embeddings must be two-dimensional")
    if image_embeddings.size(1) != text_embeddings.size(1):
        raise ValueError("image and text embedding dimensions must match")
    if len(image_ids) != image_embeddings.size(0) or len(text_image_ids) != text_embeddings.size(0):
        raise ValueError("retrieval IDs do not match embedding rows")
    if len(captions) != text_embeddings.size(0):
        raise ValueError("captions do not match text embedding rows")
    if count < 1 or chunk_size < 1:
        raise ValueError("count and chunk_size must be positive")
    target = torch.device(device)
    images = image_embeddings.float().to(target)
    texts = text_embeddings.float().to(target)
    text_owners = [str(value) for value in text_image_ids]
    image_names = [str(value) for value in image_ids]
    text_by_owner = _owner_indices(text_owners)
    image_by_id = {value: index for index, value in enumerate(image_names)}
    rows: list[dict[str, object]] = []

    for start in range(0, len(image_names), chunk_size):
        stop = min(start + chunk_size, len(image_names))
        similarities = images[start:stop] @ texts.t()
        for local, query_index in enumerate(range(start, stop)):
            query_id = image_names[query_index]
            positives = text_by_owner.get(query_id, [])
            if positives:
                similarities[local, torch.tensor(positives, device=target)] = -torch.inf
        k = min(count, similarities.size(1))
        scores, indices = similarities.topk(k, dim=1)
        for local, query_index in enumerate(range(start, stop)):
            for rank in range(k):
                negative_index = int(indices[local, rank])
                rows.append(
                    {
                        "direction": "i2t",
                        "query_index": query_index,
                        "query_id": image_names[query_index],
                        "query_text": None,
                        "negative_index": negative_index,
                        "negative_id": str(negative_index),
                        "negative_owner_id": text_owners[negative_index],
                        "negative_text": captions[negative_index],
                        "hard_negative_rank": rank + 1,
                        "score": float(scores[local, rank].cpu()),
                    }
                )

    for start in range(0, len(captions), chunk_size):
        stop = min(start + chunk_size, len(captions))
        similarities = texts[start:stop] @ images.t()
        for local, query_index in enumerate(range(start, stop)):
            owner = text_owners[query_index]
            target_index = image_by_id.get(owner)
            if target_index is not None:
                similarities[local, target_index] = -torch.inf
        k = min(count, similarities.size(1))
        scores, indices = similarities.topk(k, dim=1)
        for local, query_index in enumerate(range(start, stop)):
            for rank in range(k):
                negative_index = int(indices[local, rank])
                rows.append(
                    {
                        "direction": "t2i",
                        "query_index": query_index,
                        "query_id": text_owners[query_index],
                        "query_text": captions[query_index],
                        "negative_index": negative_index,
                        "negative_id": image_names[negative_index],
                        "negative_owner_id": image_names[negative_index],
                        "negative_text": None,
                        "hard_negative_rank": rank + 1,
                        "score": float(scores[local, rank].cpu()),
                    }
                )
    return pd.DataFrame(rows)


def enriched_classification_hard_negatives(
    payload: dict[str, Any],
    task: str,
    count: int = 8,
) -> pd.DataFrame:
    if count < 1:
        raise ValueError("count must be positive")
    logits = payload["logits"].float()
    targets = payload["targets"].long()
    sample_ids = [str(value) for value in payload["sample_ids"]]
    class_names = [str(value) for value in payload["class_names"]]
    if logits.ndim != 2 or logits.size(0) != targets.numel() or len(sample_ids) != targets.numel():
        raise ValueError(f"Invalid classification cache for {task}")
    probabilities = logits.softmax(dim=1)
    masked = probabilities.clone()
    masked[torch.arange(len(targets)), targets] = -torch.inf
    k = min(count, max(1, logits.size(1) - 1))
    scores, labels = masked.topk(k, dim=1)
    true_scores = probabilities.gather(1, targets[:, None]).squeeze(1)
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        target_index = int(targets[sample_index])
        for rank in range(k):
            negative_index = int(labels[sample_index, rank])
            rows.append(
                {
                    "task": task,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "true_class_index": target_index,
                    "true_class": class_names[target_index],
                    "true_class_probability": float(true_scores[sample_index]),
                    "negative_class_index": negative_index,
                    "negative_class": class_names[negative_index],
                    "negative_probability": float(scores[sample_index, rank]),
                    "true_vs_negative_margin": float(true_scores[sample_index] - scores[sample_index, rank]),
                    "hard_negative_rank": rank + 1,
                    "true_prompts": json.dumps([value.format(class_name=class_names[target_index]) for value in PROMPT_TEMPLATES]),
                    "negative_prompts": json.dumps([value.format(class_name=class_names[negative_index]) for value in PROMPT_TEMPLATES]),
                }
            )
    return pd.DataFrame(rows)


def _full_cache_manifest(root: Path) -> pd.DataFrame:
    path = root / "results/phase1_multitask/evaluation_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run Step 11 first: {path}")
    frame = pd.read_csv(path)
    if "mode" not in frame:
        frame["mode"] = frame["cache"].map(lambda value: "full" if str(value).endswith("__full.pt") else "smoke")
    return frame[frame["status"].eq("complete") & frame["mode"].eq("full")].copy()


def _fingerprint(paths: list[Path], settings: dict[str, object]) -> str:
    parts = [json.dumps(settings, sort_keys=True)]
    for path in paths:
        stat = path.stat()
        parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def write_diagnostic_metadata(project_root: str | Path) -> Path:
    """Add a non-trainable sidecar without rewriting historical large files."""
    root = Path(project_root).resolve()
    output = root / "results/phase15/hard_negatives"
    summary_path = output / "hard_negative_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Diagnostic hard-negative summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "usage": USAGE,
            "allowed_for_training": False,
            "source_split": "validation_or_test",
            "historical_files_preserved": True,
        }
    )
    _atomic_json(summary, summary_path)
    destination = output / "diagnostic_metadata.json"
    _atomic_json(
        {
            "usage": USAGE,
            "allowed_for_training": False,
            "source_split": "validation_or_test",
            "retrieval_source_split": SOURCE_SPLITS["coco_retrieval"],
            "classification_source_splits": {task: SOURCE_SPLITS[task] for task in SOURCE_SPLITS if task != "coco_retrieval"},
            "row_count": int(summary.get("retrieval_rows", 0)) + int(summary.get("classification_rows", 0)),
            "historical_files_preserved": True,
        },
        destination,
    )
    return destination


def run_hard_negative_mining(
    project_root: str | Path,
    count: int = 8,
    classification_count: int = 1,
    config_limit: int = 3,
    chunk_size: int = 128,
    device: str | None = None,
    resume: bool = True,
    config_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    if count < 1 or classification_count < 1 or config_limit < 1 or chunk_size < 1:
        raise ValueError("negative counts, config_limit, and chunk_size must be positive")
    output = root / "results/phase15/hard_negatives"
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "results/phase1_multitask/task_results_long.csv")
    ranking = select_reliable_configs(results, config_limit)
    selection_path = output / "selected_configs.csv"
    _atomic_to_csv(ranking, selection_path)
    selected = list(config_ids) if config_ids is not None else ranking["config_id"].tolist()
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA mining was requested but no CUDA device is visible")
    manifest = _full_cache_manifest(root)
    cache_by_task = {
        (str(row.config_id), str(row.task)): Path(str(row.cache))
        for row in manifest.itertuples()
        if pd.notna(row.cache)
    }
    required_tasks = ["coco_retrieval", "cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"]
    progress_path = output / "hard_negative_manifest.csv"
    progress = pd.read_csv(progress_path) if resume and progress_path.exists() else pd.DataFrame()
    progress_rows = progress.to_dict("records") if not progress.empty else []
    captions_path = root / "data/val_all_captions.csv"
    _, captions, caption_owners = _read_coco(captions_path, root)

    for position, config_id in enumerate(selected, 1):
        cache_paths = [cache_by_task.get((config_id, task)) for task in required_tasks]
        missing = [task for task, path in zip(required_tasks, cache_paths) if path is None or not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing full caches for {config_id}: {missing}")
        concrete_paths = [value for value in cache_paths if value is not None]
        settings = {
            "config_id": config_id,
            "count": count,
            "classification_count": classification_count,
            "chunk_size": chunk_size,
            "tasks": required_tasks,
            "version": 2,
        }
        fingerprint = _fingerprint(concrete_paths, settings)
        retrieval_shard = shards / f"{config_id}__retrieval.csv.gz"
        classification_shard = shards / f"{config_id}__classification.csv.gz"
        previous = next((row for row in progress_rows if row.get("config_id") == config_id), None)
        if (
            resume and previous is not None and previous.get("fingerprint") == fingerprint
            and retrieval_shard.exists() and classification_shard.exists()
        ):
            print(f"Hard negatives {position}/{len(selected)} already complete: {config_id}", flush=True)
            continue
        print(f"Hard negatives {position}/{len(selected)}: {config_id} on {target}", flush=True)
        retrieval = load_cache(concrete_paths[0])
        if retrieval is None:
            raise RuntimeError(f"Invalid retrieval cache: {concrete_paths[0]}")
        if [str(value) for value in retrieval["text_image_ids"]] != [str(value) for value in caption_owners]:
            raise RuntimeError("COCO caption ordering does not match the saved retrieval cache")
        retrieval_rows = mine_bidirectional_retrieval(
            retrieval["image_embeddings"], retrieval["text_embeddings"],
            [str(value) for value in retrieval["image_ids"]],
            [str(value) for value in retrieval["text_image_ids"]], captions,
            count=count, chunk_size=chunk_size, device=str(target),
        )
        retrieval_rows.insert(0, "config_id", config_id)
        retrieval_rows.insert(1, "source_split", SOURCE_SPLITS["coco_retrieval"])
        retrieval_rows.insert(2, "usage", USAGE)
        retrieval_rows.insert(3, "allowed_for_training", False)
        retrieval_rows.insert(4, "usage_restriction", USAGE_RESTRICTION)
        _atomic_to_csv(retrieval_rows, retrieval_shard)
        classification_parts = []
        for task, cache_path in zip(required_tasks[1:], concrete_paths[1:]):
            payload = load_cache(cache_path)
            if payload is None:
                raise RuntimeError(f"Invalid classification cache: {cache_path}")
            task_rows = enriched_classification_hard_negatives(payload, task, count=classification_count)
            task_rows.insert(0, "config_id", config_id)
            task_rows.insert(1, "source_split", SOURCE_SPLITS[task])
            task_rows.insert(2, "usage", USAGE)
            task_rows.insert(3, "allowed_for_training", False)
            task_rows.insert(4, "usage_restriction", USAGE_RESTRICTION)
            classification_parts.append(task_rows)
        classification_rows = pd.concat(classification_parts, ignore_index=True)
        _atomic_to_csv(classification_rows, classification_shard)
        progress_rows = [row for row in progress_rows if row.get("config_id") != config_id]
        progress_rows.append(
            {
                "config_id": config_id,
                "fingerprint": fingerprint,
                "retrieval_rows": len(retrieval_rows),
                "classification_rows": len(classification_rows),
                "retrieval_shard": str(retrieval_shard),
                "classification_shard": str(classification_shard),
                "status": "complete",
            }
        )
        _atomic_to_csv(pd.DataFrame(progress_rows).sort_values("config_id"), progress_path)
        del retrieval_rows, classification_rows, classification_parts, retrieval
        if target.type == "cuda":
            torch.cuda.empty_cache()

    completed = pd.DataFrame(progress_rows)
    completed = completed[completed["config_id"].isin(selected) & completed["status"].eq("complete")]
    retrieval_files = [Path(value) for value in completed["retrieval_shard"]]
    classification_files = [Path(value) for value in completed["classification_shard"]]
    retrieval_output = output / "retrieval_hard_negatives.csv.gz"
    classification_output = output / "classification_hard_negatives.csv.gz"
    _combine_gzip_csvs(retrieval_files, retrieval_output)
    _combine_gzip_csvs(classification_files, classification_output)
    summary: dict[str, object] = {
        "selected_config_ids": selected,
        "selection_ranking": ranking.to_dict("records"),
        "retrieval_negatives_per_query": count,
        "classification_negatives_per_sample": classification_count,
        "device": str(target),
        "retrieval_directions": ["i2t", "t2i"],
        "classification_tasks": required_tasks[1:],
        "compositional_tasks": "not enabled in Step 11",
        "usage": USAGE,
        "allowed_for_training": False,
        "source_split": "validation_or_test",
        "usage_restriction": USAGE_RESTRICTION,
        "phase2_training_note": "Mine separate negatives from COCO train or another training split; these held-out benchmark errors must not be used for training.",
        "retrieval_rows": int(completed["retrieval_rows"].sum()),
        "classification_rows": int(completed["classification_rows"].sum()),
        "outputs": {
            "retrieval": str(retrieval_output),
            "classification": str(classification_output),
            "manifest": str(progress_path),
            "selected_configs": str(selection_path),
        },
    }
    _atomic_json(summary, output / "hard_negative_summary.json")
    _atomic_json(
        {
            "usage": USAGE,
            "allowed_for_training": False,
            "source_split": "validation_or_test",
            "retrieval_source_split": SOURCE_SPLITS["coco_retrieval"],
            "classification_source_splits": {task: SOURCE_SPLITS[task] for task in required_tasks[1:]},
            "row_count": summary["retrieval_rows"] + summary["classification_rows"],
        },
        output / "diagnostic_metadata.json",
    )
    return summary
