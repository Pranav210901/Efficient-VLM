from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_image_transform

from .checkpoint_selection import VARIANTS, CheckpointSpec, load_frozen_model, resolve_best_checkpoints
from .classification_evaluator import (
    classification_metrics,
    classification_predictions,
    evaluate_zeroshot_classification,
    per_class_accuracy,
    prepare_zeroshot_dataset,
)
from .config import DEFAULT_DATASETS, PROMPT_TEMPLATES, VISION_ENCODERS, output_root
from .datasets import ZERO_SHOT_DATASET_PATHS, dataset_split_id, validate_datasets
from .embedding_cache import load_cache, save_cache
from .retrieval_evaluator import evaluate_retrieval, retrieval_predictions, text_to_image_predictions


class _Images(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.transform = build_image_transform(image_size=image_size, train=False)

    def __len__(self) -> int: return len(self.paths)
    def __getitem__(self, index: int): return self.transform(Image.open(self.paths[index]).convert("RGB")), str(self.paths[index])


def _read_coco(path: Path, project_root: Path, max_images: int | None = None) -> tuple[list[Path], list[str], list[str]]:
    grouped: dict[str, list[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["image_path"]), []).append(str(row["caption"]))
    selected = list(grouped.items())[:max_images]
    image_paths = [Path(key) if Path(key).is_absolute() else project_root / key for key, _ in selected]
    captions = [caption for _, values in selected for caption in values]
    caption_ids = [str(path) for path, (_, values) in zip(image_paths, selected) for _ in values]
    return image_paths, captions, caption_ids


@torch.no_grad()
def _encode_coco(model, config: dict[str, Any], csv_path: Path, project_root: Path, device: str, max_images: int | None) -> dict[str, Any]:
    image_paths, captions, text_image_ids = _read_coco(csv_path, project_root, max_images)
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    image_size = int(config.get("data", {}).get("image_size", 224))
    loader = DataLoader(_Images(image_paths, image_size), batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Encoding {len(image_paths)} unique COCO images in {len(loader)} batches", flush=True)
    image_parts = []
    image_interval = max(1, len(loader) // 10)
    for index, (images, _) in enumerate(loader, 1):
        image_parts.append(model.encode_image(images.to(device)).cpu())
        if index == 1 or index == len(loader) or index % image_interval == 0:
            print(f"Image encoding: {index}/{len(loader)} batches", flush=True)
    text_batches = (len(captions) + batch_size - 1) // batch_size
    print(f"Encoding {len(captions)} COCO captions in {text_batches} batches", flush=True)
    text_parts = []
    text_interval = max(1, text_batches // 10)
    for batch_index, start in enumerate(range(0, len(captions), batch_size), 1):
        text_parts.append(model.encode_text(captions[start : start + batch_size]).cpu())
        if batch_index == 1 or batch_index == text_batches or batch_index % text_interval == 0:
            print(f"Text encoding: {batch_index}/{text_batches} batches", flush=True)
    return {"image_embeddings": torch.cat(image_parts), "text_embeddings": torch.cat(text_parts), "image_ids": [str(path) for path in image_paths], "text_image_ids": text_image_ids}


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_tables(root: Path, long_rows: list[dict[str, Any]], status_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(long_rows)
    status = pd.DataFrame(status_rows)
    manifest = pd.DataFrame(manifest_rows)
    paths = {"results_long": root / "task_results_long.csv", "status": root / "task_status.csv", "manifest": root / "evaluation_manifest.csv", "results_wide": root / "task_results_wide.csv", "summary": root / "configuration_summary.csv"}
    _atomic_to_csv(long, paths["results_long"])
    _atomic_to_csv(status, paths["status"])
    _atomic_to_csv(manifest, paths["manifest"])
    if not long.empty:
        wide = long.pivot_table(index=["config_id", "vision_encoder", "text_encoder", "variant"], columns=["task", "metric"], values="value", aggfunc="first").reset_index()
        wide.columns = ["__".join(str(v) for v in value if str(v)) if isinstance(value, tuple) else str(value) for value in wide.columns]
        _atomic_to_csv(wide, paths["results_wide"])
        summary = long.groupby(["config_id", "vision_encoder", "text_encoder", "variant", "task"], as_index=False).size()
        _atomic_to_csv(summary, paths["summary"])
    else:
        _atomic_to_csv(pd.DataFrame(), paths["results_wide"])
        _atomic_to_csv(pd.DataFrame(), paths["summary"])
    return paths


def run_multitask_evaluation(
    project_root: str | Path,
    datasets: list[str] | tuple[str, ...] = DEFAULT_DATASETS,
    config_scope: str = "baseline_all_pairs",
    mode: str = "smoke",
    device: str | None = None,
    reliable_blf_variants: tuple[str, ...] = (),
    config_ids: tuple[str, ...] | list[str] | None = None,
    output_root_override: str | Path | None = None,
    split_role: str = "historical",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if split_role not in {"historical", "development", "final_test"}:
        raise ValueError("split_role must be historical, development, or final_test")
    default_output = output_root(root) if split_role == "historical" else root / f"results/phase1_multitask_{split_role}"
    out = Path(output_root_override).resolve() if output_root_override else default_output
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    if config_scope not in {"baseline_all_pairs", "baseline_plus_reliable_blf"}:
        raise ValueError("config_scope must be baseline_all_pairs or baseline_plus_reliable_blf")
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    requested = set(config_ids) if config_ids is not None else None
    if requested:
        # Explicit IDs are the authoritative allow-list. Deriving their
        # variants here lets the multi-GPU coordinator request a small BLF
        # subset without opening the entire BLF matrix.
        requested_variants = {value.rsplit("__", 1)[-1] for value in requested}
        unknown_variants = requested_variants - set(VARIANTS)
        if unknown_variants:
            raise ValueError(f"Unknown variants in configuration IDs: {sorted(unknown_variants)}")
        variants = tuple(value for value in VARIANTS if value in requested_variants)
    else:
        variants = ("baseline",) + (reliable_blf_variants if config_scope == "baseline_plus_reliable_blf" else ())
    specs = resolve_best_checkpoints(root / "checkpoints", list(VISION_ENCODERS), variants, config_ids=requested, load_epochs=False)
    if requested is not None:
        missing = requested - {spec.config_id for spec in specs}
        if missing: raise KeyError(f"Unknown or unavailable configuration IDs: {sorted(missing)}")
    if not specs: raise FileNotFoundError("No canonical best checkpoints were found")
    statuses = validate_datasets(root, datasets, split_role=split_role)
    if split_role == "final_test":
        for status in statuses:
            if status["task"] == "coco_retrieval":
                status["status"] = "missing"
                status["detail"] = "external final retrieval benchmark is not configured"
    available = {row["task"] for row in statuses if row["status"] == "ready"}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_images = 8 if mode == "smoke" else None
    max_classification_samples = 32 if mode == "smoke" else None
    long_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    def append_metrics(spec: CheckpointSpec, task: str, metrics: dict[str, float], best_epoch: int) -> None:
        for metric, value in metrics.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite {task} metric {metric} for {spec.config_id}")
            long_rows.append(
                {
                    "config_id": spec.config_id,
                    "vision_encoder": spec.vision_encoder,
                    "text_encoder": spec.text_encoder,
                    "variant": spec.variant,
                    "mode": mode,
                    "task": task,
                    "metric": metric,
                    "value": float(value),
                    "best_epoch": best_epoch,
                    "checkpoint": str(spec.checkpoint),
                    "split_role": split_role,
                    "dataset_split": dataset_split_id(task, split_role),
                }
            )

    for spec in specs[:1] if mode == "smoke" else specs:
        print(f"Starting evaluation: {spec.config_id} ({spec.checkpoint})", flush=True)
        best_epoch = spec.epoch
        model = None
        config: dict[str, Any] | None = None

        def ensure_model():
            nonlocal model, config, best_epoch
            if model is None or config is None:
                print(f"Loading frozen model on {device}: {spec.config_id}", flush=True)
                model, config, best_epoch = load_frozen_model(spec, str(device))
                print(f"Loaded best checkpoint from epoch {best_epoch}", flush=True)
            return model, config

        try:
            for task in datasets:
                if task not in available:
                    manifest.append(
                        {
                            "config_id": spec.config_id,
                            "task": task,
                            "mode": mode,
                            "status": "unavailable",
                            "detail": "dataset validation did not report ready",
                        }
                    )
                    continue

                if task == "coco_retrieval":
                    csv_path = root / "data/val_all_captions.csv"
                    split_suffix = "" if split_role == "historical" else f"__{split_role}"
                    cache_path = out / "cache" / f"{spec.config_id}__coco{split_suffix}__{mode}.pt"
                    metadata = {
                        "evaluator_version": 2,
                        "checkpoint": str(spec.checkpoint),
                        "checkpoint_mtime_ns": spec.checkpoint.stat().st_mtime_ns,
                        "mode": mode,
                        "split_role": split_role,
                        "dataset_split": dataset_split_id(task, split_role),
                        "max_images": max_images,
                        "dataset": str(csv_path),
                        "dataset_mtime_ns": csv_path.stat().st_mtime_ns,
                    }
                    cached = load_cache(cache_path, metadata)
                    if cached is None:
                        loaded_model, loaded_config = ensure_model()
                        cached = _encode_coco(loaded_model, loaded_config, csv_path, root, str(device), max_images)
                        cached["best_epoch"] = best_epoch
                        save_cache(cache_path, cached, metadata)
                        print(f"Saved embedding cache: {cache_path}", flush=True)
                    else:
                        best_epoch = int(cached.get("best_epoch", best_epoch))
                        print(f"Reusing embedding cache: {cache_path}", flush=True)

                    metrics = evaluate_retrieval(
                        cached["image_embeddings"],
                        cached["text_embeddings"],
                        cached["image_ids"],
                        cached["text_image_ids"],
                    )
                    append_metrics(spec, task, metrics, best_epoch)
                    predictions = retrieval_predictions(
                        cached["image_embeddings"],
                        cached["text_embeddings"],
                        cached["image_ids"],
                        cached["text_image_ids"],
                    )
                    prediction_path = out / "predictions" / f"{spec.config_id}__coco_i2t{split_suffix}__{mode}.jsonl"
                    _atomic_write_text(prediction_path, "".join(json.dumps(row) + "\n" for row in predictions))
                    t2i_predictions = text_to_image_predictions(
                        cached["image_embeddings"],
                        cached["text_embeddings"],
                        cached["image_ids"],
                        cached["text_image_ids"],
                        topk=10,
                    )
                    t2i_prediction_path = out / "predictions" / f"{spec.config_id}__coco_t2i{split_suffix}__{mode}.jsonl"
                    _atomic_write_text(t2i_prediction_path, "".join(json.dumps(row) + "\n" for row in t2i_predictions))
                    manifest.append(
                        {
                            "config_id": spec.config_id,
                            "task": task,
                            "mode": mode,
                            "status": "complete",
                            "checkpoint": str(spec.checkpoint),
                            "cache": str(cache_path),
                            "predictions": str(prediction_path),
                            "predictions_t2i": str(t2i_prediction_path),
                            "samples": len(cached["image_ids"]),
                            "caption_queries": len(cached["text_image_ids"]),
                            "split_role": split_role,
                            "dataset_split": dataset_split_id(task, split_role),
                        }
                    )
                    print(
                        f"Completed {spec.config_id}/{task}: "
                        f"coco5_i2t_R@1={metrics.get('coco5_i2t_R@1', float('nan')):.4f}",
                        flush=True,
                    )

                elif task in ZERO_SHOT_DATASET_PATHS:
                    status = next(value for value in statuses if value["task"] == task)
                    split_suffix = "" if split_role == "historical" else f"__{split_role}"
                    cache_path = out / "cache" / f"{spec.config_id}__{task}{split_suffix}__{mode}.pt"
                    metadata = {
                        "evaluator_version": 2,
                        "checkpoint": str(spec.checkpoint),
                        "checkpoint_mtime_ns": spec.checkpoint.stat().st_mtime_ns,
                        "task": task,
                        "mode": mode,
                        "split_role": split_role,
                        "dataset_split": dataset_split_id(task, split_role),
                        "max_samples": max_classification_samples,
                        "dataset_path": status["path"],
                        "dataset_samples": status["samples"],
                        "dataset_classes": status["classes_or_categories"],
                        "prompt_templates": PROMPT_TEMPLATES,
                    }
                    cached = load_cache(cache_path, metadata)
                    if cached is None:
                        loaded_model, loaded_config = ensure_model()
                        image_size = int(loaded_config.get("data", {}).get("image_size", 224))
                        batch_size = int(loaded_config.get("training", {}).get("batch_size", 64))
                        num_workers = min(4, int(loaded_config.get("data", {}).get("num_workers", 0)))
                        dataset, class_names = prepare_zeroshot_dataset(root, task, image_size, split_role=split_role)
                        cached = evaluate_zeroshot_classification(
                            loaded_model,
                            dataset,
                            class_names,
                            device=str(device),
                            task=task,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            max_samples=max_classification_samples,
                            templates=PROMPT_TEMPLATES,
                        )
                        cached["best_epoch"] = best_epoch
                        save_cache(cache_path, cached, metadata)
                        print(f"Saved zero-shot cache: {cache_path}", flush=True)
                    else:
                        best_epoch = int(cached.get("best_epoch", best_epoch))
                        print(f"Reusing zero-shot cache: {cache_path}", flush=True)

                    logits = cached["logits"]
                    targets = cached["targets"]
                    class_names = list(cached["class_names"])
                    metrics = classification_metrics(logits, targets, len(class_names))
                    append_metrics(spec, task, metrics, best_epoch)
                    sample_rows = classification_predictions(
                        logits,
                        targets,
                        list(cached["sample_ids"]),
                        class_names,
                        task,
                    )
                    prediction_path = out / "predictions" / f"{spec.config_id}__{task}{split_suffix}__{mode}.jsonl"
                    _atomic_write_text(prediction_path, "".join(json.dumps(row) + "\n" for row in sample_rows))
                    per_class_path = out / "predictions" / f"{spec.config_id}__{task}{split_suffix}__{mode}__per_class.csv"
                    _atomic_to_csv(pd.DataFrame(per_class_accuracy(logits, targets, class_names)), per_class_path)
                    manifest.append(
                        {
                            "config_id": spec.config_id,
                            "task": task,
                            "mode": mode,
                            "status": "complete",
                            "checkpoint": str(spec.checkpoint),
                            "cache": str(cache_path),
                            "predictions": str(prediction_path),
                            "per_class": str(per_class_path),
                            "samples": int(targets.numel()),
                            "classes": len(class_names),
                            "split_role": split_role,
                            "dataset_split": dataset_split_id(task, split_role),
                        }
                    )
                    print(
                        f"Completed {spec.config_id}/{task}: top1_accuracy={metrics['top1_accuracy']:.4f}",
                        flush=True,
                    )

                else:
                    manifest.append(
                        {
                            "config_id": spec.config_id,
                            "task": task,
                            "mode": mode,
                            "status": "not_run",
                            "detail": "no evaluator is enabled for this task",
                        }
                    )

                # Preserve every completed task if a later dataset fails.
                _write_tables(out, long_rows, statuses, manifest)
        finally:
            if model is not None:
                del model
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
    paths = _write_tables(out, long_rows, statuses, manifest)
    return {"mode": mode, "device": device, "checkpoints": len(specs), "datasets": statuses, "results": pd.DataFrame(long_rows), "manifest": pd.DataFrame(manifest), "paths": paths}


def run_multitask_smoke_test(project_root: str | Path, datasets: list[str] | tuple[str, ...] = DEFAULT_DATASETS, device: str | None = None) -> dict[str, Any]:
    return run_multitask_evaluation(project_root, datasets, mode="smoke", device=device)
