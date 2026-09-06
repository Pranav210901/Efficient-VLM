from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import _hardware_guard, _student_loader
from src.alignment_v3.fingerprint import hash_config, read_fingerprint, sha256_file
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import _fingerprint, _metrics_from_embeddings
from src.alignment_v3.training import load_training_checkpoint, train
from src.phase15.io_utils import atomic_csv, atomic_json
from src.training.evaluate import extract_embeddings
from src.training.losses import contrastive_loss_with_memory
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DEFAULT = "configs/data_scale_pilot/pipeline.yaml"


def load_pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return load_config(value)


def output_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _select(values: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else int(index)
    if selected < 0 or selected >= len(values):
        raise IndexError(f"index {selected} outside manifest of size {len(values)}")
    return values[selected]


def _group_csv(path: Path) -> OrderedDict[str, list[str]]:
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["image_path"]), []).append(str(row["caption"]))
    if not grouped:
        raise ValueError(f"empty image-caption CSV: {path}")
    return grouped


def _write_grouped(path: Path, grouped: OrderedDict[str, list[str]], selected: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "caption"])
        writer.writeheader()
        for image_path, captions in grouped.items():
            if image_path in selected:
                for caption in captions:
                    writer.writerow({"image_path": image_path, "caption": caption})
    os.replace(temporary, path)


def prepare_subsets(pipeline: dict[str, Any]) -> dict[str, Any]:
    spec = pipeline["split"]
    source = ROOT / str(spec["source_train_csv"])
    grouped = _group_csv(source)
    images = sorted(grouped)
    random.Random(int(spec["subset_seed"])).shuffle(images)
    destination = ROOT / str(spec["subset_root"])
    rows = []
    previous: set[str] = set()
    for percentage in spec["percentages"]:
        count = len(images) if int(percentage) == 100 else int(len(images) * int(percentage) / 100)
        selected = set(images[:count])
        if not previous.issubset(selected):
            raise AssertionError("COCO subsets are not nested")
        path = destination / f"coco_train_{int(percentage):03d}pct.csv"
        _write_grouped(path, grouped, selected)
        captions = sum(len(grouped[value]) for value in selected)
        rows.append(
            {
                "percentage": int(percentage),
                "csv": str(path.relative_to(ROOT)),
                "csv_sha256": sha256_file(path),
                "images": len(selected),
                "captions": captions,
                "image_set_sha256": hashlib.sha256(
                    "\n".join(sorted(selected)).encode()
                ).hexdigest(),
                "contains_previous_subset": previous.issubset(selected),
            }
        )
        previous = selected
    manifest = {
        "status": "COMPLETE",
        "source_csv": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "source_images": len(grouped),
        "source_captions": sum(map(len, grouped.values())),
        "subset_seed": int(spec["subset_seed"]),
        "image_level_sampling": True,
        "nested": True,
        "subsets": rows,
    }
    atomic_json(manifest, output_root(pipeline) / "manifests/coco_subsets.json")
    return manifest


def prepare_flickr_validation(pipeline: dict[str, Any]) -> dict[str, Any]:
    spec = pipeline["split"]
    existing_path = output_root(pipeline) / "manifests/flickr_validation.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text())
        csv_path = ROOT / str(spec["flickr_validation_csv"])
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("images")
            == int(spec["expected_flickr_validation_images"])
            and existing.get("captions")
            == int(spec["expected_flickr_validation_captions"])
            and existing.get("test_overlap_images") == 0
            and existing.get("test_sealed") is True
            and csv_path.is_file()
            and existing.get("csv_sha256") == sha256_file(csv_path)
        ):
            return {**existing, "prepare_action": "REUSED_VERIFIED"}

    try:
        from datasets import Dataset, concatenate_datasets
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Flickr validation provenance must be rebuilt, but this Python "
            "environment lacks Hugging Face datasets. Activate .venv or install "
            "datasets before rebuilding."
        ) from exc

    paths = sorted(glob.glob(str(ROOT / str(spec["flickr_cache_glob"]))))
    if not paths:
        raise FileNotFoundError("cached Flickr30k Arrow shards are unavailable; network fetch is forbidden")
    dataset = concatenate_datasets([Dataset.from_file(path) for path in paths])
    validation = dataset.filter(
        lambda row: str(row["split"]).lower() in {"val", "validation"},
        desc="select Flickr30k Karpathy validation",
    )
    image_dir = ROOT / "data/flickr30k/validation_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / str(spec["flickr_validation_csv"])
    rows = []
    image_hashes = []
    for record in validation:
        image_id = str(record["img_id"])
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            record["image"].convert("RGB").save(image_path)
        relative = str(image_path.relative_to(ROOT))
        image_hashes.append({"path": relative, "sha256": sha256_file(image_path)})
        for caption in record["caption"]:
            rows.append({"image_path": relative, "caption": str(caption)})
    unique = {row["image_path"] for row in rows}
    if len(unique) != int(spec["expected_flickr_validation_images"]):
        raise ValueError(f"Flickr validation image count mismatch: {len(unique)}")
    if len(rows) != int(spec["expected_flickr_validation_captions"]):
        raise ValueError(f"Flickr validation caption count mismatch: {len(rows)}")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    test_images = {
        Path(value).stem
        for value in _group_csv(ROOT / str(spec["flickr_test_csv"]))
    }
    validation_ids = {Path(value).stem for value in unique}
    overlap = validation_ids & test_images
    if overlap:
        raise RuntimeError(f"Flickr validation/test overlap: {len(overlap)} images")
    manifest = {
        "status": "COMPLETE",
        "source": "cached nlphuji/flickr30k Arrow shards",
        "network_used": False,
        "cache_shards": [str(Path(value).relative_to(ROOT)) for value in paths],
        "cache_shard_sha256": {str(Path(value).relative_to(ROOT)): sha256_file(value) for value in paths},
        "split_field": "val",
        "csv": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256_file(csv_path),
        "images": len(unique),
        "captions": len(rows),
        "test_overlap_images": 0,
        "decision_bearing": True,
        "test_sealed": True,
        "image_hashes": image_hashes,
    }
    atomic_json(manifest, output_root(pipeline) / "manifests/flickr_validation.json")
    return manifest


def canonical_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    percentages = [int(value) for value in pipeline["split"]["percentages"]]
    jobs: list[dict[str, Any]] = []
    for percentage in percentages:
        memberships = ["practical_scaling"]
        if percentage == 100:
            memberships.append("fixed_update")
        jobs.append(
            {
                "percentage": percentage,
                "schedule": "practical_12_epochs",
                "memberships": memberships,
            }
        )
    for percentage in percentages:
        if percentage == 100:
            continue
        jobs.append(
            {
                "percentage": percentage,
                "schedule": "fixed_optimizer_steps",
                "memberships": ["fixed_update"],
            }
        )
    jobs.sort(key=lambda row: (row["percentage"], row["schedule"]))
    for index, job in enumerate(jobs):
        job["index"] = index
        job["seed"] = int(pipeline["experiment"]["seed"])
        job["run_id"] = f"coco_{job['percentage']:03d}pct__{job['schedule']}__seed_{job['seed']}"
    if len(jobs) != 7:
        raise AssertionError(f"expected seven unique pilot cells, got {len(jobs)}")
    return jobs


def _subset_manifest(pipeline: dict[str, Any]) -> dict[str, Any]:
    path = output_root(pipeline) / "manifests/coco_subsets.json"
    if not path.is_file():
        raise FileNotFoundError("COCO subset manifest is missing")
    return json.loads(path.read_text())


def build_config(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(load_config(ROOT / str(pipeline["base_config"])))
    subsets = {int(row["percentage"]): row for row in _subset_manifest(pipeline)["subsets"]}
    subset = subsets[int(job["percentage"])]
    full = subsets[100]
    batch = int(pipeline["experiment"]["batch_size"])
    full_steps_per_epoch = int(full["images"]) // batch
    target_steps = full_steps_per_epoch * int(pipeline["experiment"]["practical_epochs"])
    subset_steps_per_epoch = int(subset["images"]) // batch
    if subset_steps_per_epoch < 1:
        raise ValueError("subset produces zero optimizer steps")
    if job["schedule"] == "fixed_optimizer_steps":
        epochs = int(math.ceil(target_steps / subset_steps_per_epoch))
        max_steps: int | None = target_steps
    else:
        epochs = int(pipeline["experiment"]["practical_epochs"])
        max_steps = None
    checkpoint = ROOT / str(pipeline["checkpoint_root"]) / job["run_id"]
    config["seed"] = int(job["seed"])
    config["experiment_id"] = "coco_data_scale_pilot"
    config["run_id"] = job["run_id"]
    config["data"]["train_csv"] = subset["csv"]
    config["data"]["val_csv"] = pipeline["split"]["coco_dev_csv"]
    config["data"]["num_workers"] = int(pipeline["resources"]["dataloader_workers"])
    config["training"]["save_dir"] = str(checkpoint.relative_to(ROOT))
    config["training"]["epochs"] = epochs
    config["training"]["select_on_dev"] = False
    config["training"]["early_stopping_patience"] = epochs + 1
    config["training"].pop("max_optimizer_steps", None)
    if max_steps is not None:
        config["training"]["max_optimizer_steps"] = max_steps
    config["provenance"]["data_scale_pilot"] = {
        "percentage": int(job["percentage"]),
        "schedule": job["schedule"],
        "memberships": job["memberships"],
        "subset_csv_sha256": subset["csv_sha256"],
        "fixed_update_target_steps": target_steps,
        "actual_schedule_steps": max_steps or (epochs * subset_steps_per_epoch),
        "flickr_test_sealed": True,
    }
    return config


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    flickr_path = output_root(pipeline) / "manifests/flickr_validation.json"
    subsets_path = output_root(pipeline) / "manifests/coco_subsets.json"
    if not flickr_path.is_file() or not subsets_path.is_file():
        raise RuntimeError("run prepare before validating the data-scale pilot")
    flickr = json.loads(flickr_path.read_text())
    subsets = json.loads(subsets_path.read_text())
    if flickr["test_overlap_images"] != 0 or not flickr["test_sealed"]:
        raise RuntimeError("Flickr validation provenance gate failed")
    if not subsets["nested"]:
        raise RuntimeError("COCO subsets are not nested")
    predictions_path = ROOT / str(pipeline["predictions_path"])
    if not predictions_path.is_file():
        raise RuntimeError(
            f"prediction gate is missing: {predictions_path}. "
            "Predictions must be approved before GPU submission."
        )
    predictions = json.loads(predictions_path.read_text())
    statements = predictions.get("unseen_predictions", [])
    if not statements or any(not str(value.get("statement", "")).strip() for value in statements):
        raise RuntimeError("prediction gate contains an empty statement")
    prediction_ids = {str(value.get("id", "")) for value in statements}
    expected_ids = {
        "practical_curve",
        "fixed_update_curve",
        "saturation_point",
        "overfitting_check",
    }
    if prediction_ids != expected_ids:
        raise RuntimeError(
            f"prediction gate IDs differ from the approved design: {prediction_ids}"
        )
    if not predictions.get("written_before_any_runs"):
        raise RuntimeError("prediction gate is not declared pre-run")
    decision_rule = predictions.get("decision_rule", {})
    if not all(
        str(decision_rule.get(key, "")).strip()
        for key in ("cc3m_justified", "cc3m_not_justified", "inconclusive")
    ):
        raise RuntimeError("prediction gate contains an incomplete CC3M decision rule")
    configs = []
    for job in canonical_jobs(pipeline):
        config = build_config(pipeline, job)
        configs.append(
            {
                **job,
                "train_csv": config["data"]["train_csv"],
                "epochs": config["training"]["epochs"],
                "max_optimizer_steps": config["training"].get("max_optimizer_steps"),
                "resolved_lr": config["training"]["lr"],
                "config_hash": hash_config(config),
            }
        )
    result = {
        "status": "READY",
        "unique_training_jobs": 7,
        "flickr_validation": {
            "images": flickr["images"],
            "captions": flickr["captions"],
            "test_overlap_images": 0,
        },
        "jobs": configs,
        "cc3m_decision_source": "fixed_update_curve_only",
        "test_evaluation_jobs": 0,
        "prediction_gate": "PASS",
    }
    atomic_json(result, output_root(pipeline) / "manifests/validation.json")
    return result


def train_job(pipeline: dict[str, Any], index: int | None, *, resume: bool = True) -> dict[str, Any]:
    _hardware_guard(pipeline)
    job = _select(canonical_jobs(pipeline), index)
    config = build_config(pipeline, job)
    fingerprint = _fingerprint(config)
    checkpoint = ROOT / str(config["training"]["save_dir"])
    existing = read_fingerprint(checkpoint / "fingerprint.json")
    if existing is not None and existing.digest != fingerprint.digest:
        raise RuntimeError(f"refusing cross-config resume: {checkpoint}")
    if (checkpoint / "inference.pt").is_file() and existing is not None:
        return {"status": "SKIPPED_FRESH", **job}
    result = train(config, fingerprint, resume=resume)
    return {**job, **result}


def evaluate_job(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = _hardware_guard(pipeline)
    job = _select(canonical_jobs(pipeline), index)
    config = build_config(pipeline, job)
    checkpoint = ROOT / str(config["training"]["save_dir"])
    fingerprint = read_fingerprint(checkpoint / "fingerprint.json")
    if fingerprint is None or hash_config(load_config(checkpoint / "config.yaml")) != fingerprint.config_hash:
        raise RuntimeError(f"checkpoint integrity failure: {checkpoint}")
    device = provenance.pop("device")
    model = build_model(config).to(device).eval()
    load_training_checkpoint(
        checkpoint / "best.pt",
        model,
        device=device,
        expected_fingerprint=fingerprint,
    )
    destination = output_root(pipeline) / "per_run" / job["run_id"]
    result = {
        "status": "COMPLETE",
        **job,
        **provenance,
        "checkpoint_fingerprint": fingerprint.digest,
    }
    datasets = {
        "coco_dev": pipeline["split"]["coco_dev_csv"],
        "flickr30k_validation": pipeline["split"]["flickr_validation_csv"],
    }
    for dataset_id, path in datasets.items():
        loader = _student_loader(config, ROOT / str(path), 256)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            embeddings = extract_embeddings(model, loader, device)
        metrics = _metrics_from_embeddings(embeddings, [1, 5, 10])
        payload = {
            "status": "COMPLETE",
            "dataset_id": dataset_id,
            "dataset_sha256": sha256_file(ROOT / str(path)),
            **job,
            **metrics,
            **provenance,
            "checkpoint_fingerprint": fingerprint.digest,
        }
        atomic_json(payload, destination / f"{dataset_id}_metrics.json")
        result[dataset_id] = metrics
    loss_batch_size = int(pipeline["experiment"]["batch_size"])

    def post_training_loss(csv_path: Path) -> float:
        loader = _student_loader(config, csv_path, loss_batch_size)
        losses: list[float] = []
        model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                captions = [str(value) for value in batch["captions"]]
                image_ids = [str(value) for value in batch["image_paths"]]
                text_ids = [
                    str(value)
                    for value in batch.get("text_image_paths", image_ids)
                ]
                outputs = model(images, captions)
                loss = contrastive_loss_with_memory(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    outputs["logit_scale"],
                    image_ids,
                    text_ids,
                    memory=None,
                )
                losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError(f"loss evaluation produced no batches: {csv_path}")
        return sum(losses) / len(losses)

    train_loss = post_training_loss(ROOT / str(config["data"]["train_csv"]))
    dev_loss = post_training_loss(ROOT / str(pipeline["split"]["coco_dev_csv"]))
    result["post_training_loss"] = {
        "protocol": "final checkpoint, no queue, identical batch size on train and dev",
        "batch_size": loss_batch_size,
        "train": train_loss,
        "dev": dev_loss,
        "dev_minus_train": dev_loss - train_loss,
    }
    atomic_json(result, destination / "evaluation.json")
    return result


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for job in canonical_jobs(pipeline):
        path = output_root(pipeline) / "per_run" / job["run_id"] / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation: {path}")
        payload = json.loads(path.read_text())
        checkpoint_fingerprint = payload.get("checkpoint_fingerprint")
        if checkpoint_fingerprint is None:
            config = build_config(pipeline, job)
            fingerprint = read_fingerprint(
                ROOT
                / str(config["training"]["save_dir"])
                / "fingerprint.json"
            )
            if fingerprint is None:
                raise RuntimeError(
                    f"missing checkpoint fingerprint for report row {job['run_id']}"
                )
            checkpoint_fingerprint = fingerprint.digest
        rows.append(
            {
                **job,
                "coco_dev_mean_R1": payload["coco_dev"]["mean_R@1"],
                "flickr_validation_mean_R1": payload["flickr30k_validation"]["mean_R@1"],
                "post_training_train_loss": payload["post_training_loss"]["train"],
                "post_training_dev_loss": payload["post_training_loss"]["dev"],
                "post_training_dev_minus_train_loss": payload[
                    "post_training_loss"
                ]["dev_minus_train"],
                "checkpoint_fingerprint": checkpoint_fingerprint,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, output_root(pipeline) / "data_scale_results.csv")
    practical = frame[frame["memberships"].apply(lambda value: "practical_scaling" in value)]
    fixed = frame[frame["memberships"].apply(lambda value: "fixed_update" in value)]
    practical = practical.sort_values("percentage")
    fixed = fixed.sort_values("percentage")
    fixed_scores = [
        float(value) * 100.0 for value in fixed["flickr_validation_mean_R1"]
    ]
    fixed_delta_pp = fixed_scores[-1] - fixed_scores[0]
    fixed_monotonic = all(
        later >= earlier for earlier, later in zip(fixed_scores, fixed_scores[1:])
    )
    if not fixed_monotonic:
        cc3m_decision = "INCONCLUSIVE"
    elif fixed_delta_pp > 1.0:
        cc3m_decision = "CC3M_JUSTIFIED"
    else:
        cc3m_decision = "CC3M_NOT_JUSTIFIED"
    result = {
        "status": "COMPLETE",
        "cc3m_decision": cc3m_decision,
        "cc3m_decision_source": "fixed_update_curve_only",
        "fixed_update_100_minus_25_pp": fixed_delta_pp,
        "fixed_update_monotonic": fixed_monotonic,
        "materiality_margin_pp": 1.0,
        "replication_scope": {
            "pilot": "Single-seed diagnostic (seed 42).",
            "interpretation": (
                "The fixed-update result justifies broader data under the "
                "pre-registered rule but does not establish a scaling law."
            ),
            "deferred_replication": (
                "Replication is deferred to the CC3M runs, which will use "
                "three seeds."
            ),
        },
        "flickr_test_evaluated": False,
        "unique_runs": 7,
        "practical_curve": practical.to_dict(orient="records"),
        "fixed_update_curve": fixed.to_dict(orient="records"),
        "fallback_interpretations": pipeline["decision"]["fallback_interpretations"],
    }
    atomic_json(result, output_root(pipeline) / "report.json")
    return result


def prepare(pipeline: dict[str, Any]) -> dict[str, Any]:
    return {
        "flickr_validation": prepare_flickr_validation(pipeline),
        "coco_subsets": prepare_subsets(pipeline),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "validate", "train", "evaluate", "report"),
    )
    parser.add_argument("--pipeline", default=PIPELINE_DEFAULT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    pipeline = load_pipeline(args.pipeline)
    if args.command == "prepare":
        result = prepare(pipeline)
    elif args.command == "validate":
        result = validate(pipeline)
    elif args.command == "train":
        result = train_job(pipeline, args.index, resume=not args.no_resume)
    elif args.command == "evaluate":
        result = evaluate_job(pipeline, args.index)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
