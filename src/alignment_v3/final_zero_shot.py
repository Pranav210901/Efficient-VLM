from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from freezeshift.model_adaptation import build_model as build_freezeshift_model
from src.alignment_v3.efficiency_frontier import (
    _reference_weight_manifest,
    parameter_counts_reference,
    parameter_counts_student,
)
from src.alignment_v3.fingerprint import Fingerprint, hash_config, hash_payload, sha256_file
from src.alignment_v3.model import build_model as build_alignment_model
from src.alignment_v3.references import REFERENCE_CHECKPOINTS, build_reference
from src.alignment_v3.runner import _metrics_from_embeddings
from src.data.collate import image_text_collate
from src.data.transforms import build_image_transform
from src.multitask.classification_evaluator import (
    IndexedClassificationDataset,
    classification_predictions,
    evaluate_zeroshot_classification,
)
from src.multitask.datasets import (
    ZERO_SHOT_DATASET_PATHS,
    dataset_split_id,
    load_zeroshot_dataset,
    zeroshot_split_name,
    zeroshot_subset_indices,
)
from src.multitask.prompt_templates import clean_class_name
from src.phase15.evaluation_protocol import create_eurosat_split_manifest
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.training.evaluate import extract_embeddings
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = "configs/final_zero_shot/pipeline.yaml"
QUARANTINE_ROOT = ROOT / "quarantine/github_submission_cleanup_20260810/original_paths"


def _pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return load_config(value if value.is_absolute() else ROOT / value)


def _output(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for entry in pipeline["students"]:
        for seed, spec in entry["seeds"].items():
            jobs.append(
                {
                    "kind": "student",
                    "id": str(entry["id"]),
                    "label": str(entry["label"]),
                    "builder": str(entry["builder"]),
                    "seed": int(seed),
                    "selected_epoch": int(spec["selected_epoch"]),
                    "checkpoint": str(spec["checkpoint"]),
                    "expected_inference_trainable_parameters": int(
                        entry["expected_inference_trainable_parameters"]
                    ),
                }
            )
    for reference_id in pipeline["references"]:
        jobs.append(
            {
                "kind": "reference",
                "id": str(reference_id),
                "label": str(reference_id),
                "builder": "open_clip",
                "seed": None,
                "selected_epoch": None,
                "checkpoint": None,
            }
        )
    return jobs


def _select(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else index
    jobs = _jobs(pipeline)
    if selected < 0 or selected >= len(jobs):
        raise IndexError(f"evaluation index {selected} outside 0..{len(jobs)-1}")
    return jobs[selected]


def _hardware_guard(pipeline: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("final zero-shot evaluation requires CUDA")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    guard = pipeline["hardware"]
    if str(guard["gpu_substring"]) not in name:
        raise RuntimeError(f"expected {guard['gpu_substring']}, got {name}")
    if bool(guard.get("require_blackwell", False)) and "Blackwell" not in name:
        raise RuntimeError(f"native Blackwell comparison required, got {name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("native BF16 support is required")
    return device, {
        "node": platform.node(),
        "gpu_model": name,
        "precision": "native_bf16_autocast",
        "torch_version": torch.__version__,
    }


def _sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path).encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def prepare_data(pipeline: dict[str, Any], *, download: bool) -> dict[str, Any]:
    """Acquire missing public datasets and freeze dataset identities; no model is loaded."""
    try:
        from torchvision.datasets import CIFAR100, EuroSAT, OxfordIIITPet
    except Exception as exc:
        raise RuntimeError("torchvision is required to prepare classification datasets") from exc

    roots = {name: ROOT / path for name, path in ZERO_SHOT_DATASET_PATHS.items()}
    constructors: list[tuple[str, Callable[[], Any]]] = [
        ("cifar100_zeroshot", lambda: CIFAR100(root=roots["cifar100_zeroshot"], train=False, download=download)),
        ("pets_zeroshot", lambda: OxfordIIITPet(root=roots["pets_zeroshot"], split="test", target_types="category", download=download)),
        ("eurosat_zeroshot", lambda: EuroSAT(root=roots["eurosat_zeroshot"], download=download)),
    ]
    datasets: dict[str, Any] = {}
    for name, constructor in constructors:
        try:
            datasets[name] = constructor()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{name} is unavailable. Run prepare with --download from a network-enabled submit node."
            ) from exc
    create_eurosat_split_manifest(ROOT)

    identities: dict[str, Any] = {}
    for name, dataset in datasets.items():
        indices = zeroshot_subset_indices(ROOT, name, dataset, "final_test")
        chosen = list(range(len(dataset))) if indices is None else list(indices)
        labels = list(getattr(dataset, "targets", getattr(dataset, "_labels", [])))
        samples = getattr(dataset, "samples", getattr(dataset, "_images", None))
        rows = []
        for index in chosen:
            source = samples[index][0] if samples is not None and isinstance(samples[index], (tuple, list)) else (samples[index] if samples is not None else f"{name}:{index}")
            target = int(labels[index]) if labels else int(dataset[index][1])
            rows.append((str(source), target))
        identities[name] = {
            "dataset_split_id": dataset_split_id(name, "final_test"),
            "split": zeroshot_split_name(name, "final_test"),
            "samples": len(chosen),
            "classes": len(dataset.classes),
            "semantic_sha256": hash_payload(rows),
        }

    flickr = pipeline["datasets"]["flickr30k_test"]
    source_csv = ROOT / str(flickr["csv"])
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    resolved_rows: list[dict[str, str]] = []
    with source_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(str(row["image_path"]))
            candidates = [source if source.is_absolute() else ROOT / source]
            if not source.is_absolute():
                candidates.append(QUARANTINE_ROOT / source)
            path = next((value for value in candidates if value.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Flickr image unavailable: {source}")
            resolved_rows.append({"image_path": str(path.resolve()), "caption": str(row["caption"])})
    image_count = len({row["image_path"] for row in resolved_rows})
    if len(resolved_rows) != int(flickr["expected_captions"]) or image_count != int(flickr["expected_images"]):
        raise ValueError(f"Flickr structure mismatch: {image_count} images/{len(resolved_rows)} captions")
    destination = _output(pipeline) / "manifests/resolved_flickr30k_test.csv"
    atomic_csv(pd.DataFrame(resolved_rows), destination)
    identities["flickr30k_test"] = {
        "dataset_split_id": "flickr30k_karpathy_test",
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "resolved_csv": str(destination),
        "resolved_csv_sha256": sha256_file(destination),
        "samples": image_count,
        "captions": len(resolved_rows),
        "image_path_manifest_sha256": hash_payload(sorted({row["image_path"] for row in resolved_rows})),
    }
    payload = {
        "status": "READY",
        "download_permitted": bool(download),
        "datasets": identities,
        "frozen_roster_sha256": hash_config(_jobs(pipeline)),
        "pipeline_sha256": sha256_file(ROOT / DEFAULT_PIPELINE),
    }
    atomic_json(payload, _output(pipeline) / "manifests/data_manifest.json")
    return payload


class GroupedRetrievalDataset(Dataset):
    def __init__(self, csv_path: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault(str(row["image_path"]), []).append(str(row["caption"]))
        self.rows = list(grouped.items())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        path, captions = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "caption": captions, "image_path": path}


def _load_student(job: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    path = ROOT / str(job["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(path)
    export = torch.load(path, map_location="cpu", weights_only=False)
    fingerprint = Fingerprint.from_dict(export["fingerprint"])
    config = dict(export["config"])
    training_config = dict(config)
    training_config["recipe"] = dict(export["training_recipe"])
    if hash_config(training_config) != fingerprint.config_hash:
        raise RuntimeError(f"checkpoint config/fingerprint mismatch: {path}")
    if bool(config.get("recipe", {}).get("distillation", False)):
        raise RuntimeError("inference export incorrectly retains distillation")
    leaked = [key for key in export["model_state"] if key.startswith(("teacher_image_head.", "teacher_text_head.", "teacher_heads."))]
    if leaked or not bool(export.get("training_only_heads_removed", False)):
        raise RuntimeError(f"training-only heads leaked into inference export: {leaked[:3]}")
    builder = build_alignment_model if job["builder"] == "alignment_v3" else build_freezeshift_model
    model = builder(config, pretrained=False)
    model.load_state_dict(export["model_state"], strict=True)
    model.to(device).eval()
    summary = model.parameter_summary()
    expected = int(job["expected_inference_trainable_parameters"])
    if int(summary["params_trainable_inference"]) != expected:
        raise RuntimeError(f"inference trainable parameter mismatch: {summary} expected={expected}")
    counts = parameter_counts_student(model)
    provenance = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_fingerprint": fingerprint.digest,
        "config_hash": fingerprint.config_hash,
        "selected_epoch": job["selected_epoch"],
        "inference_trainable_parameters": expected,
        **counts,
    }
    return model, config, provenance


def _classification_dataset(task: str, transform: Callable[[Image.Image], torch.Tensor]) -> tuple[Dataset, list[str]]:
    dataset = load_zeroshot_dataset(ROOT, task, transform=transform, split_role="final_test")
    wrapped = IndexedClassificationDataset(dataset, task, ROOT)
    indices = zeroshot_subset_indices(ROOT, task, dataset, "final_test")
    selected: Dataset = wrapped if indices is None else torch.utils.data.Subset(wrapped, indices)
    return selected, [clean_class_name(str(value)) for value in dataset.classes]


def _student_transform(config: dict[str, Any]):
    data = config["data"]
    return build_image_transform(
        image_size=int(data["image_size"]),
        train=False,
        mean=data.get("image_mean"),
        std=data.get("image_std"),
        interpolation=str(data.get("interpolation", "bicubic")),
    )


def evaluate(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    manifest_path = _output(pipeline) / "manifests/data_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("data manifest is missing; run prepare first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("frozen_roster_sha256") != hash_config(_jobs(pipeline)):
        raise RuntimeError("frozen roster changed after data preparation")
    device, hardware = _hardware_guard(pipeline)
    job = _select(pipeline, index)
    if job["kind"] == "student":
        model, config, model_provenance = _load_student(job, device)
        transform = _student_transform(config)
    else:
        model = build_reference(job["id"]).to(device).eval()
        config = None
        transform = model.preprocess
        model_provenance = {
            **parameter_counts_reference(model),
            "inference_trainable_parameters": 0,
            "reference_weights": _reference_weight_manifest(model, job["id"]),
        }
    if any(module.training for module in model.modules()):
        raise RuntimeError("evaluation model contains a module in training mode")

    destination = _output(pipeline) / "per_run" / job["id"] / (
        f"seed_{job['seed']}" if job["seed"] is not None else "reference"
    )
    destination.mkdir(parents=True, exist_ok=True)
    resolved_csv = _output(pipeline) / "manifests/resolved_flickr30k_test.csv"
    retrieval_loader = DataLoader(
        GroupedRetrievalDataset(resolved_csv, transform),
        batch_size=int(pipeline["evaluation"]["retrieval_batch_size"]),
        shuffle=False,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        pin_memory=True,
        persistent_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) > 0,
        collate_fn=image_text_collate,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = extract_embeddings(model, retrieval_loader, device)
    retrieval_metrics = _metrics_from_embeddings(embeddings, list(pipeline["evaluation"]["k_values"]))
    torch.save(
        {key: value for key, value in embeddings.items() if key in {"image_embeds", "text_embeds", "image_paths", "text_image_paths"}},
        destination / "flickr30k_test_embeddings.pt",
    )
    atomic_json({"dataset": "flickr30k_karpathy_test", **retrieval_metrics}, destination / "flickr30k_test_metrics.json")

    classification: dict[str, Any] = {}
    for task in pipeline["datasets"]["classification"]:
        dataset, class_names = _classification_dataset(str(task), transform)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = evaluate_zeroshot_classification(
                model,
                dataset,
                class_names,
                device,
                str(task),
                batch_size=int(pipeline["evaluation"]["classification_batch_size"]),
                num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
                templates=tuple(pipeline["evaluation"]["prompt_templates"]),
            )
        predictions = classification_predictions(
            result["logits"], result["targets"], result["sample_ids"], result["class_names"], str(task)
        )
        atomic_csv(pd.DataFrame(predictions), destination / f"{task}_predictions.csv")
        atomic_csv(pd.DataFrame(result["per_class"]), destination / f"{task}_per_class.csv")
        payload = {
            "dataset_split_id": dataset_split_id(str(task), "final_test"),
            "split": zeroshot_split_name(str(task), "final_test"),
            **result["metrics"],
        }
        atomic_json(payload, destination / f"{task}_metrics.json")
        classification[str(task)] = payload

    payload = {
        "status": "COMPLETE",
        **job,
        **hardware,
        **model_provenance,
        "flickr30k_test": retrieval_metrics,
        "classification": classification,
        "no_training_performed": True,
    }
    atomic_json(payload, destination / "evaluation.json")
    return payload


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    jobs = _jobs(pipeline)
    if len(jobs) != 9 or sum(job["kind"] == "student" for job in jobs) != 6:
        raise RuntimeError(f"expected 6 student + 3 reference jobs, got {len(jobs)}")
    if set(pipeline["references"]) != set(REFERENCE_CHECKPOINTS):
        raise RuntimeError("reference registry does not match the locally established frontier")
    checkpoints = []
    for job in jobs:
        if job["kind"] != "student":
            continue
        path = ROOT / str(job["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(path)
        export = torch.load(path, map_location="cpu", weights_only=False)
        fingerprint = Fingerprint.from_dict(export["fingerprint"])
        training_config = dict(export["config"])
        training_config["recipe"] = dict(export["training_recipe"])
        if hash_config(training_config) != fingerprint.config_hash:
            raise RuntimeError(f"config fingerprint mismatch: {path}")
        checkpoints.append({**job, "sha256": sha256_file(path), "fingerprint": fingerprint.digest})
    result = {
        "status": "READY",
        "job_count": len(jobs),
        "student_jobs": 6,
        "reference_jobs": 3,
        "checkpoints": checkpoints,
        "no_optimizer_code_path": True,
    }
    atomic_json(result, _output(pipeline) / "manifests/validation.json")
    return result


def _mean_sd(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        min(values),
        max(values),
    )


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for job in _jobs(pipeline):
        path = _output(pipeline) / "per_run" / job["id"] / (
            f"seed_{job['seed']}" if job["seed"] is not None else "reference"
        ) / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation artifact: {path}")
        evaluations.append(json.loads(path.read_text()))
    metric_specs = [("flickr30k_test", "mean_R@1")]
    metric_specs += [(str(task), "top1_accuracy") for task in pipeline["datasets"]["classification"]]
    for value in evaluations:
        for dataset, metric in metric_specs:
            source = value[dataset] if dataset == "flickr30k_test" else value["classification"][dataset]
            rows.append(
                {
                    "model_id": value["id"],
                    "label": value["label"],
                    "kind": value["kind"],
                    "seed": value["seed"],
                    "dataset": dataset,
                    "primary_metric": metric,
                    "value": float(source[metric]),
                    "i2t_R@1": source.get("i2t_R@1"),
                    "t2i_R@1": source.get("t2i_R@1"),
                    "top5_accuracy": source.get("top5_accuracy"),
                    "macro_f1": source.get("macro_f1"),
                    "balanced_accuracy": source.get("balanced_accuracy"),
                    "inference_trainable_parameters": value["inference_trainable_parameters"],
                    "full_stack_inference_parameters": value["full_stack_inference_parameters"],
                }
            )
    frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (model_id, dataset), group in frame.groupby(["model_id", "dataset"], sort=False):
        mean, sd, minimum, maximum = _mean_sd(group["value"].tolist())
        aggregate_rows.append(
            {
                "model_id": model_id,
                "label": group.iloc[0]["label"],
                "kind": group.iloc[0]["kind"],
                "dataset": dataset,
                "metric": group.iloc[0]["primary_metric"],
                "n": len(group),
                "mean": mean,
                "sd": sd,
                "min": minimum,
                "max": maximum,
                "inference_trainable_parameters": int(group.iloc[0]["inference_trainable_parameters"]),
                "full_stack_inference_parameters": int(group.iloc[0]["full_stack_inference_parameters"]),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    report_root = _output(pipeline) / "report"
    atomic_csv(frame, report_root / "per_seed_results.csv")
    atomic_csv(aggregate, report_root / "aggregate_results.csv")
    lines = [
        "# Final zero-shot evaluation",
        "",
        "No learning or parameter updates were performed. Student values are three-seed mean ± SD; references are fixed single checkpoints.",
        "",
        aggregate.to_markdown(index=False),
        "",
        "Flickr30k test is the primary final transfer evaluation. Classification benchmarks are external zero-shot diagnostics with historical project exposure and are not described as pristine confirmatory tests.",
    ]
    atomic_text("\n".join(lines) + "\n", report_root / "report.md")
    payload = {"status": "COMPLETE", "evaluations": len(evaluations), "aggregate_rows": aggregate_rows}
    atomic_json(payload, report_root / "report.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "prepare", "evaluate", "report"))
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    pipeline = _pipeline(args.pipeline)
    if args.command == "validate":
        result = validate(pipeline)
    elif args.command == "prepare":
        result = prepare_data(pipeline, download=args.download)
    elif args.command == "evaluate":
        result = evaluate(pipeline, args.index)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
