from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.alignment_v3.fingerprint import (
    Fingerprint,
    hash_config,
    read_fingerprint,
    sha256_file,
)
from src.alignment_v3.model import AlignmentV3Model, build_model
from src.alignment_v3.references import (
    REFERENCE_CHECKPOINTS,
    NativePairedModel,
    build_reference,
    native_loader,
)
from src.alignment_v3.runner import _fingerprint, _metrics_from_embeddings
from src.alignment_v3.splits import hash_dataset_split
from src.alignment_v3.training import load_training_checkpoint, train
from src.data import build_dataloaders
from src.data.transforms import build_image_transform
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.training.evaluate import extract_embeddings
from src.utils.config import deep_update, load_config
from src.utils.device import get_device


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DEFAULT = "configs/efficiency_frontier/pipeline.yaml"


def load_pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return load_config(value)


def output_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _select(values: list[Any], index: int | None) -> Any:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else int(index)
    if selected < 0 or selected >= len(values):
        raise IndexError(f"index {selected} outside manifest of size {len(values)}")
    return values[selected]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or not {"image_path", "caption"}.issubset(rows[0]):
        raise ValueError(f"invalid image-caption CSV: {path}")
    return rows


def _dataset_audit(spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(spec["csv"])
    rows = _read_rows(path)
    image_paths = sorted({str(row["image_path"]) for row in rows})
    missing = [
        value
        for value in image_paths
        if not (Path(value) if Path(value).is_absolute() else ROOT / value).is_file()
    ]
    result = {
        "id": spec["id"],
        "csv": str(path.relative_to(ROOT)),
        "csv_sha256": sha256_file(path),
        "caption_count": len(rows),
        "image_count": len(image_paths),
        "semantic_image_hash": hash_dataset_split(image_paths),
        "missing_image_count": len(missing),
        "missing_images": missing[:20],
    }
    if len(rows) != int(spec["expected_captions"]):
        raise ValueError(f"{spec['id']} caption count mismatch: {len(rows)}")
    if len(image_paths) != int(spec["expected_images"]):
        raise ValueError(f"{spec['id']} image count mismatch: {len(image_paths)}")
    if missing:
        raise FileNotFoundError(f"{spec['id']} is missing {len(missing)} images")
    expected_hash = spec.get("semantic_hash")
    if expected_hash and result["semantic_image_hash"] != expected_hash:
        raise ValueError(f"{spec['id']} semantic hash mismatch")
    return result


def _checkpoint_bundle(path: Path) -> dict[str, Any]:
    required = ("best.pt", "config.yaml", "fingerprint.json", "metrics.csv", "run_summary.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return {"path": str(path.relative_to(ROOT)), "status": "MISSING", "missing": missing}
    config = load_config(path / "config.yaml")
    fingerprint = read_fingerprint(path / "fingerprint.json")
    if fingerprint is None:
        raise ValueError(f"invalid fingerprint: {path}")
    if hash_config(config) != fingerprint.config_hash:
        raise ValueError(f"config/fingerprint mismatch: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "status": "COMPLETE",
        "seed": int(config["seed"]),
        "fingerprint": fingerprint.digest,
        "checkpoint_bytes": (path / "best.pt").stat().st_size,
        "precision": str(config["training"]["precision"]),
        "loss_type": str(config["recipe"]["loss_type"]),
        "batch_size": int(config["training"]["batch_size"]),
        "memory_queue_size": int(config["training"]["memory_queue_size"]),
    }


def student_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for entry in pipeline["students"]:
        for seed, value in entry["seeds"].items():
            jobs.append(
                {
                    "kind": "student",
                    "entry_id": str(entry["id"]),
                    "label": str(entry["label"]),
                    "role": str(entry["role"]),
                    "seed": int(seed),
                    "checkpoint_dir": str(value["checkpoint_dir"]),
                }
            )
    return jobs


def reference_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "reference",
            "entry_id": str(value["id"]),
            "label": str(value["checkpoint_id"]),
            "role": "reference",
            "seed": None,
        }
        for value in pipeline["references"]["entries"]
    ]


def evaluation_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    return student_jobs(pipeline) + reference_jobs(pipeline)


def profiling_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    representative = int(pipeline["profiling"]["representative_seed"])
    verification = int(pipeline["profiling"]["verification_seed"])
    students = [
        job for job in student_jobs(pipeline) if job["seed"] in {representative, verification}
    ]
    return students + reference_jobs(pipeline)


def missing_v4_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "run_id": f"historical_v4__seed_{int(seed)}",
            "checkpoint_dir": (
                f"{pipeline['checkpoint_root']}/v4_missing_seeds/"
                f"historical_v4__seed_{int(seed)}"
            ),
        }
        for seed in pipeline["historical_v4_completion"]["train_seeds"]
    ]


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    datasets = {
        name: _dataset_audit(spec)
        for name, spec in pipeline["datasets"].items()
        if name in {"primary", "supplementary"}
    }
    bundles = [
        {**job, **_checkpoint_bundle(ROOT / job["checkpoint_dir"])}
        for job in student_jobs(pipeline)
    ]
    missing = [row for row in bundles if row["status"] == "MISSING"]
    allowed = {
        f"{pipeline['checkpoint_root']}/v4_missing_seeds/historical_v4__seed_43",
        f"{pipeline['checkpoint_root']}/v4_missing_seeds/historical_v4__seed_44",
    }
    unexpected = [row for row in missing if row["checkpoint_dir"] not in allowed]
    if unexpected:
        raise RuntimeError(f"unexpected missing checkpoint bundles: {unexpected}")
    source = load_config(
        ROOT / str(pipeline["historical_v4_completion"]["source_seed_42_config"])
    )
    assert_v4_source_config(source, pipeline["historical_v4_completion"]["expected"])
    refs = {
        str(value["id"]): (str(value["model_name"]), str(value["pretrained"]))
        for value in pipeline["references"]["entries"]
    }
    if refs != REFERENCE_CHECKPOINTS:
        raise ValueError(
            f"frontier reference registry differs from runtime registry: {refs} != {REFERENCE_CHECKPOINTS}"
        )
    result = {
        "status": "READY_WITH_EXPECTED_MISSING_V4_SEEDS" if missing else "READY",
        "datasets": datasets,
        "checkpoint_bundles": bundles,
        "expected_missing_v4_seeds": [row["seed"] for row in missing],
        "evaluation_jobs": len(evaluation_jobs(pipeline)),
        "profiling_jobs": len(profiling_jobs(pipeline)),
        "limitations": list(pipeline["limitations"]),
    }
    atomic_json(result, output_root(pipeline) / "manifests/validation.json")
    return result


def assert_v4_source_config(config: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = {
        "loss_type": config["recipe"]["loss_type"],
        "captions_per_image": config["recipe"]["captions_per_image"],
        "batch_size": config["training"]["batch_size"],
        "memory_queue_size": config["training"]["memory_queue_size"],
        "base_lr": config["training"]["base_lr"],
        "sqrt_scale_factor": config["training"]["sqrt_scale_factor"],
        "sweep_multiplier": config["training"]["sweep_multiplier"],
        "resolved_lr": config["training"]["lr"],
        "epochs": config["training"]["epochs"],
        "early_stopping_patience": config["training"]["early_stopping_patience"],
        "precision": config["training"]["precision"],
    }
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(f"historical v4 seed-42 config mismatch: {mismatches}")


def create_dataset_manifests(pipeline: dict[str, Any]) -> dict[str, Any]:
    destination = output_root(pipeline) / "manifests"
    primary = pipeline["datasets"]["primary"]
    flickr_audit = _dataset_audit(primary)
    rows = _read_rows(ROOT / str(primary["csv"]))
    image_paths = sorted({str(row["image_path"]) for row in rows})
    image_hashes = []
    for value in image_paths:
        path = Path(value) if Path(value).is_absolute() else ROOT / value
        image_hashes.append({"path": value, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(
        "".join(f"{row['path']}:{row['sha256']}\n" for row in image_hashes).encode()
    ).hexdigest()
    flickr = {
        **flickr_audit,
        "recorded_dataset": primary["recorded_dataset"],
        "recorded_revision": primary["recorded_revision"],
        "revision_cryptographically_verified": False,
        "aggregate_image_manifest_sha256": aggregate,
        "images": image_hashes,
        "limitation": (
            "The original fetch did not pass the recorded revision. Structural "
            "agreement and these prospective local hashes are the available evidence."
        ),
    }
    atomic_json(flickr, destination / "flickr30k_files.json")
    datasets = {
        "primary": flickr_audit,
        "supplementary": _dataset_audit(pipeline["datasets"]["supplementary"]),
        "coco_status": (
            "COCO validation was used for model selection in earlier project phases; "
            "it is not an independent test set."
        ),
        "untouched_coco_test_available": False,
    }
    atomic_json(datasets, destination / "datasets.json")
    return {"status": "COMPLETE", "flickr": flickr_audit, "datasets": datasets}


def build_missing_v4_config(
    pipeline: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    source_path = ROOT / str(
        pipeline["historical_v4_completion"]["source_seed_42_config"]
    )
    source = load_config(source_path)
    assert_v4_source_config(source, pipeline["historical_v4_completion"]["expected"])
    if int(job["seed"]) == int(pipeline["historical_v4_completion"]["protected_seed"]):
        raise RuntimeError("seed 42 is protected and must never be rebuilt")
    config = deepcopy(source)
    config["seed"] = int(job["seed"])
    config["run_id"] = str(job["run_id"])
    config["experiment_id"] = "historical_v4"
    config["training"]["save_dir"] = str(job["checkpoint_dir"])
    config["data"]["num_workers"] = int(pipeline["resources"]["cpus_per_task"])
    return config


def train_missing_v4(
    pipeline: dict[str, Any], index: int | None, *, resume: bool
) -> dict[str, Any]:
    _hardware_guard(pipeline)
    job = _select(missing_v4_jobs(pipeline), index)
    config = build_missing_v4_config(pipeline, job)
    fingerprint = _fingerprint(config)
    save_dir = ROOT / str(config["training"]["save_dir"])
    existing = read_fingerprint(save_dir / "fingerprint.json")
    if existing is not None and existing.digest != fingerprint.digest:
        raise RuntimeError(f"refusing cross-config resume: {save_dir}")
    if (save_dir / "inference.pt").is_file() and existing is not None:
        return {"status": "SKIPPED_FRESH", **job}
    result = train(config, fingerprint, resume=resume)
    return {**job, **result}


def _load_student(job: dict[str, Any], device: torch.device) -> tuple[AlignmentV3Model, dict[str, Any], Any]:
    checkpoint_dir = ROOT / str(job["checkpoint_dir"])
    bundle = _checkpoint_bundle(checkpoint_dir)
    if bundle["status"] != "COMPLETE":
        raise RuntimeError(f"incomplete checkpoint bundle: {bundle}")
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    assert fingerprint is not None
    inference_path = checkpoint_dir / "inference.pt"
    if inference_path.is_file():
        export = torch.load(inference_path, map_location=device, weights_only=False)
        export_fingerprint = Fingerprint.from_dict(export["fingerprint"])
        if export_fingerprint.digest != fingerprint.digest:
            raise AssertionError(
                "inference export fingerprint does not match checkpoint bundle"
            )
        config = dict(export["config"])
        if bool(config.get("recipe", {}).get("distillation", False)):
            raise AssertionError("inference export still enables distillation heads")
        model = build_model(config).to(device).eval()
        model.load_state_dict(export["model_state"], strict=True)
        leaked = [
            key
            for key in export["model_state"]
            if key.startswith(
                ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
            )
        ]
        if leaked:
            raise AssertionError(
                f"training-only teacher heads leaked into inference export: {leaked[:3]}"
            )
    else:
        config = load_config(checkpoint_dir / "config.yaml")
        model = build_model(config).to(device).eval()
        load_training_checkpoint(
            checkpoint_dir / "best.pt",
            model,
            device=device,
            expected_fingerprint=fingerprint,
        )
    return model, config, fingerprint


def _hardware_guard(pipeline: dict[str, Any]) -> dict[str, Any]:
    device = get_device("auto")
    if device.type != "cuda":
        raise RuntimeError("frontier GPU stages require CUDA")
    name = torch.cuda.get_device_name(device)
    if "RTX PRO 6000" not in name or "Blackwell" not in name:
        raise RuntimeError(f"frontier requires RTX PRO 6000 Blackwell, got {name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("native BF16 support is required")
    return {
        "device": device,
        "gpu_model": name,
        "node": platform.node(),
        "precision": "native_bf16",
    }


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _student_loader(config: dict[str, Any], csv_path: Path, batch_size: int):
    evaluation_config = deep_update(
        config,
        {
            "data": {"val_csv": str(csv_path)},
            "training": {"batch_size": int(batch_size)},
        },
    )
    _, loader = build_dataloaders(evaluation_config)
    return loader


def evaluate(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = _hardware_guard(pipeline)
    job = _select(evaluation_jobs(pipeline), index)
    destination = output_root(pipeline) / "per_run" / job["entry_id"] / (
        f"seed_{job['seed']}" if job["seed"] is not None else "reference"
    )
    device = provenance.pop("device")
    if job["kind"] == "student":
        model, config, fingerprint = _load_student(job, device)
        parameters = parameter_counts_student(model)
    else:
        model = build_reference(job["entry_id"]).to(device).eval()
        config = None
        fingerprint = None
        parameters = parameter_counts_reference(model)
    result: dict[str, Any] = {
        "status": "COMPLETE",
        **job,
        **parameters,
        **provenance,
    }
    for dataset_name in ("primary", "supplementary"):
        spec = pipeline["datasets"][dataset_name]
        csv_path = ROOT / str(spec["csv"])
        if job["kind"] == "student":
            loader = _student_loader(
                config, csv_path, int(pipeline["evaluation"]["batch_size"])
            )
        else:
            loader = native_loader(
                model,
                csv_path,
                image_root=ROOT,
                batch_size=int(pipeline["evaluation"]["batch_size"]),
                num_workers=int(pipeline["resources"]["cpus_per_task"]),
            )
        with _autocast(device):
            embeddings = extract_embeddings(model, loader, device)
        metrics = _metrics_from_embeddings(
            embeddings, list(pipeline["evaluation"]["k_values"])
        )
        dataset_id = str(spec["id"])
        payload = {
            "status": "COMPLETE",
            "dataset_id": dataset_id,
            "dataset_role": dataset_name,
            "dataset_sha256": sha256_file(csv_path),
            **job,
            **metrics,
            **provenance,
            **parameters,
            "checkpoint_fingerprint": fingerprint.digest if fingerprint else None,
        }
        atomic_json(payload, destination / f"{dataset_id}_metrics.json")
        result[dataset_id] = metrics
    atomic_json(result, destination / "evaluation.json")
    return result


def _unique_parameters(modules: Iterable[torch.nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total


def parameter_counts_student(model: AlignmentV3Model) -> dict[str, int]:
    full = _unique_parameters([model])
    query = _unique_parameters([model.text_encoder, model.text_projection])
    result = {
        "full_stack_inference_parameters": full,
        "query_side_inference_parameters": query,
    }
    verify_parameter_counts(model, result, kind="student")
    return result


def parameter_counts_reference(model: NativePairedModel) -> dict[str, int]:
    full = _unique_parameters([model])
    excluded = {"logit_scale", "logit_bias"}
    query_ids = {
        id(parameter)
        for name, parameter in model.model.named_parameters()
        if not name.startswith("visual.") and name not in excluded
    }
    query = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in query_ids
    )
    result = {
        "full_stack_inference_parameters": full,
        "query_side_inference_parameters": query,
    }
    verify_parameter_counts(model, result, kind="reference")
    return result


def verify_parameter_counts(
    model: torch.nn.Module, reported: dict[str, int], *, kind: str
) -> None:
    if kind == "student":
        leaked = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(
                ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
            )
        ]
        if leaked:
            raise AssertionError(
                "inference-parameter reporting cannot include training-only "
                f"teacher heads: {leaked[:3]}"
            )
    loaded = sum(parameter.numel() for parameter in model.parameters())
    if loaded != int(reported["full_stack_inference_parameters"]):
        raise AssertionError(
            f"{kind} loaded/full-stack parameter mismatch: {loaded} != "
            f"{reported['full_stack_inference_parameters']}"
        )
    if kind == "student":
        expected_query = _unique_parameters(
            [model.text_encoder, model.text_projection]  # type: ignore[attr-defined]
        )
    else:
        excluded = {"logit_scale", "logit_bias"}
        query_ids = {
            id(parameter)
            for name, parameter in model.model.named_parameters()  # type: ignore[attr-defined]
            if not name.startswith("visual.") and name not in excluded
        }
        expected_query = sum(
            parameter.numel() for parameter in model.parameters() if id(parameter) in query_ids
        )
    if expected_query != int(reported["query_side_inference_parameters"]):
        raise AssertionError(
            f"{kind} loaded/query-side parameter mismatch: {expected_query} != "
            f"{reported['query_side_inference_parameters']}"
        )


def _native_context(model: torch.nn.Module) -> int:
    if isinstance(model, NativePairedModel):
        return model.native_context_length
    return int(model.text_encoder.native_context_length)  # type: ignore[attr-defined]


def _tokenize_native(
    model: torch.nn.Module,
    captions: list[str],
    *,
    pad_to_native_context: bool = True,
):
    if isinstance(model, NativePairedModel):
        return model.tokenize(
            captions, pad_to_native_context=pad_to_native_context
        )
    return model.text_encoder.tokenize(  # type: ignore[attr-defined]
        captions, pad_to_native_context=pad_to_native_context
    )


def _token_shape_and_utilization(tokens: object) -> dict[str, Any]:
    if hasattr(tokens, "get"):
        input_ids = tokens["input_ids"]
        mask = tokens.get("attention_mask")
    else:
        input_ids = tokens
        mask = None
    padded_length = int(input_ids.shape[-1])
    if mask is None:
        lengths = torch.full(
            (int(input_ids.shape[0]),), padded_length, dtype=torch.long
        )
        fixed = True
    else:
        lengths = mask.detach().cpu().sum(dim=-1)
        fixed = bool(torch.all(lengths == padded_length))
    values = lengths.to(torch.float64).numpy()
    return {
        "batch_size": int(input_ids.shape[0]),
        "padded_sequence_length": padded_length,
        "raw_token_length_mean": float(np.mean(values)),
        "raw_token_length_median": float(np.median(values)),
        "raw_token_length_p95": float(np.percentile(values, 95)),
        "raw_token_length_max": int(np.max(values)),
        "mean_padding_tokens_per_caption": float(padded_length - np.mean(values)),
        "token_utilization_fraction": float(np.mean(values) / padded_length),
        "tokenizer_output_fixed_length": fixed,
    }


def _encode_tokens(model: torch.nn.Module, tokens: object) -> torch.Tensor:
    return model.encode_text_tokens(tokens)  # type: ignore[attr-defined]


def _quartiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
    }


def _cuda_times(callable_value, warmup: int, repeats: int, device: torch.device) -> list[float]:
    with torch.inference_mode(), _autocast(device):
        for _ in range(warmup):
            callable_value()
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            callable_value()
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
    return values


def _assert_fully_eval(model: torch.nn.Module, label: str) -> None:
    training = [name or "<root>" for name, module in model.named_modules() if module.training]
    if training:
        raise RuntimeError(
            f"{label} contains {len(training)} modules in training mode: {training[:10]}"
        )


def _cpu_times(callable_value, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        callable_value()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_value()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def _sample_batch(csv_path: Path, batch_size: int) -> tuple[list[Image.Image], list[str]]:
    rows = _read_rows(csv_path)
    selected = rows[:batch_size]
    images = []
    for row in selected:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images, [str(row["caption"]) for row in selected]


def _student_transform(config: dict[str, Any]):
    return build_image_transform(
        image_size=int(config["data"]["image_size"]),
        train=False,
        mean=config["data"].get("image_mean"),
        std=config["data"].get("image_std"),
        interpolation=str(config["data"].get("interpolation", "bicubic")),
    )


def _profile_flops(
    model: torch.nn.Module,
    images: torch.Tensor,
    tokens: object,
    device: torch.device,
) -> dict[str, float]:
    def measured(callable_value) -> float:
        # torch.profiler attributes FLOPs only to a small subset of CUDA
        # kernels.  That produced the plausible-looking but impossible 1,152
        # FLOP value for DINOv3 at both 224px and 192px.  FlopCounterMode
        # counts dispatched PyTorch operators and is the operator-level method
        # previously validated against the 256px DINOv3 count.
        from torch.utils.flop_counter import FlopCounterMode

        with torch.inference_mode(), _autocast(device):
            with FlopCounterMode(display=False) as counter:
                callable_value()
        return float(counter.get_total_flops())

    image_one = images[:1]
    if hasattr(tokens, "items"):
        token_one = type(tokens)({key: value[:1] for key, value in tokens.items()})
    else:
        token_one = tokens[:1]
    image_flops = measured(lambda: model.encode_image(image_one))  # type: ignore[attr-defined]
    text_flops = measured(lambda: _encode_tokens(model, token_one))
    embedding_dim = int(model.encode_image(image_one).shape[-1])  # type: ignore[attr-defined]
    normalization_flops = 3.0 * embedding_dim
    return {
        "image_flops": image_flops + normalization_flops,
        "caption_flops": text_flops + normalization_flops,
        "normalization_flops_added_per_embedding": normalization_flops,
        "multiply_add_flops": 2,
        "flop_source": (
            "torch.utils.flop_counter.FlopCounterMode operator-level counts "
            "plus explicit L2-normalization accounting"
        ),
        "flop_measurement_version": "operator_level_v2",
    }


def _reference_weight_manifest(model: NativePairedModel, reference_id: str) -> dict[str, Any]:
    import open_clip

    config = open_clip.get_pretrained_cfg(model.model_name, model.pretrained)
    cache_dir = os.environ.get("HF_HOME") or None
    value = Path(open_clip.download_pretrained(config, cache_dir=cache_dir))
    paths = sorted(value.rglob("*")) if value.is_dir() else [value]
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]
    aggregate = hashlib.sha256(
        "".join(f"{row['path']}:{row['sha256']}\n" for row in files).encode()
    ).hexdigest()
    return {
        "reference_id": reference_id,
        "model_name": model.model_name,
        "pretrained": model.pretrained,
        "open_clip_version": open_clip.__version__,
        "upstream_revision_recorded_at_fetch": False,
        "files": files,
        "aggregate_sha256": aggregate,
    }


def profile(
    pipeline: dict[str, Any],
    index: int | None,
    *,
    dynamic_padding: bool = False,
) -> dict[str, Any]:
    provenance = _hardware_guard(pipeline)
    job = _select(profiling_jobs(pipeline), index)
    destination = output_root(pipeline) / "per_run" / job["entry_id"] / (
        f"seed_{job['seed']}" if job["seed"] is not None else "reference"
    )
    device = provenance.pop("device")
    if job["kind"] == "student":
        model, config, fingerprint = _load_student(job, device)
        transform = _student_transform(config)
        image_size = int(config["data"]["image_size"])
        parameters = parameter_counts_student(model)
    else:
        model = build_reference(job["entry_id"]).to(device).eval()
        config = None
        fingerprint = None
        transform = model.preprocess
        image_size = int(model.image_size)
        parameters = parameter_counts_reference(model)
    _assert_fully_eval(model, job["entry_id"])
    spec = pipeline["profiling"]
    batch_size = int(spec["paired_batch_size"])
    warmup = int(spec["warmup_iterations"])
    repeats = int(spec["timed_repeats"])
    pil_images, captions = _sample_batch(
        ROOT / str(pipeline["datasets"]["primary"]["csv"]), batch_size
    )
    preprocessing_times = _cpu_times(
        lambda: [transform(image) for image in pil_images], warmup, repeats
    )
    tokenization_times = _cpu_times(
        lambda: _tokenize_native(
            model, captions, pad_to_native_context=not dynamic_padding
        ),
        warmup,
        repeats,
    )
    image_tensor = torch.stack([transform(image) for image in pil_images]).to(device)
    cpu_tokens = _tokenize_native(
        model, captions, pad_to_native_context=not dynamic_padding
    )
    token_stats = _token_shape_and_utilization(cpu_tokens)
    tokens = cpu_tokens.to(device)
    image_times = _cuda_times(
        lambda: model.encode_image(image_tensor), warmup, repeats, device  # type: ignore[attr-defined]
    )
    text_times = _cuda_times(
        lambda: _encode_tokens(model, tokens), warmup, repeats, device
    )
    full_times = _cuda_times(
        lambda: (model.encode_image(image_tensor), _encode_tokens(model, tokens)),  # type: ignore[attr-defined]
        warmup,
        repeats,
        device,
    )
    try:
        flops = {
            **_profile_flops(model, image_tensor, tokens, device),
            "flop_measurement_status": "COMPLETE",
            "flop_measurement_error": None,
        }
    except (AttributeError, RuntimeError) as exc:
        # PyTorch 2.8's FlopCounterMode module tracker can fail inside
        # MultiheadAttention under inference_mode. Timing has already completed
        # and is the primary output of this stage, so record the instrumentation
        # limitation rather than discarding valid latency measurements.
        flops = {
            "image_flops": None,
            "caption_flops": None,
            "normalization_flops_added_per_embedding": None,
            "multiply_add_flops": 2,
            "flop_source": None,
            "flop_measurement_version": "operator_level_v2",
            "flop_measurement_status": "UNAVAILABLE_PROFILER_INCOMPATIBILITY",
            "flop_measurement_error": f"{type(exc).__name__}: {exc}",
        }
    payload = {
        "status": "COMPLETE",
        **job,
        **provenance,
        **parameters,
        **flops,
        "native_image_resolution": image_size,
        "native_text_context_length": _native_context(model),
        "padding_workload": (
            "dynamic_batch" if dynamic_padding else "maximum_context_stress"
        ),
        **token_stats,
        "paired_batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_repeats": repeats,
        "full_stack_neural_latency_ms": _quartiles(full_times),
        "image_neural_latency_ms": _quartiles(image_times),
        "text_neural_latency_ms": _quartiles(text_times),
        "cpu_preprocessing_latency_ms": _quartiles(preprocessing_times),
        "cpu_tokenization_latency_ms": _quartiles(tokenization_times),
        "checkpoint_fingerprint": fingerprint.digest if fingerprint else None,
    }
    profile_name = "profile_dynamic_padding.json" if dynamic_padding else "profile.json"
    atomic_json(payload, destination / profile_name)
    if isinstance(model, NativePairedModel):
        weights = _reference_weight_manifest(model, job["entry_id"])
        atomic_json(
            weights,
            output_root(pipeline)
            / "manifests/reference_weights"
            / f"{job['entry_id']}.json",
        )
    return payload


def profile_mobileclip_fusion(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Paired deployment-graph control; never overwrites the frontier profile."""
    from timm.utils import reparameterize_model

    provenance = _hardware_guard(pipeline)
    device = provenance.pop("device")
    entry_id = "mobileclip2_s0_dfndr2b"
    model = build_reference(entry_id).to(device).eval()
    _assert_fully_eval(model, f"{entry_id}:unfused")
    trunk = model.model.visual.trunk
    spec = pipeline["profiling"]
    batch_size = int(spec["paired_batch_size"])
    warmup = int(spec["warmup_iterations"])
    repeats = int(spec["timed_repeats"])
    pil_images, captions = _sample_batch(
        ROOT / str(pipeline["datasets"]["primary"]["csv"]), batch_size
    )
    images = torch.stack([model.preprocess(image) for image in pil_images]).to(device)
    tokens = model.tokenize(captions).to(device)

    def state(root: torch.nn.Module) -> dict[str, int]:
        modules = list(root.modules())
        return {
            "mobileone_blocks": sum(
                type(module).__name__ == "MobileOneBlock" for module in modules
            ),
            "mobileone_fused": sum(
                type(module).__name__ == "MobileOneBlock"
                and getattr(module, "reparam_conv", None) is not None
                for module in modules
            ),
            "large_kernel_blocks": sum(
                type(module).__name__ == "ReparamLargeKernelConv"
                for module in modules
            ),
            "large_kernel_fused": sum(
                type(module).__name__ == "ReparamLargeKernelConv"
                and getattr(module, "reparam_conv", None) is not None
                for module in modules
            ),
        }

    before_state = state(trunk)
    before_flops = _profile_flops(model, images, tokens, device)
    with torch.inference_mode(), _autocast(device):
        before_output = model.encode_image(images).detach().float()
    before_image = _cuda_times(
        lambda: model.encode_image(images), warmup, repeats, device
    )
    before_full = _cuda_times(
        lambda: (model.encode_image(images), model.encode_text_tokens(tokens)),
        warmup,
        repeats,
        device,
    )

    reparameterize_model(trunk, inplace=True)
    model.eval()
    _assert_fully_eval(model, f"{entry_id}:fused")
    after_state = state(trunk)
    after_flops = _profile_flops(model, images, tokens, device)
    with torch.inference_mode(), _autocast(device):
        after_output = model.encode_image(images).detach().float()
    after_image = _cuda_times(
        lambda: model.encode_image(images), warmup, repeats, device
    )
    after_full = _cuda_times(
        lambda: (model.encode_image(images), model.encode_text_tokens(tokens)),
        warmup,
        repeats,
        device,
    )
    difference = (before_output - after_output).abs()
    cosine = torch.nn.functional.cosine_similarity(
        before_output, after_output, dim=-1
    )
    payload = {
        "status": "COMPLETE_PENDING_EQUIVALENCE_REVIEW",
        "entry_id": entry_id,
        **provenance,
        "precision": "native_bf16",
        "paired_batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_repeats": repeats,
        "image_resolution": int(model.image_size),
        "image_flops": after_flops["image_flops"],
        "caption_flops": after_flops["caption_flops"],
        "flop_source": after_flops["flop_source"],
        "flop_measurement_version": after_flops["flop_measurement_version"],
        "architecture": "FastViT fastvit_mci0 convolutional/hybrid",
        "before_reparameterization": {
            "graph_state": before_state,
            "flops": before_flops,
            "image_latency_ms": _quartiles(before_image),
            "full_stack_latency_ms": _quartiles(before_full),
        },
        "after_reparameterization": {
            "graph_state": after_state,
            "flops": after_flops,
            "image_latency_ms": _quartiles(after_image),
            "full_stack_latency_ms": _quartiles(after_full),
        },
        "numerical_equivalence": {
            "max_absolute_embedding_difference": float(difference.max().cpu()),
            "mean_absolute_embedding_difference": float(difference.mean().cpu()),
            "minimum_cosine_similarity": float(cosine.min().cpu()),
            "mean_cosine_similarity": float(cosine.mean().cpu()),
            "threshold_pre_registered": False,
            "note": "Report measurements for review; do not silently promote the fused result.",
        },
        "protocol_notes": [
            "The paired before/after measurements use identical weights, inputs, hardware, precision and allocation.",
            "OpenCLIP uses its native 224x224 resolution; MobileCLIP2 and the DINOv3 students use native/configured 256x256 inputs.",
        ],
    }
    atomic_json(
        payload,
        output_root(pipeline) / "controls/mobileclip_fusion_profile.json",
    )
    return payload


def evaluate_mobileclip_fusion(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the exact reparameterized graph used by the latency control."""
    from timm.utils import reparameterize_model

    provenance = _hardware_guard(pipeline)
    device = provenance.pop("device")
    entry_id = "mobileclip2_s0_dfndr2b"
    model = build_reference(entry_id).to(device).eval()
    parameters_before = parameter_counts_reference(model)
    reparameterize_model(model.model.visual.trunk, inplace=True)
    model.eval()
    _assert_fully_eval(model, f"{entry_id}:fused-retrieval")
    parameters_after = parameter_counts_reference(model)
    root = output_root(pipeline)
    original = json.loads(
        (
            root
            / "per_run"
            / entry_id
            / "reference"
            / "evaluation.json"
        ).read_text()
    )
    result: dict[str, Any] = {
        "status": "COMPLETE_PENDING_FRONTIER_PROMOTION",
        "entry_id": entry_id,
        **provenance,
        "deployment_graph": "official timm FastViT reparameterization",
        "parameters_before_fusion": parameters_before,
        "parameters_after_fusion": parameters_after,
        "datasets": {},
    }
    for dataset_name in ("primary", "supplementary"):
        spec = pipeline["datasets"][dataset_name]
        dataset_id = str(spec["id"])
        csv_path = ROOT / str(spec["csv"])
        loader = native_loader(
            model,
            csv_path,
            image_root=ROOT,
            batch_size=int(pipeline["evaluation"]["batch_size"]),
            num_workers=int(pipeline["resources"]["cpus_per_task"]),
        )
        with _autocast(device):
            embeddings = extract_embeddings(model, loader, device)
        fused = _metrics_from_embeddings(
            embeddings, list(pipeline["evaluation"]["k_values"])
        )
        unfused = original[dataset_id]
        comparison = {
            "dataset_id": dataset_id,
            "dataset_role": dataset_name,
            "dataset_sha256": sha256_file(csv_path),
            "unfused": unfused,
            "fused": fused,
            "delta_pp": {
                key: (float(fused[key]) - float(unfused[key])) * 100.0
                for key in ("i2t_R@1", "t2i_R@1", "mean_R@1")
            },
        }
        result["datasets"][dataset_id] = comparison
        atomic_json(
            comparison,
            root
            / "controls"
            / f"mobileclip_fusion_{dataset_id}_metrics.json",
        )
    latency_path = root / "controls/mobileclip_fusion_profile.json"
    if not latency_path.is_file():
        raise FileNotFoundError(
            "paired MobileCLIP fusion profile is required before retrieval comparison"
        )
    result["latency_control"] = json.loads(latency_path.read_text())
    atomic_json(result, root / "controls/mobileclip_fusion_evaluation.json")
    return result


def dynamic_padding_report(pipeline: dict[str, Any]) -> dict[str, Any]:
    root = output_root(pipeline)
    original = pd.read_csv(root / "frontier.csv")
    aggregate_rows = original[original["row_type"] == "aggregate"].copy()
    records = []
    for job in profiling_jobs(pipeline):
        folder = root / "per_run" / job["entry_id"] / (
            f"seed_{job['seed']}" if job["seed"] is not None else "reference"
        )
        payload = json.loads((folder / "profile_dynamic_padding.json").read_text())
        records.append(
            {
                "entry_id": job["entry_id"],
                "seed": job["seed"],
                "padding_workload": payload["padding_workload"],
                "native_text_context_length": payload["native_text_context_length"],
                "padded_sequence_length": payload["padded_sequence_length"],
                "raw_token_length_mean": payload["raw_token_length_mean"],
                "raw_token_length_median": payload["raw_token_length_median"],
                "raw_token_length_p95": payload["raw_token_length_p95"],
                "raw_token_length_max": payload["raw_token_length_max"],
                "mean_padding_tokens_per_caption": payload[
                    "mean_padding_tokens_per_caption"
                ],
                "token_utilization_fraction": payload["token_utilization_fraction"],
                "tokenizer_output_fixed_length": payload[
                    "tokenizer_output_fixed_length"
                ],
                "text_latency_median_ms": payload["text_neural_latency_ms"]["median"],
                "full_latency_median_ms": payload[
                    "full_stack_neural_latency_ms"
                ]["median"],
                "full_latency_q1_ms": payload["full_stack_neural_latency_ms"]["q1"],
                "full_latency_q3_ms": payload["full_stack_neural_latency_ms"]["q3"],
                "node": payload["node"],
                "gpu_model": payload["gpu_model"],
            }
        )
    profiles = pd.DataFrame(records)
    representative = profiles[
        profiles["seed"].isna() | (profiles["seed"] == 42)
    ].copy()
    for column in (
        "padded_sequence_length",
        "raw_token_length_mean",
        "raw_token_length_median",
        "raw_token_length_p95",
        "raw_token_length_max",
        "mean_padding_tokens_per_caption",
        "token_utilization_fraction",
        "tokenizer_output_fixed_length",
        "text_latency_median_ms",
        "full_latency_median_ms",
        "full_latency_q1_ms",
        "full_latency_q3_ms",
    ):
        mapping = dict(zip(representative["entry_id"], representative[column]))
        aggregate_rows[f"dynamic_{column}"] = aggregate_rows["entry_id"].map(mapping)
    atomic_csv(profiles, root / "tables/dynamic_padding_profiles.csv")
    atomic_csv(aggregate_rows, root / "frontier_dynamic_padding.csv")
    report = {
        "status": "COMPLETE",
        "scope": "profiling-only; training and retrieval accuracy are unchanged",
        "methodological_finding": (
            "Maximum supported context and typical deployed length diverge across "
            "fixed-length CLIP tokenizers and dynamically padded sentence encoders."
        ),
        "conclusion_deferred_until_measurement": True,
        "rows": records,
    }
    atomic_json(report, root / "dynamic_padding_report.json")
    return report


def _evaluation_payload(
    root: Path, entry_id: str, seed: int | None, dataset_id: str
) -> dict[str, Any]:
    folder = root / "per_run" / entry_id / (
        f"seed_{seed}" if seed is not None else "reference"
    )
    return json.loads((folder / f"{dataset_id}_metrics.json").read_text())


def _profile_payload(root: Path, entry_id: str, seed: int | None) -> dict[str, Any]:
    folder = root / "per_run" / entry_id / (
        f"seed_{seed}" if seed is not None else "reference"
    )
    return json.loads((folder / "profile.json").read_text())


def parameter_pareto(frame: pd.DataFrame, performance: str) -> set[str]:
    frontier = set()
    for _, row in frame.iterrows():
        dominated = False
        for _, other in frame.iterrows():
            if other["entry_id"] == row["entry_id"]:
                continue
            no_more_cost = (
                other["full_stack_inference_parameters"]
                <= row["full_stack_inference_parameters"]
            )
            no_less_performance = other[performance] >= row[performance]
            strict = (
                other["full_stack_inference_parameters"]
                < row["full_stack_inference_parameters"]
                or other[performance] > row[performance]
            )
            if no_more_cost and no_less_performance and strict:
                dominated = True
                break
        if not dominated:
            frontier.add(str(row["entry_id"]))
    return frontier


def latency_pareto(
    frame: pd.DataFrame, performance: str, *, iqr_aware: bool
) -> set[str]:
    frontier = set()
    for _, row in frame.iterrows():
        dominated = False
        for _, other in frame.iterrows():
            if other["entry_id"] == row["entry_id"]:
                continue
            if iqr_aware:
                strictly_faster = other["latency_q3_ms"] < row["latency_q1_ms"]
                overlaps = not (
                    other["latency_q3_ms"] < row["latency_q1_ms"]
                    or row["latency_q3_ms"] < other["latency_q1_ms"]
                )
                better_cost = strictly_faster or (
                    overlaps and other[performance] > row[performance]
                )
            else:
                better_cost = other["latency_median_ms"] <= row["latency_median_ms"]
            no_less_performance = other[performance] >= row[performance]
            strict = (
                other[performance] > row[performance]
                or (
                    not iqr_aware
                    and other["latency_median_ms"] < row["latency_median_ms"]
                )
                or (iqr_aware and other["latency_q3_ms"] < row["latency_q1_ms"])
            )
            if better_cost and no_less_performance and strict:
                dominated = True
                break
        if not dominated:
            frontier.add(str(row["entry_id"]))
    return frontier


def _aggregate_profiles(root: Path, entry_id: str, is_reference: bool) -> dict[str, Any]:
    representative = _profile_payload(root, entry_id, None if is_reference else 42)
    result = {
        "full_stack_inference_parameters": representative[
            "full_stack_inference_parameters"
        ],
        "query_side_inference_parameters": representative[
            "query_side_inference_parameters"
        ],
        "image_flops": representative["image_flops"],
        "caption_flops": representative["caption_flops"],
        "native_image_resolution": representative["native_image_resolution"],
        "native_text_context_length": representative["native_text_context_length"],
        "latency_median_ms": representative["full_stack_neural_latency_ms"]["median"],
        "latency_q1_ms": representative["full_stack_neural_latency_ms"]["q1"],
        "latency_q3_ms": representative["full_stack_neural_latency_ms"]["q3"],
        "image_latency_median_ms": representative["image_neural_latency_ms"]["median"],
        "text_latency_median_ms": representative["text_neural_latency_ms"]["median"],
        "cpu_preprocessing_median_ms": representative[
            "cpu_preprocessing_latency_ms"
        ]["median"],
        "cpu_tokenization_median_ms": representative[
            "cpu_tokenization_latency_ms"
        ]["median"],
        "latency_node": representative["node"],
        "latency_gpu": representative["gpu_model"],
    }
    if not is_reference:
        verification = _profile_payload(root, entry_id, 43)
        result["seed43_latency_median_ms"] = verification[
            "full_stack_neural_latency_ms"
        ]["median"]
        result["seed43_minus_seed42_latency_ms"] = (
            result["seed43_latency_median_ms"] - result["latency_median_ms"]
        )
    return result


def aggregate(pipeline: dict[str, Any]) -> dict[str, Any]:
    root = output_root(pipeline)
    weight_manifests = []
    for entry in pipeline["references"]["entries"]:
        path = root / "manifests/reference_weights" / f"{entry['id']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing prospective reference checksum: {path}")
        weight_manifests.append(json.loads(path.read_text()))
    atomic_json(
        {
            "status": "COMPLETE",
            "open_clip_version": pipeline["references"]["open_clip_version"],
            "upstream_revisions_recorded_at_fetch": False,
            "entries": weight_manifests,
        },
        root / "manifests/reference_weights.json",
    )
    primary_id = str(pipeline["datasets"]["primary"]["id"])
    supplementary_id = str(pipeline["datasets"]["supplementary"]["id"])
    rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for entry in pipeline["students"]:
        seed_rows = []
        for seed in (42, 43, 44):
            primary = _evaluation_payload(root, entry["id"], seed, primary_id)
            supplementary = _evaluation_payload(root, entry["id"], seed, supplementary_id)
            row = {
                "entry_id": entry["id"],
                "label": entry["label"],
                "role": entry["role"],
                "row_type": "seed",
                "seed": seed,
                "flickr_i2t_R1": primary["i2t_R@1"],
                "flickr_t2i_R1": primary["t2i_R@1"],
                "flickr_mean_R1": primary["mean_R@1"],
                "coco_validation_i2t_R1": supplementary["i2t_R@1"],
                "coco_validation_t2i_R1": supplementary["t2i_R@1"],
                "coco_validation_mean_R1": supplementary["mean_R@1"],
            }
            rows.append(row)
            seed_rows.append(row)
        frame = pd.DataFrame(seed_rows)
        aggregate_row = {
            "entry_id": entry["id"],
            "label": entry["label"],
            "role": entry["role"],
            "row_type": "aggregate",
            "seed": None,
        }
        for prefix in ("flickr", "coco_validation"):
            for direction in ("i2t", "t2i", "mean"):
                column = f"{prefix}_{direction}_R1"
                aggregate_row[column] = float(frame[column].mean())
                if direction == "mean":
                    aggregate_row[f"{column}_sd"] = float(frame[column].std(ddof=1))
                    aggregate_row[f"{column}_min"] = float(frame[column].min())
                    aggregate_row[f"{column}_max"] = float(frame[column].max())
        aggregate_row.update(_aggregate_profiles(root, entry["id"], False))
        aggregate_rows.append(aggregate_row)
        rows.append(aggregate_row)
    for entry in pipeline["references"]["entries"]:
        primary = _evaluation_payload(root, entry["id"], None, primary_id)
        supplementary = _evaluation_payload(root, entry["id"], None, supplementary_id)
        row = {
            "entry_id": entry["id"],
            "label": entry["checkpoint_id"],
            "role": "reference",
            "row_type": "aggregate",
            "seed": None,
            "flickr_i2t_R1": primary["i2t_R@1"],
            "flickr_t2i_R1": primary["t2i_R@1"],
            "flickr_mean_R1": primary["mean_R@1"],
            "flickr_mean_R1_sd": None,
            "flickr_mean_R1_min": primary["mean_R@1"],
            "flickr_mean_R1_max": primary["mean_R@1"],
            "coco_validation_i2t_R1": supplementary["i2t_R@1"],
            "coco_validation_t2i_R1": supplementary["t2i_R@1"],
            "coco_validation_mean_R1": supplementary["mean_R@1"],
            "coco_validation_mean_R1_sd": None,
            "coco_validation_mean_R1_min": supplementary["mean_R@1"],
            "coco_validation_mean_R1_max": supplementary["mean_R@1"],
            **_aggregate_profiles(root, entry["id"], True),
        }
        aggregate_rows.append(row)
        rows.append(row)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    parameter_frontier = parameter_pareto(aggregate_frame, "flickr_mean_R1")
    latency_frontier = latency_pareto(
        aggregate_frame, "flickr_mean_R1", iqr_aware=True
    )
    raw_frontier = latency_pareto(
        aggregate_frame, "flickr_mean_R1", iqr_aware=False
    )
    for row in rows:
        if row["row_type"] != "aggregate":
            continue
        row["flickr_parameter_pareto"] = row["entry_id"] in parameter_frontier
        row["flickr_latency_iqr_pareto"] = row["entry_id"] in latency_frontier
        row["flickr_latency_raw_median_pareto"] = row["entry_id"] in raw_frontier
        row["flickr_R1_per_million_parameters"] = (
            row["flickr_mean_R1"]
            / (row["full_stack_inference_parameters"] / 1_000_000)
        )
        row["flickr_R1_per_ms"] = (
            row["flickr_mean_R1"] / row["latency_median_ms"]
        )
    frontier = pd.DataFrame(rows)
    atomic_csv(frontier, root / "frontier.csv")
    _write_plots(frontier[frontier["row_type"] == "aggregate"], root)
    _write_frontier_reports(
        pipeline,
        frontier[frontier["row_type"] == "aggregate"],
        parameter_frontier,
        latency_frontier,
        raw_frontier,
    )
    return {
        "status": "COMPLETE",
        "parameter_frontier": sorted(parameter_frontier),
        "latency_iqr_frontier": sorted(latency_frontier),
        "latency_raw_median_frontier": sorted(raw_frontier),
    }


def _write_plots(frame: pd.DataFrame, root: Path) -> None:
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for prefix, title in (
        ("flickr", "Flickr30k Karpathy test (primary)"),
        ("coco_validation", "COCO validation (contaminated, supplementary)"),
    ):
        performance = f"{prefix}_mean_R1"
        for cost, label, filename in (
            (
                "full_stack_inference_parameters",
                "Full-stack inference parameters",
                f"{prefix}_r1_vs_parameters.png",
            ),
            (
                "latency_median_ms",
                "Paired-batch neural latency median (ms)",
                f"{prefix}_r1_vs_latency.png",
            ),
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(frame[cost], frame[performance] * 100)
            for _, row in frame.iterrows():
                ax.annotate(str(row["entry_id"]), (row[cost], row[performance] * 100))
            ax.set_xlabel(label)
            ax.set_ylabel("Mean bidirectional R@1 (%)")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=180)
            plt.close(fig)


def _write_frontier_reports(
    pipeline: dict[str, Any],
    frame: pd.DataFrame,
    parameter_frontier: set[str],
    latency_frontier: set[str],
    raw_frontier: set[str],
) -> None:
    root = output_root(pipeline)
    status = frame[
        [
            "entry_id",
            "label",
            "flickr_mean_R1",
            "full_stack_inference_parameters",
            "latency_median_ms",
            "latency_q1_ms",
            "latency_q3_ms",
        ]
    ].copy()
    status["parameter_pareto"] = status["entry_id"].isin(parameter_frontier)
    status["latency_iqr_pareto"] = status["entry_id"].isin(latency_frontier)
    status["latency_raw_median_pareto"] = status["entry_id"].isin(raw_frontier)
    atomic_csv(status, root / "tables/pareto_status.csv")
    atomic_csv(
        status[
            ~(
                status["parameter_pareto"]
                | status["latency_iqr_pareto"]
                | status["latency_raw_median_pareto"]
            )
        ],
        root / "tables/dominated_models.csv",
    )
    correction = frame[frame["entry_id"].isin(["historical_v4", "locked_minilm"])]
    atomic_csv(correction, root / "tables/recipe_correction.csv")
    report = {
        "status": "COMPLETE",
        "primary_dataset": "Flickr30k Karpathy test",
        "supplementary_dataset": (
            "COCO validation (used for model selection in earlier project phases; "
            "not an independent test set)"
        ),
        "untouched_coco_test_available": False,
        "parameter_frontier": sorted(parameter_frontier),
        "latency_iqr_frontier": sorted(latency_frontier),
        "latency_raw_median_frontier": sorted(raw_frontier),
        "latency_frontier_difference": sorted(latency_frontier ^ raw_frontier),
        "flop_convention": (
            "One multiply-add is two FLOPs. Projection and normalization are "
            "included. Caption FLOPs use each model's native context and are not "
            "directly comparable across differing context lengths."
        ),
        "limitations": pipeline["limitations"],
    }
    atomic_json(report, root / "report.json")
    text = f"""# Efficiency frontier

Primary dataset: Flickr30k Karpathy test. COCO is supplementary and contaminated
by earlier project selection. The project has no untouched COCO test split.

Parameter Pareto frontier: {", ".join(sorted(parameter_frontier))}

IQR-aware latency frontier: {", ".join(sorted(latency_frontier))}

Raw-median latency frontier: {", ".join(sorted(raw_frontier))}

Caption FLOPs use each model's native context length and are therefore not
directly comparable without qualification. One multiply-add is counted as two
FLOPs; the alternative convention halves every reported FLOP value. COCO
captions average approximately 11 tokens, so padding policy affects FLOP
accounting more strongly than typical caption realism.

The students were selected on development data and are reported on Flickr30k
for the first time here. Upstream reference revisions were not recorded at
fetch time; prospective local checksums are supplied instead.
"""
    atomic_text(text, root / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "manifest",
            "train-missing-v4",
            "evaluate",
            "profile",
            "profile-dynamic",
            "profile-mobileclip-fusion",
            "evaluate-mobileclip-fusion",
            "report-dynamic",
            "aggregate",
        ),
    )
    parser.add_argument("--pipeline", default=PIPELINE_DEFAULT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    pipeline = load_pipeline(args.pipeline)
    if args.command == "validate":
        result = validate(pipeline)
    elif args.command == "manifest":
        result = create_dataset_manifests(pipeline)
    elif args.command == "train-missing-v4":
        result = train_missing_v4(pipeline, args.index, resume=not args.no_resume)
    elif args.command == "evaluate":
        result = evaluate(pipeline, args.index)
    elif args.command == "profile":
        result = profile(pipeline, args.index)
    elif args.command == "profile-dynamic":
        result = profile(pipeline, args.index, dynamic_padding=True)
    elif args.command == "profile-mobileclip-fusion":
        result = profile_mobileclip_fusion(pipeline)
    elif args.command == "evaluate-mobileclip-fusion":
        result = evaluate_mobileclip_fusion(pipeline)
    elif args.command == "report-dynamic":
        result = dynamic_padding_report(pipeline)
    else:
        result = aggregate(pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
