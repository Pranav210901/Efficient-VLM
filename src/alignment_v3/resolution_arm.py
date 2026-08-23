from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _hardware_guard,
    _load_student,
    _metrics_from_embeddings,
    _student_loader,
    profile as frontier_profile,
)
from src.phase15.io_utils import atomic_csv, atomic_json
from src.alignment_v3.runner import (
    ROOT,
    checkpoint_root,
    load_pipeline,
    resolution_jobs,
)
from src.training.evaluate import extract_embeddings
from src.utils.config import load_config


def _select(values: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    selected = (
        int(os.environ["SLURM_ARRAY_TASK_ID"])
        if index is None and "SLURM_ARRAY_TASK_ID" in os.environ
        else 0
        if index is None and len(values) == 1
        else None
        if index is None
        else int(index)
    )
    if selected is None:
        raise ValueError(
            "--index is required outside a Slurm array when multiple jobs exist"
        )
    if selected < 0 or selected >= len(values):
        raise IndexError(f"index {selected} outside manifest of size {len(values)}")
    return values[selected]


def _checkpoint_job(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "student",
        "entry_id": str(job["experiment_id"]),
        "label": f"DINOv3 ViT-S/16 at {job['image_size']}px",
        "role": "resolution_arm",
        "seed": int(job["seed"]),
        "checkpoint_dir": str(
            (
                checkpoint_root(pipeline)
                / "resolution"
                / str(job["run_id"])
            ).relative_to(ROOT)
        ),
    }


def _guard(job: dict[str, Any], config: dict[str, Any], model: Any | None = None) -> dict[str, int]:
    expected = int(job["image_size"])
    saved = int(config["data"]["image_size"])
    guard = config.get("provenance", {}).get("resolution_guard", {})
    patch = (expected // 16) ** 2
    total = patch + 5
    checks = {
        "saved_config": saved,
        "guard_config": int(guard.get("configured_image_size", -1)),
        "patch_tokens": int(guard.get("patch_tokens", -1)),
        "total_tokens": int(guard.get("total_transformer_tokens", -1)),
    }
    if checks != {
        "saved_config": expected,
        "guard_config": expected,
        "patch_tokens": patch,
        "total_tokens": total,
    }:
        raise RuntimeError(f"resolution identity mismatch: expected={expected}, observed={checks}")
    if model is not None:
        vision = model.vision_encoder.model
        patch_size = getattr(vision.patch_embed, "patch_size", (16, 16))
        if tuple(patch_size) != (16, 16):
            raise RuntimeError(f"loaded patch size mismatch: {patch_size}")
        if int(getattr(vision, "num_prefix_tokens", -1)) != 5:
            raise RuntimeError(
                f"loaded prefix-token mismatch: {getattr(vision, 'num_prefix_tokens', None)}"
            )
        projector_input = int(model.image_projection.skip.in_features)
        if projector_input != int(model.vision_encoder.output_dim):
            raise RuntimeError(
                "projector input dimension changed with token count: "
                f"{projector_input} != {model.vision_encoder.output_dim}"
            )
    return {
        "image_resolution": expected,
        "patch_size": 16,
        "patch_tokens": patch,
        "prefix_tokens": 5,
        "total_transformer_tokens": total,
    }


def validate(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows = []
    for job in resolution_jobs(pipeline):
        from src.alignment_v3.runner import build_job_config

        config = build_job_config(pipeline, job, "resolution")
        identity = _guard(job, config)
        if int(config["training"]["batch_size"]) != 1024:
            raise RuntimeError("resolution arm batch size drift")
        if int(config["training"]["memory_queue_size"]) != 0:
            raise RuntimeError("resolution arm queue must be disabled")
        if config["data"]["train_captions_per_image"] is not None:
            raise RuntimeError("resolution arm must use all captions")
        rows.append({**job, **identity, "resolved_lr": float(config["training"]["lr"])})
    atomic_csv(pd.DataFrame(rows), ROOT / "results/resolution_arm/manifests/jobs_verified.csv")
    return {"status": "READY", "jobs": len(rows), "rows": rows}


def evaluate_validation(pipeline_path: str, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(resolution_jobs(pipeline), index)
    checkpoint_job = _checkpoint_job(pipeline, job)
    provenance = _hardware_guard(
        {
            "resources": {},
        }
    )
    device = provenance.pop("device")
    model, config, fingerprint = _load_student(checkpoint_job, device)
    identity = _guard(job, config, model)
    csv_path = ROOT / "data/flickr30k/validation.csv"
    loader = _student_loader(config, csv_path, 256)
    # Inspect the actual evaluation transform rather than assuming the dataset
    # exposes a convenience image_size attribute. Both Resize and CenterCrop
    # must resolve to the saved experiment coordinate.
    target_sizes: list[int] = []
    for transform in getattr(loader.dataset.transform, "transforms", []):
        if type(transform).__name__ not in {"Resize", "CenterCrop"}:
            continue
        value = transform.size
        if isinstance(value, (tuple, list)):
            if len(value) != 2 or int(value[0]) != int(value[1]):
                raise RuntimeError(f"non-square evaluation transform size: {value}")
            target_sizes.append(int(value[0]))
        else:
            target_sizes.append(int(value))
    expected_size = int(job["image_size"])
    if len(target_sizes) != 2 or any(value != expected_size for value in target_sizes):
        raise RuntimeError(
            "dataloader resolution mismatch: "
            f"transform_sizes={target_sizes}, expected={expected_size}"
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, [1, 5, 10])
    payload = {
        "status": "COMPLETE",
        **job,
        **identity,
        **metrics,
        **provenance,
        "dataset": "Flickr30k Karpathy validation",
        "test_split_used": False,
        "checkpoint_fingerprint": fingerprint.digest,
        "dataloader_image_size": expected_size,
        "dataloader_transform_sizes": target_sizes,
    }
    destination = ROOT / "results/resolution_arm/validation" / str(job["run_id"])
    atomic_json(payload, destination / "metrics.json")
    return payload


def _profile_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        job
        for job in resolution_jobs(pipeline)
        if int(job["seed"]) in {42, 43}
    ]


def profile(pipeline_path: str, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(_profile_jobs(pipeline), index)
    checkpoint_job = _checkpoint_job(pipeline, job)
    profile_pipeline = {
        "output_root": "results/resolution_arm/profiling",
        "students": [
            {
                "id": checkpoint_job["entry_id"],
                "label": checkpoint_job["label"],
                "role": checkpoint_job["role"],
                "seeds": {
                    str(job["seed"]): {
                        "checkpoint_dir": checkpoint_job["checkpoint_dir"]
                    }
                },
            }
        ],
        "references": {"entries": []},
        "datasets": {
            "primary": {
                "csv": "data/flickr30k/validation.csv",
            }
        },
        "profiling": {
            "representative_seed": int(job["seed"]),
            "verification_seed": int(job["seed"]),
            "paired_batch_size": 64,
            "warmup_iterations": 20,
            "timed_repeats": 100,
        },
        "resources": {"cpus_per_task": 12},
    }
    payload = frontier_profile(profile_pipeline, 0, dynamic_padding=True)
    observed = int(payload["native_image_resolution"])
    if observed != int(job["image_size"]):
        raise RuntimeError(
            f"profiler/FLOP resolution mismatch: {observed} != {job['image_size']}"
        )
    payload.update(_guard(job, load_config(ROOT / checkpoint_job["checkpoint_dir"] / "config.yaml")))
    atomic_json(
        payload,
        ROOT
        / "results/resolution_arm/profiling/per_run"
        / str(job["experiment_id"])
        / f"seed_{job['seed']}"
        / "profile_dynamic_padding.json",
    )
    return payload


def report(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    validation_rows = []
    for job in resolution_jobs(pipeline):
        path = ROOT / "results/resolution_arm/validation" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing validation result {path}")
        validation_rows.append(json.loads(path.read_text()))
    profile_rows = []
    for job in _profile_jobs(pipeline):
        path = (
            ROOT
            / "results/resolution_arm/profiling/per_run"
            / job["experiment_id"]
            / f"seed_{job['seed']}"
            / "profile_dynamic_padding.json"
        )
        if not path.is_file():
            raise RuntimeError(f"missing profile {path}")
        profile_rows.append(json.loads(path.read_text()))
    validation = pd.DataFrame(validation_rows)
    profiles = pd.DataFrame(profile_rows)
    rows = []
    for resolution in (224, 192):
        values = validation[validation["image_size"] == resolution]["mean_R@1"]
        representative = profiles[
            (profiles["image_resolution"] == resolution)
            & (profiles["seed"] == 42)
        ].iloc[0]
        verification = profiles[
            (profiles["image_resolution"] == resolution)
            & (profiles["seed"] == 43)
        ].iloc[0]
        image_latency = representative["image_neural_latency_ms"]
        full_latency = representative["full_stack_neural_latency_ms"]
        rows.append(
            {
                "resolution": resolution,
                "patch_tokens": int(representative["patch_tokens"]),
                "total_transformer_tokens": int(
                    representative["total_transformer_tokens"]
                ),
                "flickr_validation_mean_R1": float(values.mean()),
                "flickr_validation_sd_R1": float(values.std(ddof=1)),
                "flickr_validation_min_R1": float(values.min()),
                "flickr_validation_max_R1": float(values.max()),
                "image_latency_median_ms": float(image_latency["median"]),
                "full_stack_latency_median_ms": float(full_latency["median"]),
                "image_flops": float(representative["image_flops"]),
                "seed43_image_latency_median_ms": float(
                    verification["image_neural_latency_ms"]["median"]
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("resolution", ascending=False)
    destination = ROOT / "results/resolution_arm/report"
    atomic_csv(table, destination / "resolution_curve.csv")
    result = {
        "status": "COMPLETE",
        "test_split_used": False,
        "selected_resolution": 224,
        "selection_rule_path": "configs/resolution_arm/selection.yaml",
        "selection_applied_before_test_split_evaluation": True,
        "scientific_attribution": (
            "DINOv3's native dynamic-resolution rotary positional encoding was "
            "used without modifying or retraining the frozen backbone. Accuracy "
            "changes therefore measure reduced spatial detail without a learned "
            "absolute-position interpolation mismatch."
        ),
        "rows": rows,
    }
    atomic_json(result, destination / "report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "evaluate", "profile", "report"))
    parser.add_argument("--pipeline", default="configs/resolution_arm/pipeline.yaml")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.pipeline)
    elif args.command == "evaluate":
        result = evaluate_validation(args.pipeline, args.index)
    elif args.command == "profile":
        result = profile(args.pipeline, args.index)
    else:
        result = report(args.pipeline)
    print(result)


if __name__ == "__main__":
    main()
