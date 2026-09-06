"""Preregistered closure of the residual fresh-queue mechanism.

Stage 1 completes the fresh-capacity curve with 1,024 and 4,096 entries.
Stage 2 separates semantically suspicious negatives from mere count reduction
and tests whether queue denominator mass is over-weighted.  All arms preserve
the completed queue-identification recipe and reuse its controls.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.fingerprint import read_fingerprint, sha256_file, write_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.queue_factorial import (
    ROOT,
    build_config as build_factorial_config,
    hardware_guard,
    load_pipeline,
    prediction_gate,
)
from src.alignment_v3.queue_identification import WAVE0_CONTROLS
from src.alignment_v3.training import load_training_checkpoint, train
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.training.evaluate import extract_embeddings
from src.utils.config import deep_update
from src.utils.device import get_device

DEFAULT_PIPELINE = "configs/queue_mechanism/pipeline.yaml"


def mechanism_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    capacity = pipeline["capacity"]
    batch = int(capacity["batch_size"])
    for queue_size in capacity["new_queue_sizes"]:
        for seed in capacity["seeds"]:
            jobs.append(
                {
                    "index": len(jobs),
                    "stage": "capacity",
                    "arm": "fresh_capacity",
                    "run_id": f"b{batch}__queue_{int(queue_size)}__both_fresh__seed_{int(seed)}",
                    "experiment_id": "queue_mechanism_capacity",
                    "batch_size": batch,
                    "target_age_steps": float(queue_size) / batch,
                    "memory_queue_size": int(queue_size),
                    "queue_mode": "both_fresh",
                    "queue_filter_mode": "none",
                    "queue_weight_mode": "none",
                    "seed": int(seed),
                    "memberships": ["fresh_capacity_curve"],
                }
            )
    spec = pipeline["mechanism"]
    batch = int(spec["batch_size"])
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            arm_id = str(arm["id"])
            jobs.append(
                {
                    "index": len(jobs),
                    "stage": "mechanism",
                    "arm": arm_id,
                    "run_id": f"b{batch}__queue_{int(spec['queue_size'])}__{arm_id}__seed_{int(seed)}",
                    "experiment_id": "queue_mechanism_fixed_capacity",
                    "batch_size": batch,
                    "target_age_steps": float(spec["queue_size"]) / batch,
                    "memory_queue_size": int(spec["queue_size"]),
                    "queue_mode": "both_fresh",
                    "queue_filter_mode": str(arm["queue_filter_mode"]),
                    "queue_weight_mode": str(arm["queue_weight_mode"]),
                    "seed": int(seed),
                    "memberships": ["fixed_capacity_mechanism"],
                }
            )
    if len(jobs) != 15:
        raise AssertionError(f"expected 15 new queue-mechanism runs, got {len(jobs)}")
    return jobs


def _select(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    if index is None:
        raise SystemExit("--index is required")
    jobs = mechanism_jobs(pipeline)
    if not 0 <= index < len(jobs):
        raise SystemExit(f"--index must be in [0,{len(jobs) - 1}]")
    return jobs[index]


def build_config(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    config = build_factorial_config(pipeline, job)
    guard = pipeline["mechanism"]["semantic_guard"]
    config = deep_update(
        config,
        {
            "training": {
                "queue_filter_mode": job["queue_filter_mode"],
                "queue_weight_mode": job["queue_weight_mode"],
                "queue_filter_threshold": float(guard["image_text_threshold"]),
                "queue_semantic_cache": str(guard["cache"]),
                "fresh_projection_eval_mode": True,
            },
            "provenance": {
                "queue_mechanism_stage": job["stage"],
                "queue_mechanism_arm": job["arm"],
                "semantic_calibration": str(guard["calibration"]),
                "semantic_cache_sha256": str(guard["cache_sha256"]),
            },
        },
    )
    return config


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    predictions = prediction_gate(pipeline)
    jobs = mechanism_jobs(pipeline)
    guard = pipeline["mechanism"]["semantic_guard"]
    cache = ROOT / str(guard["cache"])
    calibration_path = ROOT / str(guard["calibration"])
    if not cache.is_file():
        raise FileNotFoundError(cache)
    actual_cache_sha = sha256_file(cache)
    if actual_cache_sha != str(guard["cache_sha256"]):
        raise RuntimeError("semantic teacher cache differs from preregistered checksum")
    calibration = json.loads(calibration_path.read_text())
    expected_threshold = float(calibration["teacher_image_text"]["frozen_threshold"])
    if expected_threshold != float(guard["image_text_threshold"]):
        raise RuntimeError("semantic filter threshold differs from frozen calibration")
    for seed, path in WAVE0_CONTROLS.items():
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"missing seed-{seed} queue-free control: {path}")
    for queue_size in (16384, 65536):
        age = queue_size // int(pipeline["capacity"]["batch_size"])
        for seed in pipeline["capacity"]["seeds"]:
            path = ROOT / f"results/queue_identification/runs/b1024__age_{age}__both_fresh__seed_{seed}/metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing reused fresh control: {path}")
    destination = ROOT / str(pipeline["output_root"]) / "manifests"
    atomic_json(jobs, destination / "jobs.json")
    payload = {
        "status": "READY",
        "jobs": len(jobs),
        "capacity_jobs": sum(row["stage"] == "capacity" for row in jobs),
        "mechanism_jobs": sum(row["stage"] == "mechanism" for row in jobs),
        "prediction_ids": [row["id"] for row in predictions["unseen_predictions"]],
        "semantic_cache_sha256": actual_cache_sha,
        "semantic_threshold": expected_threshold,
        "reused_controls": 9,
    }
    atomic_json(payload, destination / "validation.json")
    return payload


def run_train(pipeline: dict[str, Any], index: int | None, *, resume: bool) -> dict[str, Any]:
    validate(pipeline)
    provenance = hardware_guard()
    job = _select(pipeline, index)
    config = build_config(pipeline, job)
    from src.alignment_v3.runner import _fingerprint

    result = train(config, _fingerprint(config), resume=resume)
    return {"run_id": job["run_id"], **provenance, **result}


def run_eval(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = hardware_guard()
    job = _select(pipeline, index)
    config = build_config(pipeline, job)
    checkpoint_dir = ROOT / config["training"]["save_dir"]
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError(f"missing checkpoint fingerprint: {checkpoint_dir}")
    device = get_device("auto")
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint_dir / "best.pt", model, device=device, expected_fingerprint=fingerprint
    )
    eval_config = deep_update(
        config, {"training": {"batch_size": int(config["evaluation"]["batch_size"])}}
    )
    _, loader = build_dataloaders(eval_config)
    embeddings = extract_embeddings(model, loader, device)
    from src.alignment_v3.runner import _metrics_from_embeddings

    metrics = _metrics_from_embeddings(embeddings, list(config["evaluation"]["k_values"]))
    destination = ROOT / str(pipeline["output_root"]) / "runs" / job["run_id"]
    row = {
        "status": "COMPLETE",
        **job,
        **metrics,
        **provenance,
        "fingerprint_digest": fingerprint.digest,
    }
    atomic_json(row, destination / "metrics.json")
    write_fingerprint(destination / "fingerprint.json", fingerprint)
    return row


def _metric(path: Path) -> float | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    for key in ("dev_mean_r_at_1", "mean_r_at_1", "mean_R@1"):
        if key in value:
            return float(value[key])
    return None


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby(groups, as_index=False).agg(
        n=("value", "count"),
        mean=("value", "mean"),
        sd=("value", "std"),
        minimum=("value", "min"),
        maximum=("value", "max"),
    )


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / str(pipeline["output_root"])
    run_by_key = {(j["stage"], j["arm"], j["memory_queue_size"], j["seed"]): j for j in mechanism_jobs(pipeline)}
    capacity_rows: list[dict[str, Any]] = []
    for queue_size in [0, 1024, 4096, 16384, 65536]:
        for seed in pipeline["capacity"]["seeds"]:
            seed = int(seed)
            if queue_size == 0:
                path = ROOT / WAVE0_CONTROLS[seed]
                source = "reused_queue_free"
            elif queue_size in {16384, 65536}:
                age = queue_size // 1024
                path = ROOT / f"results/queue_identification/runs/b1024__age_{age}__both_fresh__seed_{seed}/metrics.json"
                source = "reused_fresh_identification"
            else:
                job = run_by_key[("capacity", "fresh_capacity", queue_size, seed)]
                path = root / "runs" / job["run_id"] / "metrics.json"
                source = "new"
            value = _metric(path)
            if value is not None:
                capacity_rows.append({"queue_size": queue_size, "seed": seed, "value": value, "source": source})

    mechanism_rows: list[dict[str, Any]] = []
    for seed in pipeline["mechanism"]["seeds"]:
        seed = int(seed)
        control_path = ROOT / f"results/queue_identification/runs/b1024__age_16__both_fresh__seed_{seed}/metrics.json"
        control = _metric(control_path)
        if control is not None:
            mechanism_rows.append({"arm": "unweighted_fresh_control", "seed": seed, "value": control, "source": "reused"})
        for arm in pipeline["mechanism"]["arms"]:
            arm_id = str(arm["id"])
            job = run_by_key[("mechanism", arm_id, int(pipeline["mechanism"]["queue_size"]), seed)]
            value = _metric(root / "runs" / job["run_id"] / "metrics.json")
            if value is not None:
                mechanism_rows.append({"arm": arm_id, "seed": seed, "value": value, "source": "new"})

    capacity_frame = pd.DataFrame(capacity_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    capacity_summary = _summary(capacity_frame, ["queue_size"])
    mechanism_summary = _summary(mechanism_frame, ["arm"])
    atomic_csv(capacity_frame, root / "report/capacity_per_seed.csv")
    atomic_csv(capacity_summary, root / "report/capacity_summary.csv")
    atomic_csv(mechanism_frame, root / "report/mechanism_per_seed.csv")
    atomic_csv(mechanism_summary, root / "report/mechanism_summary.csv")

    effects: dict[str, Any] = {}
    if not mechanism_frame.empty:
        wide = mechanism_frame.pivot(index="seed", columns="arm", values="value")
        mde = float(pipeline["mechanism"]["minimum_practical_effect_pp"])
        for left, right, name in (
            ("semantic_filter", "matched_random_filter", "semantic_vs_matched_random"),
            ("mass_normalized", "unweighted_fresh_control", "mass_normalized_vs_control"),
            ("semantic_filter", "unweighted_fresh_control", "semantic_vs_control"),
        ):
            if left in wide and right in wide:
                values = (wide[left] - wide[right]).dropna() * 100.0
                if len(values):
                    effects[name] = {
                        "n": int(len(values)),
                        "mean_pp": float(values.mean()),
                        "sd_pp": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "clears_practical_threshold": bool(float(values.mean()) >= mde),
                    }

    complete_capacity = len(capacity_frame) == 15
    complete_mechanism = len(mechanism_frame) == 12
    payload = {
        "status": "COMPLETE" if complete_capacity and complete_mechanism else "PARTIAL",
        "capacity_complete": complete_capacity,
        "mechanism_complete": complete_mechanism,
        "capacity_rows": capacity_rows,
        "mechanism_rows": mechanism_rows,
        "effects": effects,
        "minimum_practical_effect_pp": float(pipeline["mechanism"]["minimum_practical_effect_pp"]),
    }
    atomic_json(payload, root / "report/report.json")
    lines = [
        "# Residual fresh-queue mechanism",
        "",
        f"Status: **{payload['status']}**. Primary metric is Flickr30k-validation mean bidirectional R@1.",
        "",
        "## Fresh-capacity curve",
        "",
        capacity_summary.to_markdown(index=False) if not capacity_summary.empty else "No capacity results yet.",
        "",
        "## Fixed-capacity mechanisms",
        "",
        mechanism_summary.to_markdown(index=False) if not mechanism_summary.empty else "No mechanism results yet.",
        "",
        "## Preregistered paired effects",
        "",
        "```json",
        json.dumps(effects, indent=2),
        "```",
        "",
        "All new arms use the fresh reprojected queue. The semantic threshold and teacher cache were frozen before these outcomes; matched-random filtering removes the same per-query count.",
    ]
    atomic_text("\n".join(lines) + "\n", root / "report/report.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("jobs", "validate", "train", "eval", "report"))
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    pipeline = load_pipeline(args.pipeline)
    if args.command == "jobs":
        result: Any = mechanism_jobs(pipeline)
    elif args.command == "validate":
        result = validate(pipeline)
    elif args.command == "train":
        result = run_train(pipeline, args.index, resume=not args.no_resume)
    elif args.command == "eval":
        result = run_eval(pipeline, args.index)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
