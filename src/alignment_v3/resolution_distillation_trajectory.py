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
    _metrics_from_embeddings,
    _student_loader,
    profile as frontier_profile,
)
from src.alignment_v3.fingerprint import read_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.resolution_arm import _guard, _select
from src.alignment_v3.runner import (
    ROOT,
    build_job_config,
    checkpoint_root,
    load_pipeline,
    sensitivity_jobs,
)
from src.alignment_v3.training import load_training_checkpoint
from src.phase15.io_utils import atomic_csv, atomic_json
from src.training.evaluate import extract_embeddings
from src.utils.config import load_config


def epoch_jobs(pipeline: dict[str, Any]) -> list[dict[str, int]]:
    return [
        {"seed": int(seed), "epoch": int(epoch)}
        for seed in pipeline["distillation_sensitivity"]["seeds"]
        for epoch in range(1, int(pipeline["trajectory"]["epochs"]) + 1)
    ]


def _run_id(seed: int) -> str:
    return f"distill_strength_1p0__seed_{int(seed)}"


def _artifact_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _checkpoint_dir(pipeline: dict[str, Any], seed: int) -> Path:
    return (
        checkpoint_root(pipeline)
        / "sensitivity"
        / _run_id(seed)
    )


def validate(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    if "test" in str(pipeline["optional_transfer"]["flickr30k_csv"]).lower():
        raise RuntimeError("Flickr30k test is sealed")
    if int(pipeline["trajectory"]["epochs"]) != 24:
        raise RuntimeError("trajectory must contain exactly 24 epochs")
    jobs = sensitivity_jobs(pipeline)
    if len(jobs) != 3 or any(not bool(job["distillation"]) for job in jobs):
        raise RuntimeError("trajectory requires exactly three distilled seeds")
    rows = []
    for job in jobs:
        config = build_job_config(pipeline, job, "sensitivity")
        identity = _guard({"image_size": 224}, config)
        training = config["training"]
        checks = {
            "epochs": int(training["epochs"]),
            "disable_early_stopping": bool(training["disable_early_stopping"]),
            "save_every_epoch": bool(training["save_every_epoch"]),
        }
        expected = {
            "epochs": 24,
            "disable_early_stopping": True,
            "save_every_epoch": True,
        }
        if checks != expected:
            raise RuntimeError(f"trajectory training controls mismatch: {checks}")
        rows.append(
            {
                **job,
                **identity,
                **checks,
                "resolved_lr": float(training["lr"]),
                "warmup_fraction": float(training.get("warmup_fraction", 0.05)),
                "schedule_total_epochs": 24,
            }
        )
    atomic_csv(
        pd.DataFrame(rows),
        _artifact_root(pipeline) / "manifests/jobs_verified.csv",
    )
    return {
        "status": "READY",
        "training_jobs": 3,
        "epoch_evaluation_jobs": len(epoch_jobs(pipeline)),
        "flickr_test_sealed": True,
        "rows": rows,
    }


def verify_snapshots(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows = []
    for job in epoch_jobs(pipeline):
        path = _checkpoint_dir(pipeline, job["seed"]) / f"epoch_{job['epoch']:02d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != job["epoch"]:
            raise RuntimeError(f"epoch identity mismatch: {path}")
        rows.append(
            {
                **job,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "inode": path.stat().st_ino,
                "coco_dev_mean_R1": float(payload["metrics"]["mean_R@1"]),
                "lr": float(payload["metrics"]["lr"]),
            }
        )
    # Hard links are intentional, but no two immutable epoch names may point
    # to the same checkpoint state.
    inodes = [row["inode"] for row in rows]
    if len(inodes) != len(set(inodes)):
        raise RuntimeError("two immutable epoch snapshots share an inode")
    atomic_csv(
        pd.DataFrame(rows),
        _artifact_root(pipeline) / "manifests/epoch_snapshots.csv",
    )
    return {"status": "COMPLETE", "snapshots": len(rows)}


def evaluate_epoch(pipeline_path: str, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(epoch_jobs(pipeline), index)
    checkpoint_dir = _checkpoint_dir(pipeline, job["seed"])
    config = load_config(checkpoint_dir / "config.yaml")
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError(f"missing fingerprint in {checkpoint_dir}")
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    model = build_model(config).to(device).eval()
    checkpoint_path = checkpoint_dir / f"epoch_{job['epoch']:02d}.pt"
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        model,
        device=device,
        expected_fingerprint=fingerprint,
    )
    identity = _guard({"image_size": 224}, config, model)
    loader = _student_loader(config, ROOT / "data/flickr30k/validation.csv", 256)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, [1, 5, 10])
    payload = {
        "status": "COMPLETE",
        **job,
        **identity,
        **metrics,
        **provenance,
        "coco_dev_mean_R1": float(checkpoint["metrics"]["mean_R@1"]),
        "lr": float(checkpoint["metrics"]["lr"]),
        "dataset": "Flickr30k Karpathy validation",
        "test_split_used": False,
        "checkpoint_fingerprint": fingerprint.digest,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }
    destination = (
        _artifact_root(pipeline)
        / "trajectory"
        / f"seed_{job['seed']}"
        / f"epoch_{job['epoch']:02d}.json"
    )
    atomic_json(payload, destination)
    return payload


def _unique_peak(
    frame: pd.DataFrame,
    column: str,
    seed: int,
    *,
    tie_break: str | None = None,
    tie_events: list[dict[str, Any]] | None = None,
) -> pd.Series:
    maximum = float(frame[column].max())
    peaks = frame[frame[column] == maximum]
    if len(peaks) == 1:
        return peaks.iloc[0]
    if tie_break != "earliest_epoch":
        raise RuntimeError(
            f"seed {seed} has tied {column} maxima at epochs "
            f"{peaks['epoch'].astype(int).tolist()}; no tie-break was pre-registered"
        )
    selected = peaks.sort_values("epoch").iloc[0]
    if tie_events is not None:
        tie_events.append(
            {
                "seed": int(seed),
                "metric": column,
                "maximum": maximum,
                "tied_epochs": peaks["epoch"].astype(int).tolist(),
                "selected_epoch": int(selected["epoch"]),
                "rule": "earliest_epoch",
            }
        )
    return selected


def report(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    reporting = dict(pipeline.get("reporting", {}))
    peak_tie_break = reporting.get("peak_tie_break")
    tie_events: list[dict[str, Any]] = []
    rows = []
    for job in epoch_jobs(pipeline):
        path = (
            _artifact_root(pipeline)
            / "trajectory"
            / f"seed_{job['seed']}"
            / f"epoch_{job['epoch']:02d}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows).sort_values(["seed", "epoch"])
    per_seed = []
    for seed, values in frame.groupby("seed"):
        coco_peak = _unique_peak(
            values,
            "coco_dev_mean_R1",
            int(seed),
            tie_break=peak_tie_break,
            tie_events=tie_events,
        )
        flickr_peak = _unique_peak(
            values,
            "mean_R@1",
            int(seed),
            tie_break=peak_tie_break,
            tie_events=tie_events,
        )
        epoch24 = values[values["epoch"] == 24].iloc[0]
        per_seed.append(
            {
                "seed": int(seed),
                "coco_selected_epoch": int(coco_peak["epoch"]),
                "coco_peak_R1": float(coco_peak["coco_dev_mean_R1"]),
                "flickr_at_coco_selected_epoch_R1": float(coco_peak["mean_R@1"]),
                "flickr_selected_epoch": int(flickr_peak["epoch"]),
                "flickr_peak_R1": float(flickr_peak["mean_R@1"]),
                "coco_at_flickr_selected_epoch_R1": float(
                    flickr_peak["coco_dev_mean_R1"]
                ),
                "epoch24_coco_dev_R1": float(epoch24["coco_dev_mean_R1"]),
                "epoch24_flickr_validation_R1": float(epoch24["mean_R@1"]),
                "coco_peak_at_epoch24": int(coco_peak["epoch"]) == 24,
                "flickr_peak_at_epoch24": int(flickr_peak["epoch"]) == 24,
            }
        )
    selection = pd.DataFrame(per_seed)
    coco_rule_mean = float(selection["flickr_at_coco_selected_epoch_R1"].mean())
    flickr_rule_mean = float(selection["flickr_peak_R1"].mean())
    flickr_values = selection["flickr_peak_R1"].astype(float)
    result = {
        "status": "COMPLETE",
        "arm_id": str(pipeline.get("teacher_extension", {}).get("arm_id", "mobileclip2")),
        "teacher_ids": list(
            pipeline.get("teacher_extension", {}).get(
                "teacher_ids", ["mobileclip2_s0_dfndr2b"]
            )
        ),
        "seeds": [42, 43, 44],
        "schedule": {
            "type": "single 24-epoch cosine",
            "warmup_fraction": 0.05,
            "early_stopping": False,
        },
        "old_12epoch_complete_run_flickr_validation_mean_R1": float(
            pipeline["trajectory"]["old_12epoch_flickr_validation_mean_R1"]
        ),
        "new_24epoch_complete_run_comparable_to_old_complete_run": True,
        "intermediate_epochs_comparable_between_schedules": False,
        "comparability_boundary": (
            "The complete 24-epoch run is comparable with the complete prior "
            "12-epoch run. Epoch-number-matched intermediate checkpoints are "
            "not comparable because their learning rates differ."
        ),
        "flickr_mean_under_coco_dev_epoch_selection": coco_rule_mean,
        "flickr_mean_under_flickr_validation_epoch_selection": flickr_rule_mean,
        "flickr_validation_selected_sd": float(flickr_values.std(ddof=1)),
        "flickr_validation_selected_min": float(flickr_values.min()),
        "flickr_validation_selected_max": float(flickr_values.max()),
        "selection_gain_pp": (flickr_rule_mean - coco_rule_mean) * 100.0,
        "double_selection_disclosure": (
            "The 224px teacher-distilled configuration and checkpoint epochs "
            "were selected using Flickr30k validation. The validation figure "
            "is therefore optimistically influenced by configuration and "
            "checkpoint selection; the measured epoch-selection gain is "
            "reported explicitly."
        ),
        "seeds_with_coco_peak_at_epoch24": int(
            selection["coco_peak_at_epoch24"].sum()
        ),
        "seeds_with_flickr_peak_at_epoch24": int(
            selection["flickr_peak_at_epoch24"].sum()
        ),
        "ceiling_reached_not_converged": bool(
            selection["coco_peak_at_epoch24"].any()
            or selection["flickr_peak_at_epoch24"].any()
        ),
        "flickr_test_used": False,
        "per_seed": per_seed,
    }
    if peak_tie_break is not None:
        result["selection_tie_break"] = {
            "rule": peak_tie_break,
            "status": reporting.get("tie_break_status"),
            "disclosure": reporting.get("tie_break_disclosure"),
            "events": tie_events,
        }
    if "token_projection" in pipeline:
        result["token_projection"] = dict(pipeline["token_projection"])
        result["baseline_to_beat"] = dict(
            reporting.get(
                "baseline_to_beat",
                {
                    "flickr_validation_mean_R1": 0.44714,
                    "full_stack_latency_ms": 8.546178694814444,
                },
            )
        )
        result["lr_disclosure"] = (
            "The learning rate remains pinned from projection-only training "
            "and is untuned for token aggregators."
        )
    destination = _artifact_root(pipeline) / "report"
    atomic_csv(frame, destination / "per_epoch.csv")
    atomic_csv(selection, destination / "per_seed_peaks.csv")
    atomic_json(result, destination / "report.json")
    return result


def profile_arm(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    seed = 42
    checkpoint_dir = _checkpoint_dir(pipeline, seed)
    profile_pipeline = {
        "output_root": str(_artifact_root(pipeline).relative_to(ROOT) / "profiling"),
        "students": [
            {
                "id": str(pipeline.get("teacher_extension", {}).get("arm_id", "teacher_arm")),
                "label": str(
                    pipeline.get("teacher_extension", {}).get(
                        "label", "224px teacher-distilled student"
                    )
                ),
                "role": "teacher_extension_224",
                "seeds": {
                    str(seed): {
                        "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT))
                    }
                },
            }
        ],
        "references": {"entries": []},
        "datasets": {"primary": {"csv": "data/flickr30k/validation.csv"}},
        "profiling": {
            "representative_seed": seed,
            "verification_seed": seed,
            "paired_batch_size": 64,
            "warmup_iterations": 20,
            "timed_repeats": 100,
        },
        "resources": {"cpus_per_task": 12},
    }
    payload = frontier_profile(profile_pipeline, 0, dynamic_padding=True)
    if int(payload["native_image_resolution"]) != 224:
        raise RuntimeError(
            f"latency profiler used {payload['native_image_resolution']}px instead of 224px"
        )
    payload.update(
        {
            "latency_budget_ms": 8.55,
            "flickr_test_used": False,
            "training_teachers_excluded_from_inference": True,
        }
    )
    expected = pipeline.get("token_projection", {}).get(
        "expected_inference_trainable_parameters"
    )
    if expected is not None:
        export = torch.load(
            checkpoint_dir / "inference.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not bool(export.get("training_only_heads_removed", False)):
            raise RuntimeError("inference export does not attest teacher-head removal")
        leaked = [
            key
            for key in export["model_state"]
            if key.startswith(
                ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
            )
        ]
        if leaked:
            raise RuntimeError(
                f"training-only teacher head leaked into inference export: {leaked[:3]}"
            )
        observed = int(
            export["parameter_summary"]["params_trainable_inference"]
        )
        if observed != int(expected):
            raise RuntimeError(
                f"inference trainable parameter mismatch: {observed} != {expected}"
            )
        payload["inference_trainable_parameters"] = observed
        payload["teacher_head_keys_in_inference_export"] = 0
    atomic_json(payload, _artifact_root(pipeline) / "report/latency_profile.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate", "verify-snapshots", "evaluate", "report", "profile"),
    )
    parser.add_argument(
        "--pipeline",
        default="configs/resolution_distillation_224_long/pipeline.yaml",
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.pipeline)
    elif args.command == "verify-snapshots":
        result = verify_snapshots(args.pipeline)
    elif args.command == "evaluate":
        result = evaluate_epoch(args.pipeline, args.index)
    elif args.command == "profile":
        result = profile_arm(args.pipeline)
    else:
        result = report(args.pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
