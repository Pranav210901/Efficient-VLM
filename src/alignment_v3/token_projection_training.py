from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.alignment_v3.resolution_distillation_trajectory import (
    report as report_arm,
    validate as validate_arm,
    verify_snapshots,
)
from src.alignment_v3.runner import ROOT, build_job_config, load_pipeline, sensitivity_jobs
from src.phase15.io_utils import atomic_csv, atomic_json


PIPELINES = (
    "configs/token_projection_training/pipeline_a.yaml",
    "configs/token_projection_training/pipeline_b.yaml",
    "configs/token_projection_training/pipeline_c.yaml",
)
EXPECTED = {
    "A": ("cls_mean_concat", 1_773_699),
    "B": ("learned_query_attention", 1_644_164),
    "C": ("transformer_128", 1_710_468),
}
OUTPUT = ROOT / "results/token_projection_training"


def validate_all() -> dict[str, Any]:
    profiling = json.loads(
        (ROOT / "results/token_projection_profile/report.json").read_text()
    )
    if profiling.get("status") != "COMPLETE" or len(profiling["candidates"]) != 3:
        raise RuntimeError("complete three-candidate latency profile is required")
    if any(
        not bool(candidate["median_below_openclip_ceiling"])
        for candidate in profiling["candidates"]
    ):
        raise RuntimeError("a requested candidate failed the latency ceiling")
    rows = []
    for path in PIPELINES:
        _, pipeline = load_pipeline(path)
        arm = pipeline["token_projection"]
        candidate = str(arm["candidate"])
        expected_aggregation, expected_parameters = EXPECTED[candidate]
        if str(arm["image_token_aggregation"]) != expected_aggregation:
            raise RuntimeError(f"{candidate} aggregation identity mismatch")
        if int(arm["expected_inference_trainable_parameters"]) != expected_parameters:
            raise RuntimeError(f"{candidate} parameter contract mismatch")
        validation = validate_arm(path)
        for job in sensitivity_jobs(pipeline):
            config = build_job_config(pipeline, job, "sensitivity")
            recipe = config["recipe"]
            training = config["training"]
            if str(recipe["image_token_aggregation"]) != expected_aggregation:
                raise RuntimeError(f"{candidate} resolved aggregation mismatch")
            if (
                int(training["epochs"]) != 24
                or not bool(training["disable_early_stopping"])
                or not bool(training["save_every_epoch"])
            ):
                raise RuntimeError(f"{candidate} trajectory controls changed")
            if float(training["lr"]) != 0.002545584412271571:
                raise RuntimeError(f"{candidate} pinned LR changed")
        rows.append(
            {
                "candidate": candidate,
                "pipeline": path,
                "aggregation": expected_aggregation,
                "training_jobs": validation["training_jobs"],
                "inference_trainable_parameters": expected_parameters,
                "lr": 0.002545584412271571,
                "lr_tuned_for_aggregator": False,
                "flickr_test_sealed": True,
            }
        )
    atomic_csv(pd.DataFrame(rows), OUTPUT / "manifests/jobs_verified.csv")
    result = {
        "status": "READY",
        "training_jobs": 9,
        "epoch_evaluation_jobs": 216,
        "trained_weight_profile_jobs": 3,
        "flickr_test_sealed": True,
        "profiling_interpretation": (
            "All three aggregators add no measurable latency. The contemporaneous "
            "CLS baseline was 8.630 ms versus the historical 8.546 ms; roughly "
            "one-percent session variation exceeded every candidate delta, so no "
            "latency ordering is claimed."
        ),
        "image_tower_observation": (
            "The image tower dominates full-stack latency sufficiently that "
            "downstream token aggregation was not measurable. The practical "
            "architectural headroom is therefore larger than the nominal 0.93 ms."
        ),
        "register_token_scope": (
            "The four DINOv3 register tokens are deliberately excluded to isolate "
            "how the 196 patch tokens are treated. Including registers is an "
            "untested plausible extension."
        ),
        "rows": rows,
    }
    atomic_json(result, OUTPUT / "manifests/design.json")
    return result


def verify_all() -> dict[str, Any]:
    values = [verify_snapshots(path) for path in PIPELINES]
    return {
        "status": "COMPLETE",
        "arms": len(values),
        "snapshots": sum(int(value["snapshots"]) for value in values),
    }


def report_all() -> dict[str, Any]:
    rows = []
    reports = []
    for path in PIPELINES:
        report = report_arm(path)
        _, pipeline = load_pipeline(path)
        latency_path = ROOT / pipeline["output_root"] / "report/latency_profile.json"
        if not latency_path.is_file():
            raise FileNotFoundError(latency_path)
        latency = json.loads(latency_path.read_text())
        expected = int(
            pipeline["token_projection"]["expected_inference_trainable_parameters"]
        )
        observed = int(latency["inference_trainable_parameters"])
        if observed != expected:
            raise RuntimeError(f"inference parameters {observed} != {expected}")
        candidate = str(pipeline["token_projection"]["candidate"])
        rows.append(
            {
                "candidate": candidate,
                "aggregation": pipeline["token_projection"][
                    "image_token_aggregation"
                ],
                "flickr_mean_coco_selected": report[
                    "flickr_mean_under_coco_dev_epoch_selection"
                ],
                "flickr_mean_flickr_selected": report[
                    "flickr_mean_under_flickr_validation_epoch_selection"
                ],
                "selection_gain_pp": report["selection_gain_pp"],
                "sd": report["flickr_validation_selected_sd"],
                "min": report["flickr_validation_selected_min"],
                "max": report["flickr_validation_selected_max"],
                "seeds_peaking_at_epoch24": report[
                    "seeds_with_flickr_peak_at_epoch24"
                ],
                "trained_full_stack_latency_median_ms": latency[
                    "full_stack_neural_latency_ms"
                ]["median"],
                "trained_full_stack_latency_q1_ms": latency[
                    "full_stack_neural_latency_ms"
                ]["q1"],
                "trained_full_stack_latency_q3_ms": latency[
                    "full_stack_neural_latency_ms"
                ]["q3"],
                "inference_trainable_parameters": observed,
                "teacher_heads_excluded": True,
                "lr_tuned_for_aggregator": False,
            }
        )
        reports.append(report)
    frame = pd.DataFrame(rows).sort_values("candidate")
    atomic_csv(frame, OUTPUT / "report/summary.csv")
    result = {
        "status": "COMPLETE",
        "baseline": {
            "flickr_validation_mean_R1": 0.44714,
            "full_stack_latency_ms": 8.546178694814444,
        },
        "latency_claim": "no measurable latency cost",
        "latency_ordering_claimed": False,
        "lr_disclosure": (
            "The LR was pinned from projection-only training and was not tuned "
            "for any aggregator."
        ),
        "flickr_test_used": False,
        "arms": rows,
    }
    atomic_json(result, OUTPUT / "report/report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "verify", "report"))
    args = parser.parse_args()
    if args.command == "validate":
        value = validate_all()
    elif args.command == "verify":
        value = verify_all()
    else:
        value = report_all()
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

