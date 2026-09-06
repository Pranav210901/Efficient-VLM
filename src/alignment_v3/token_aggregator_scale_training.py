from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import profile as frontier_profile
from src.alignment_v3.resolution_distillation_trajectory import (
    report as report_arm,
    validate as validate_arm,
    verify_snapshots,
)
from src.alignment_v3.runner import (
    ROOT,
    build_job_config,
    load_pipeline,
    sensitivity_jobs,
)
from src.phase15.io_utils import atomic_csv, atomic_json


PIPELINES = (
    "configs/token_aggregator_scale_training/pipeline_c2.yaml",
    "configs/token_aggregator_scale_training/pipeline_c3.yaml",
    "configs/token_aggregator_scale_training/pipeline_c4.yaml",
    "configs/token_aggregator_scale_training/pipeline_c5.yaml",
)
SPECS = {
    "C2": (256, 8, 1, 512, False, 2_203_524),
    "C3": (128, 4, 2, 256, False, 1_842_948),
    "C4": (256, 8, 2, 512, False, 2_730_628),
    "C5": (384, 6, 1, 768, True, 2_663_044),
}
OUTPUT = ROOT / "results/token_aggregator_scale_training"
C1_REPORT = (
    ROOT
    / "results/token_projection_training/C_transformer_block/report/report.json"
)


def validate_all() -> dict[str, Any]:
    scale_profile = json.loads(
        (ROOT / "results/token_aggregator_scale_profile/report.json").read_text()
    )
    if scale_profile.get("status") != "COMPLETE":
        raise RuntimeError("completed C1-C5 scale profile is required")
    if any(
        not bool(row["upper_quartile_below_ceiling"])
        for row in scale_profile["candidates"]
    ):
        raise RuntimeError("one or more requested scale candidates failed latency")
    rows = []
    for path in PIPELINES:
        _, pipeline = load_pipeline(path)
        token = pipeline["token_projection"]
        candidate = str(token["candidate"])
        dim, heads, blocks, ff_dim, native, expected_params = SPECS[candidate]
        observed = (
            int(token["pool_dim"]),
            int(token["heads"]),
            int(token["blocks"]),
            int(token["ff_dim"]),
            bool(token["native_width_identity"]),
            int(token["expected_inference_trainable_parameters"]),
        )
        expected = (dim, heads, blocks, ff_dim, native, expected_params)
        if observed != expected:
            raise RuntimeError(f"{candidate} pipeline identity mismatch")
        validation = validate_arm(path)
        for job in sensitivity_jobs(pipeline):
            config = build_job_config(pipeline, job, "sensitivity")
            recipe, training = config["recipe"], config["training"]
            resolved = (
                int(recipe["token_pooler_dim"]),
                int(recipe["token_pooler_heads"]),
                int(recipe["token_pooler_blocks"]),
                int(recipe["token_pooler_ff_dim"]),
                bool(recipe["token_pooler_native_width_identity"]),
            )
            if resolved != expected[:5]:
                raise RuntimeError(f"{candidate} resolved model mismatch")
            if (
                int(training["epochs"]) != 24
                or not bool(training["disable_early_stopping"])
                or not bool(training["save_every_epoch"])
                or float(training["lr"]) != 0.002545584412271571
            ):
                raise RuntimeError(f"{candidate} locked training recipe changed")
        rows.append(
            {
                "candidate": candidate,
                "pipeline": path,
                "dim": dim,
                "heads": heads,
                "blocks": blocks,
                "ff_dim": ff_dim,
                "native_width_identity": native,
                "inference_trainable_parameters": expected_params,
                "training_jobs": validation["training_jobs"],
                "lr_tuned_for_aggregator": False,
                "flickr_test_sealed": True,
            }
        )
    atomic_csv(pd.DataFrame(rows), OUTPUT / "manifests/jobs_verified.csv")
    result = {
        "status": "READY",
        "training_jobs": 12,
        "epoch_evaluation_jobs": 288,
        "trained_weight_profiles": 5,
        "c1_reused_not_retrained": True,
        "flickr_test_sealed": True,
        "latency_finding_before_training": (
            "C4 added 1,252,225 parameters and only 0.204 ms over the "
            "contemporaneous CLS baseline. At these scales the frozen image "
            "tower dominates full-stack latency; the binding declared "
            "aggregator constraint is the 5M parameter budget, not latency."
        ),
        "rows": rows,
    }
    atomic_json(result, OUTPUT / "manifests/design.json")
    return result


def verify_all() -> dict[str, Any]:
    values = [verify_snapshots(path) for path in PIPELINES]
    return {
        "status": "COMPLETE",
        "arms": 4,
        "snapshots": sum(int(value["snapshots"]) for value in values),
    }


def _profile_pipeline() -> dict[str, Any]:
    students = [
        {
            "id": "cls_baseline",
            "label": "Contemporaneous trained CLS-only baseline",
            "role": "contemporaneous_cls_baseline",
            "seeds": {
                "42": {
                    "checkpoint_dir": (
                        "checkpoints/resolution_distillation_224_long/"
                        "sensitivity/distill_strength_1p0__seed_42"
                    )
                }
            },
        }
    ]
    for path in PIPELINES:
        _, pipeline = load_pipeline(path)
        candidate = str(pipeline["token_projection"]["candidate"])
        arm_id = str(pipeline["teacher_extension"]["arm_id"])
        students.append(
            {
                "id": candidate,
                "label": str(pipeline["teacher_extension"]["label"]),
                "role": "scaled_token_aggregator",
                "seeds": {
                    "42": {
                        "checkpoint_dir": (
                            f"{pipeline['checkpoint_root']}/sensitivity/"
                            "distill_strength_1p0__seed_42"
                        )
                    }
                },
            }
        )
        if arm_id != pipeline["output_root"].split("/")[-1]:
            raise RuntimeError(f"{candidate} arm/output identity mismatch")
    return {
        "output_root": "results/token_aggregator_scale_training/profiling",
        "students": students,
        "references": {"entries": []},
        "datasets": {"primary": {"csv": "data/flickr30k/validation.csv"}},
        "profiling": {
            "representative_seed": 42,
            "verification_seed": 42,
            "paired_batch_size": 64,
            "warmup_iterations": 20,
            "timed_repeats": 100,
        },
        "resources": {"cpus_per_task": 12},
    }


def controlled_profile() -> dict[str, Any]:
    pipeline = _profile_pipeline()
    payloads = [
        frontier_profile(pipeline, index, dynamic_padding=True)
        for index in range(5)
    ]
    expected_order = ("cls_baseline", "C2", "C3", "C4", "C5")
    if tuple(value["entry_id"] for value in payloads) != expected_order:
        raise RuntimeError("controlled profile order changed")
    rows = []
    for value in payloads:
        entry = str(value["entry_id"])
        latency = value["full_stack_neural_latency_ms"]
        row = {
            "candidate": entry,
            "median_ms": float(latency["median"]),
            "q1_ms": float(latency["q1"]),
            "q3_ms": float(latency["q3"]),
            "iqr_ms": float(latency["iqr"]),
            "node": value["node"],
            "gpu_model": value["gpu_model"],
            "precision": value["precision"],
        }
        if entry != "cls_baseline":
            path = PIPELINES[("C2", "C3", "C4", "C5").index(entry)]
            _, arm = load_pipeline(path)
            export_path = (
                ROOT
                / arm["checkpoint_root"]
                / "sensitivity/distill_strength_1p0__seed_42/inference.pt"
            )
            export = torch.load(export_path, map_location="cpu", weights_only=False)
            leaked = [
                key
                for key in export["model_state"]
                if key.startswith(
                    ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
                )
            ]
            observed = int(
                export["parameter_summary"]["params_trainable_inference"]
            )
            expected = SPECS[entry][-1]
            if (
                observed != expected
                or leaked
                or not bool(export.get("training_only_heads_removed", False))
            ):
                raise RuntimeError(f"{entry} inference export contract failed")
            row["inference_trainable_parameters"] = observed
            row["teacher_heads_excluded"] = True
        rows.append(row)
    baseline = rows[0]["median_ms"]
    for row in rows:
        row["delta_from_contemporaneous_cls_ms"] = row["median_ms"] - baseline
        row["q3_below_9p48"] = row["q3_ms"] < 9.48
    atomic_csv(pd.DataFrame(rows), OUTPUT / "report/trained_latency.csv")
    result = {
        "status": "COMPLETE",
        "same_allocation": True,
        "protocol": {
            "batch_size": 64,
            "warmup": 20,
            "timed_repeats": 100,
            "cuda_synchronize": True,
            "native_bf16": True,
        },
        "rows": rows,
        "flickr_test_used": False,
    }
    atomic_json(result, OUTPUT / "report/trained_latency.json")
    return result


def _effect_label(value: float) -> str:
    if value > 0:
        return "positive deviation from additivity"
    if value < 0:
        return "negative deviation from additivity"
    return "exactly additive"


def report_all() -> dict[str, Any]:
    c1 = json.loads(C1_REPORT.read_text())
    reports = {"C1": c1}
    rows = [
        {
            "candidate": "C1",
            "flickr_mean_coco_selected": c1[
                "flickr_mean_under_coco_dev_epoch_selection"
            ],
            "flickr_mean_flickr_selected": c1[
                "flickr_mean_under_flickr_validation_epoch_selection"
            ],
            "selection_gain_pp": c1["selection_gain_pp"],
            "sd": c1["flickr_validation_selected_sd"],
            "min": c1["flickr_validation_selected_min"],
            "max": c1["flickr_validation_selected_max"],
            "seeds_peaking_at_epoch24": c1["seeds_with_flickr_peak_at_epoch24"],
            "inference_trainable_parameters": 1_710_468,
            "reused": True,
        }
    ]
    for path in PIPELINES:
        _, pipeline = load_pipeline(path)
        candidate = str(pipeline["token_projection"]["candidate"])
        report = report_arm(path)
        reports[candidate] = report
        rows.append(
            {
                "candidate": candidate,
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
                "inference_trainable_parameters": SPECS[candidate][-1],
                "reused": False,
            }
        )
    latency = json.loads((OUTPUT / "report/trained_latency.json").read_text())
    latency_by_id = {row["candidate"]: row for row in latency["rows"]}
    for row in rows:
        if row["candidate"] == "C1":
            previous = json.loads(
                (
                    ROOT
                    / "results/token_projection_training/C_transformer_block/"
                    "report/latency_profile.json"
                ).read_text()
            )
            measured = previous["full_stack_neural_latency_ms"]
            timed = {
                "median_ms": measured["median"],
                "q1_ms": measured["q1"],
                "q3_ms": measured["q3"],
                "q3_below_9p48": measured["q3"] < 9.48,
            }
            row["latency_measurement_session"] = "prior_C1_trained_profile"
        else:
            timed = latency_by_id[row["candidate"]]
            row["latency_measurement_session"] = (
                "contemporaneous_C2_C3_C4_C5_and_CLS"
            )
        row["trained_latency_median_ms"] = timed["median_ms"]
        row["trained_latency_q1_ms"] = timed["q1_ms"]
        row["trained_latency_q3_ms"] = timed["q3_ms"]
        row["latency_q3_below_9p48"] = timed["q3_below_9p48"]

    score = {
        key: float(value["flickr_mean_under_flickr_validation_epoch_selection"])
        for key, value in reports.items()
    }
    width_at_depth1 = (score["C2"] - score["C1"]) * 100.0
    depth_at_width128 = (score["C3"] - score["C1"]) * 100.0
    width_at_depth2 = (score["C4"] - score["C3"]) * 100.0
    depth_at_width256 = (score["C4"] - score["C2"]) * 100.0
    width_average = (width_at_depth1 + width_at_depth2) / 2.0
    depth_average = (depth_at_width128 + depth_at_width256) / 2.0
    interaction = (
        score["C4"] - score["C3"] - score["C2"] + score["C1"]
    ) * 100.0
    c5_vs_c2 = (score["C5"] - score["C2"]) * 100.0
    factorial = {
        "width_effect_at_one_block_pp": width_at_depth1,
        "width_effect_at_two_blocks_pp": width_at_depth2,
        "mean_width_effect_pp": width_average,
        "depth_effect_at_128d_pp": depth_at_width128,
        "depth_effect_at_256d_pp": depth_at_width256,
        "mean_depth_effect_pp": depth_average,
        "larger_average_effect": (
            "width" if width_average > depth_average else "depth"
            if depth_average > width_average
            else "equal"
        ),
        "width_depth_interaction_pp": interaction,
        "composition_description": _effect_label(interaction),
    }
    saturation = {
        candidate: {
            "flickr_peak_at_epoch24_seeds": int(
                reports[candidate]["seeds_with_flickr_peak_at_epoch24"]
            ),
            "ceiling_reached_not_converged": bool(
                reports[candidate]["ceiling_reached_not_converged"]
            ),
        }
        for candidate in reports
    }
    result = {
        "status": "COMPLETE",
        "baseline_to_beat": {
            "candidate": "C1",
            "flickr_validation_mean_R1": score["C1"],
        },
        "factorial_2x2": factorial,
        "native_width_comparison": {
            "comparison": "C5 minus C2",
            "flickr_difference_pp": c5_vs_c2,
            "C5_beats_C2": c5_vs_c2 > 0,
            "parameter_difference_C5_minus_C2": SPECS["C5"][-1]
            - SPECS["C2"][-1],
        },
        "saturation_evidence": saturation,
        "saturation_interpretation": (
            "Epoch-24 peak counts diagnose schedule truncation per arm; "
            "architectural saturation is judged from the reported marginal "
            "width, depth, interaction, and native-width effects without an "
            "invented threshold."
        ),
        "latency_finding": (
            "Before training, C4 added 1.252M parameters for 0.204 ms over "
            "same-session CLS. The frozen image tower dominated full-stack "
            "latency; within this tested range the declared binding constraint "
            "was the 5M adaptation budget rather than latency."
        ),
        "lr_disclosure": (
            "The LR was pinned from projection-only training and untuned for "
            "all scaled token aggregators."
        ),
        "flickr_test_used": False,
        "arms": rows,
    }
    atomic_csv(pd.DataFrame(rows), OUTPUT / "report/summary.csv")
    atomic_json(result, OUTPUT / "report/report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate", "verify", "profile", "report")
    )
    args = parser.parse_args()
    if args.command == "validate":
        value = validate_all()
    elif args.command == "verify":
        value = verify_all()
    elif args.command == "profile":
        value = controlled_profile()
    else:
        value = report_all()
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
