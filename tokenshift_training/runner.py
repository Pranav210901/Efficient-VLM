from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from freezeshift.model_adaptation import (
    build_model as build_parent_model,
    lora_parameter_count,
    merge_lora_,
)
from freezeshift.runner import _load_selected as load_existing_selected
from src.alignment_v3 import efficiency_frontier as frontier
from src.alignment_v3 import resolution_distillation_trajectory as trajectory
from src.alignment_v3 import runner as alignment_runner
from src.alignment_v3 import training as alignment_training
from src.alignment_v3.fingerprint import read_fingerprint
from src.alignment_v3.references import REFERENCE_CHECKPOINTS, build_reference
from src.alignment_v3.training import load_training_checkpoint
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.utils.config import load_config
from tokenshift.token_reduction import expected_layout
from tokenshift_training.model import build_model


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "tokenshift_training"
PIPELINES = (
    "tokenshift_training/configs/pipeline_frozen_block8.yaml",
    "tokenshift_training/configs/pipeline_frozen_block6.yaml",
    "tokenshift_training/configs/pipeline_dual_block8.yaml",
    "tokenshift_training/configs/pipeline_dual_block6.yaml",
)
BASELINES = {
    "frozen_no_merge": "configs/text_aggregation_study/pipeline_m_t1.yaml",
    "dual_lora_no_merge": "freezeshift/configs/pipeline_dual.yaml",
}
REFERENCE_ID = "openclip_vit_b32_quickgelu_openai"
REPETITIONS = 10


def _patch_builders() -> None:
    alignment_training.build_model = build_model
    alignment_runner.build_model = build_model
    trajectory.build_model = build_model
    frontier.build_model = build_model


def _package_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(STUDY.rglob("*")):
        if not path.is_file() or any(
            part in {"results", "logs", "checkpoints", "__pycache__"}
            for part in path.parts
        ):
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pipeline_config(path: str, seed_index: int = 0):
    _, pipeline = alignment_runner.load_pipeline(path)
    job = alignment_runner.sensitivity_jobs(pipeline)[seed_index]
    config = alignment_runner.build_job_config(pipeline, job, "sensitivity")
    return pipeline, config


def _artifact_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _checkpoint_dir(pipeline: dict[str, Any], seed: int) -> Path:
    return (
        ROOT
        / str(pipeline["checkpoint_root"])
        / "sensitivity"
        / f"distill_strength_1p0__seed_{int(seed)}"
    )


def validate() -> dict[str, Any]:
    predictions = json.loads((STUDY / "predictions.json").read_text())
    if not predictions.get("written_before_any_tokenshift_training_runs"):
        raise RuntimeError("TokenShift-training predictions were not frozen")
    if not predictions.get("flickr_test_sealed"):
        raise RuntimeError("Flickr test seal is absent")

    profile = json.loads((ROOT / "tokenshift/results/profile/report.json").read_text())
    if profile.get("status") != "COMPLETE" or profile.get("stage") != "PROFILE_ONLY":
        raise RuntimeError("completed TokenShift latency profile is missing")
    profile_rows = {row["arm"]: row for row in profile["rows"]}
    for arm in ("merge_after_block_8", "merge_after_block_6"):
        row = profile_rows.get(arm)
        if row is None or float(row["paired_q3_delta_ci95_high_ms"]) >= 0:
            raise RuntimeError(f"TokenShift latency reduction was not verified for {arm}")

    baseline_reports = {}
    for arm, path in BASELINES.items():
        _, pipeline = alignment_runner.load_pipeline(path)
        report_path = _artifact_root(pipeline) / "report/report.json"
        report = json.loads(report_path.read_text())
        if report.get("status") != "COMPLETE" or report.get("flickr_test_used") is not False:
            raise RuntimeError(f"baseline is incomplete or test-contaminated: {arm}")
        baseline_reports[arm] = {
            "pipeline": path,
            "validation_mean_R1": report[
                "flickr_mean_under_flickr_validation_epoch_selection"
            ],
            "report": str(report_path.relative_to(ROOT)),
        }

    rows = []
    seen_roots: set[str] = set()
    for pipeline_index, path in enumerate(PIPELINES):
        pipeline, config = _pipeline_config(path)
        if "test" in str(pipeline["optional_transfer"]["flickr30k_csv"]).lower():
            raise RuntimeError(f"Flickr test path entered {path}")
        metadata = pipeline["tokenshift_training"]
        block = int(config["recipe"]["tokenshift_merge_after_block"])
        if block != int(metadata["merge_after_block"]) or block not in {6, 8}:
            raise RuntimeError(f"merge-location mismatch in {path}")
        output = str(pipeline["output_root"])
        checkpoint = str(pipeline["checkpoint_root"])
        if output in seen_roots or checkpoint in seen_roots:
            raise RuntimeError(f"duplicate artifact root in {path}")
        seen_roots.update({output, checkpoint})

        parent = build_parent_model(config, pretrained=False)
        parent_names = tuple(parent.state_dict())
        parent_parameters = sum(p.numel() for p in parent.parameters())
        model = build_model(config, pretrained=False)
        if tuple(model.state_dict()) != parent_names:
            raise RuntimeError(f"TokenShift changed checkpoint keys in {path}")
        if sum(p.numel() for p in model.parameters()) != parent_parameters:
            raise RuntimeError(f"TokenShift changed parameter count in {path}")
        controller = model.vision_encoder.model._tokenshift_controller
        if int(controller.merge_after_block) != block:
            raise RuntimeError(f"runtime controller mismatch in {path}")

        summary = model.parameter_summary()
        expected_parameters = int(metadata["expected_inference_trainable_parameters"])
        if int(summary["params_trainable_inference"]) != expected_parameters:
            raise RuntimeError(f"parameter contract mismatch in {path}: {summary}")
        if expected_parameters >= 5_000_000:
            raise RuntimeError(f"5M inference-adaptation budget exceeded in {path}")
        observed_lora = lora_parameter_count(model)
        expected_lora = 0 if metadata["parent"] == "frozen" else 1_966_080
        if observed_lora != expected_lora:
            raise RuntimeError(f"LoRA count mismatch in {path}: {observed_lora}")
        base_encoder_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                name.startswith("vision_encoder.model.")
                or name.startswith("text_encoder.encoder.")
            )
            and ".lora_" not in name
        ]
        if base_encoder_trainable:
            raise RuntimeError(
                f"base encoder weights became trainable in {path}: "
                f"{base_encoder_trainable[:3]}"
            )
        trajectory_result = trajectory.validate(path)
        if trajectory_result["training_jobs"] != 3:
            raise RuntimeError(f"trajectory validation failed in {path}")
        rows.append(
            {
                "pipeline_index": pipeline_index,
                "pipeline": path,
                "arm": pipeline["teacher_extension"]["arm_id"],
                "parent": metadata["parent"],
                "merge_after_block": block,
                "new_training_runs": 3,
                "lora_parameters": observed_lora,
                "inference_trainable_parameters": expected_parameters,
                "patch_tokens_after": expected_layout(block).patch_tokens_after,
                "base_encoder_weights_frozen": True,
            }
        )

    reference = REFERENCE_CHECKPOINTS.get(REFERENCE_ID)
    if reference != ("ViT-B-32-quickgelu", "openai"):
        raise RuntimeError(f"OpenCLIP registry mismatch: {reference}")
    receipt = {
        "status": "READY",
        "package_sha256": _package_hash(),
        "study": "tokenshift_training",
        "new_training_runs": 12,
        "training_array_indices": list(range(12)),
        "seeds": [42, 43, 44],
        "pipelines": rows,
        "existing_no_merge_baselines": baseline_reports,
        "tokenshift_profile": {
            "path": "tokenshift/results/profile/report.json",
            "block8_paired_delta_ms": profile_rows["merge_after_block_8"][
                "paired_q3_delta_mean_ms"
            ],
            "block6_paired_delta_ms": profile_rows["merge_after_block_6"][
                "paired_q3_delta_mean_ms"
            ],
        },
        "automatic_promotion": False,
        "flickr_test_sealed": True,
    }
    atomic_csv(pd.DataFrame(rows), STUDY / "results/manifests/design.csv")
    atomic_json(receipt, STUDY / "results/manifests/design.json")
    return receipt


def _assert_frozen() -> dict[str, Any]:
    path = STUDY / "results/manifests/design.json"
    if not path.is_file():
        raise RuntimeError("run TokenShift-training validate first")
    receipt = json.loads(path.read_text())
    if receipt["package_sha256"] != _package_hash():
        raise RuntimeError("TokenShift-training package changed after validation")
    return receipt


def smoke() -> dict[str, Any]:
    _assert_frozen()
    rows = []
    for path in (PIPELINES[0], PIPELINES[3]):
        pipeline, config = _pipeline_config(path)
        model = build_model(config, pretrained=False).eval()
        images = torch.randn(2, 3, 224, 224)
        captions = ["a person riding a bicycle", "two dogs in a field"]
        with torch.inference_mode():
            image = model.encode_image(images)
            text = model.encode_text(captions)
        if image.shape != (2, 384) or text.shape != (2, 384):
            raise RuntimeError(f"embedding interface mismatch in {path}")
        before_image = image
        before_text = text
        merged = merge_lora_(model)
        with torch.inference_mode():
            after_image = model.encode_image(images)
            after_text = model.encode_text(captions)
        maximum_delta = max(
            float((before_image - after_image).abs().max()),
            float((before_text - after_text).abs().max()),
        )
        if maximum_delta > 1e-6:
            raise RuntimeError(f"LoRA merge equivalence failed in {path}")
        rows.append(
            {
                "arm": pipeline["teacher_extension"]["arm_id"],
                "embedding_shape": [2, 384],
                "merged_lora_modules": len(merged),
                "merge_max_abs_delta_fp32": maximum_delta,
            }
        )
    result = {"status": "COMPLETE", "rows": rows}
    atomic_json(result, STUDY / "results/manifests/smoke.json")
    return result


def train(pipeline_index: int, seed_index: int, resume: bool = True) -> dict[str, Any]:
    _assert_frozen()
    _patch_builders()
    return alignment_runner.run_training_stage(
        PIPELINES[pipeline_index], "sensitivity", seed_index, resume=resume
    )


def verify(pipeline_index: int) -> dict[str, Any]:
    _assert_frozen()
    return trajectory.verify_snapshots(PIPELINES[pipeline_index])


def _epoch_result_path(pipeline: dict[str, Any], seed: int, epoch: int) -> Path:
    return (
        _artifact_root(pipeline)
        / "trajectory"
        / f"seed_{seed}"
        / f"epoch_{epoch:02d}.json"
    )


def evaluate_seed(pipeline_index: int, seed_index: int) -> dict[str, Any]:
    _assert_frozen()
    _patch_builders()
    _, pipeline = alignment_runner.load_pipeline(PIPELINES[pipeline_index])
    seed = int(pipeline["distillation_sensitivity"]["seeds"][seed_index])
    completed = 0
    skipped = 0
    for epoch in range(1, 25):
        destination = _epoch_result_path(pipeline, seed, epoch)
        if destination.is_file():
            payload = json.loads(destination.read_text())
            if (
                payload.get("status") == "COMPLETE"
                and int(payload.get("seed", -1)) == seed
                and int(payload.get("epoch", -1)) == epoch
                and payload.get("test_split_used") is False
            ):
                skipped += 1
                continue
        trajectory.evaluate_epoch(
            PIPELINES[pipeline_index], seed_index * 24 + (epoch - 1)
        )
        completed += 1
    return {
        "status": "COMPLETE",
        "pipeline_index": pipeline_index,
        "seed": seed,
        "epochs_evaluated": completed,
        "epochs_reused": skipped,
        "flickr_test_used": False,
    }


def report_arm(pipeline_index: int) -> dict[str, Any]:
    _assert_frozen()
    return trajectory.report(PIPELINES[pipeline_index])


def _load_candidate_selected(path: str, seed: int, device: torch.device):
    _, pipeline = alignment_runner.load_pipeline(path)
    report_path = _artifact_root(pipeline) / "report/report.json"
    report = json.loads(report_path.read_text())
    row = next(value for value in report["per_seed"] if int(value["seed"]) == seed)
    epoch = int(row["flickr_selected_epoch"])
    checkpoint_dir = _checkpoint_dir(pipeline, seed)
    config = load_config(checkpoint_dir / "config.yaml")
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError(f"missing fingerprint in {checkpoint_dir}")
    model = build_model(config).to(device).eval()
    load_training_checkpoint(
        checkpoint_dir / f"epoch_{epoch:02d}.pt",
        model,
        device=device,
        expected_fingerprint=fingerprint,
    )
    return model, config, model.parameter_summary(), epoch


def _profile_roster(device: torch.device):
    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, dict[str, Any] | None] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for arm, pipeline in BASELINES.items():
        model, config, summary, epoch = load_existing_selected(pipeline, 42, device)
        merged = merge_lora_(model)
        model.eval()
        models[arm] = model
        configs[arm] = config
        metadata[arm] = {
            "role": "existing_no_merge_parent",
            "selected_seed": 42,
            "selected_epoch": epoch,
            "merged_lora_modules": len(merged),
            "inference_trainable_parameters": summary["params_trainable_inference"],
            "merge_after_block": None,
        }
    for path in PIPELINES:
        _, pipeline = alignment_runner.load_pipeline(path)
        arm = pipeline["teacher_extension"]["arm_id"]
        model, config, summary, epoch = _load_candidate_selected(path, 42, device)
        merged = merge_lora_(model)
        model.eval()
        models[arm] = model
        configs[arm] = config
        metadata[arm] = {
            "role": "tokenshift_candidate",
            "selected_seed": 42,
            "selected_epoch": epoch,
            "merged_lora_modules": len(merged),
            "inference_trainable_parameters": summary["params_trainable_inference"],
            "merge_after_block": pipeline["tokenshift_training"]["merge_after_block"],
        }
    reference = build_reference(REFERENCE_ID).to(device).eval()
    models[REFERENCE_ID] = reference
    configs[REFERENCE_ID] = None
    metadata[REFERENCE_ID] = {
        "role": "efficiency_reference",
        "selected_seed": None,
        "selected_epoch": None,
        "merged_lora_modules": 0,
        "inference_trainable_parameters": 0,
        "merge_after_block": None,
    }
    for arm, model in models.items():
        frontier._assert_fully_eval(model, f"TokenShift-training profile {arm}")
    return models, configs, metadata


def _profile_workloads(models, configs, device: torch.device):
    pil_images, captions = frontier._sample_batch(
        ROOT / "data/flickr30k/validation.csv", 64
    )
    workloads = {}
    for arm, model in models.items():
        config = configs[arm]
        transform = model.preprocess if config is None else frontier._student_transform(config)
        images = torch.stack([transform(image) for image in pil_images]).to(device)
        cpu_tokens = frontier._tokenize_native(
            model, captions, pad_to_native_context=False
        )
        workloads[arm] = {
            "images": images,
            "tokens": cpu_tokens.to(device),
            "token_stats": frontier._token_shape_and_utilization(cpu_tokens),
            "image_resolution": (
                int(model.image_size) if config is None else int(config["data"]["image_size"])
            ),
        }
    return workloads


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _latency_classification(low: float, high: float) -> str:
    if high < 0:
        return "FASTER"
    if low > 0:
        return "SLOWER"
    return "LATENCY_EQUIVALENT"


def _summarize_profile(raw: pd.DataFrame, hardware: dict[str, Any]) -> dict[str, Any]:
    roster = tuple([*BASELINES, *[
        alignment_runner.load_pipeline(path)[1]["teacher_extension"]["arm_id"]
        for path in PIPELINES
    ], REFERENCE_ID])
    indexed = {
        arm: raw[raw["arm"] == arm].set_index("repetition").sort_index()
        for arm in roster
    }
    reference = indexed[REFERENCE_ID]
    parent_for = {
        "frozen_block8": "frozen_no_merge",
        "frozen_block6": "frozen_no_merge",
        "dual_block8": "dual_lora_no_merge",
        "dual_block6": "dual_lora_no_merge",
    }
    rows = []
    paired_rows = []
    for arm in roster:
        frame = indexed[arm]
        values = frame["q3_ms"].to_numpy(dtype=float)
        own_low, own_high = _bootstrap_mean_ci(values, 20260808 + roster.index(arm))
        row: dict[str, Any] = {
            "arm": arm,
            "independent_repetitions": len(values),
            "q3_mean_ms": float(values.mean()),
            "q3_median_ms": float(np.median(values)),
            "q3_sd_ms": float(values.std(ddof=1)),
            "q3_mean_ci95_low_ms": own_low,
            "q3_mean_ci95_high_ms": own_high,
            "image_q3_mean_ms": float(frame["image_q3_ms"].mean()),
            "text_q3_mean_ms": float(frame["text_q3_ms"].mean()),
        }
        if arm != REFERENCE_ID:
            delta = (frame["q3_ms"] - reference["q3_ms"]).to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(delta, 20260908 + roster.index(arm))
            row.update(
                {
                    "paired_delta_vs_openclip_mean_ms": float(delta.mean()),
                    "paired_delta_vs_openclip_ci95_low_ms": low,
                    "paired_delta_vs_openclip_ci95_high_ms": high,
                    "classification_vs_openclip": _latency_classification(low, high),
                }
            )
        if arm in parent_for:
            parent = parent_for[arm]
            delta = (frame["q3_ms"] - indexed[parent]["q3_ms"]).to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(delta, 20261008 + roster.index(arm))
            row.update(
                {
                    "parent": parent,
                    "paired_delta_vs_parent_mean_ms": float(delta.mean()),
                    "paired_delta_vs_parent_ci95_low_ms": low,
                    "paired_delta_vs_parent_ci95_high_ms": high,
                    "classification_vs_parent": _latency_classification(low, high),
                }
            )
            paired_rows.extend(
                {
                    "arm": arm,
                    "parent": parent,
                    "repetition": int(repetition),
                    "paired_q3_delta_ms": float(value),
                }
                for repetition, value in zip(frame.index, delta)
            )
        rows.append(row)
    result = {
        "status": "COMPLETE",
        "same_allocation": True,
        "primary_reference": REFERENCE_ID,
        "protocol": {
            "batch_size": 64,
            "warmup_iterations_per_repetition": 20,
            "timed_iterations_per_repetition": 100,
            "independent_repetitions": REPETITIONS,
            "order_randomized_within_repetition": True,
            "precision": "native_bf16",
        },
        "rows": rows,
        "hardware": hardware,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(rows), STUDY / "results/profile/summary.csv")
    atomic_csv(pd.DataFrame(paired_rows), STUDY / "results/profile/paired_deltas.csv")
    atomic_json(result, STUDY / "results/profile/report.json")
    return result


def profile() -> dict[str, Any]:
    _assert_frozen()
    hardware = frontier._hardware_guard({})
    device = hardware.pop("device")
    models, configs, metadata = _profile_roster(device)
    workloads = _profile_workloads(models, configs, device)
    roster = tuple(models)
    job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
    raw_path = STUDY / "results/profile/runs" / f"job_{job_id}" / "raw_runs.csv"
    existing = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame()
    records = existing.to_dict("records") if not existing.empty else []
    complete = {(int(row["repetition"]), str(row["arm"])) for row in records}
    rng = random.Random(20260808)
    orders = []
    for _ in range(REPETITIONS):
        order = list(roster)
        rng.shuffle(order)
        orders.append(order)
    for repetition, order in enumerate(orders):
        for order_position, arm in enumerate(order):
            if (repetition, arm) in complete:
                continue
            model = models[arm]
            images = workloads[arm]["images"]
            tokens = workloads[arm]["tokens"]
            full = frontier._quartiles(
                frontier._cuda_times(
                    lambda: (
                        model.encode_image(images),
                        frontier._encode_tokens(model, tokens),
                    ),
                    20,
                    100,
                    device,
                )
            )
            image = frontier._quartiles(
                frontier._cuda_times(lambda: model.encode_image(images), 20, 100, device)
            )
            text = frontier._quartiles(
                frontier._cuda_times(
                    lambda: frontier._encode_tokens(model, tokens), 20, 100, device
                )
            )
            records.append(
                {
                    "repetition": repetition,
                    "order_position": order_position,
                    "arm": arm,
                    "median_ms": full["median"],
                    "q1_ms": full["q1"],
                    "q3_ms": full["q3"],
                    "image_q3_ms": image["q3"],
                    "text_q3_ms": text["q3"],
                    "image_resolution": workloads[arm]["image_resolution"],
                    "padded_sequence_length": workloads[arm]["token_stats"][
                        "padded_sequence_length"
                    ],
                    **metadata[arm],
                }
            )
            atomic_csv(pd.DataFrame(records), raw_path)
    raw = pd.DataFrame(records)
    counts = raw.groupby("arm").size().to_dict()
    expected = {arm: REPETITIONS for arm in roster}
    if len(raw) != REPETITIONS * len(roster) or counts != expected:
        raise RuntimeError(f"incomplete profile: rows={len(raw)}, counts={counts}")
    reference_weights = frontier._reference_weight_manifest(
        models[REFERENCE_ID], REFERENCE_ID
    )
    atomic_json(reference_weights, STUDY / "results/manifests/openclip_weights.json")
    return _summarize_profile(
        raw,
        {
            **hardware,
            "slurm_job_id": job_id,
            "absolute_raw_path": str(raw_path.resolve()),
        },
    )


def final_report() -> dict[str, Any]:
    _assert_frozen()
    latency = json.loads((STUDY / "results/profile/report.json").read_text())
    latency_rows = {row["arm"]: row for row in latency["rows"]}
    baseline_values = {}
    for arm, path in BASELINES.items():
        _, pipeline = alignment_runner.load_pipeline(path)
        report = json.loads((_artifact_root(pipeline) / "report/report.json").read_text())
        baseline_values[arm] = {
            "validation_mean_R1": float(
                report["flickr_mean_under_flickr_validation_epoch_selection"]
            ),
            "validation_sd": float(report["flickr_validation_selected_sd"]),
            "latency_q3_mean_ms": latency_rows[arm]["q3_mean_ms"],
        }
    parent_for = {
        "frozen_block8": "frozen_no_merge",
        "frozen_block6": "frozen_no_merge",
        "dual_block8": "dual_lora_no_merge",
        "dual_block6": "dual_lora_no_merge",
    }
    rows = []
    for path in PIPELINES:
        _, pipeline = alignment_runner.load_pipeline(path)
        arm = pipeline["teacher_extension"]["arm_id"]
        report = json.loads((_artifact_root(pipeline) / "report/report.json").read_text())
        mean = float(report["flickr_mean_under_flickr_validation_epoch_selection"])
        sd = float(report["flickr_validation_selected_sd"])
        parent = parent_for[arm]
        parent_mean = baseline_values[parent]["validation_mean_R1"]
        parent_sd = baseline_values[parent]["validation_sd"]
        rows.append(
            {
                "arm": arm,
                "parent": parent,
                "merge_after_block": pipeline["tokenshift_training"][
                    "merge_after_block"
                ],
                "validation_mean_R1": mean,
                "validation_sd": sd,
                "validation_min": report["flickr_validation_selected_min"],
                "validation_max": report["flickr_validation_selected_max"],
                "gain_vs_no_merge_parent_pp": (mean - parent_mean) * 100.0,
                "pooled_sd_vs_parent_pp": math.sqrt((sd**2 + parent_sd**2) / 2.0)
                * 100.0,
                "inference_trainable_parameters": pipeline["tokenshift_training"][
                    "expected_inference_trainable_parameters"
                ],
                "added_tokenshift_parameters": 0,
                "latency_q3_mean_ms": latency_rows[arm]["q3_mean_ms"],
                "paired_latency_delta_vs_parent_ms": latency_rows[arm][
                    "paired_delta_vs_parent_mean_ms"
                ],
                "latency_classification_vs_openclip": latency_rows[arm][
                    "classification_vs_openclip"
                ],
                "flickr_test_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    matched = []
    for block in (8, 6):
        frozen = frame[frame["arm"] == f"frozen_block{block}"].iloc[0]
        dual = frame[frame["arm"] == f"dual_block{block}"].iloc[0]
        matched.append(
            {
                "merge_after_block": block,
                "dual_minus_frozen_validation_pp": (
                    float(dual["validation_mean_R1"])
                    - float(frozen["validation_mean_R1"])
                )
                * 100.0,
                "dual_minus_frozen_parameters": int(
                    dual["inference_trainable_parameters"]
                    - frozen["inference_trainable_parameters"]
                ),
            }
        )
    result = {
        "status": "COMPLETE_AWAITING_FROZEN_DECISION",
        "automatic_promotion": False,
        "selection_status": "NOT_SELECTED_BY_CODE",
        "existing_no_merge_baselines": baseline_values,
        "rows": rows,
        "matched_parent_comparisons": matched,
        "interpretation_guard": (
            "Frozen-weight and dual-LoRA TokenShift results are reported separately. "
            "The dual-LoRA arm is a parameter-efficient adaptation extension and "
            "must not be relabelled as a strictly frozen encoder result."
        ),
        "flickr_test_used": False,
    }
    atomic_csv(frame, STUDY / "results/report/summary.csv")
    atomic_csv(pd.DataFrame(matched), STUDY / "results/report/matched_parents.csv")
    atomic_json(result, STUDY / "results/report/report.json")
    lines = [
        "# TokenShift frozen-vs-LoRA result",
        "",
        "No arm was selected automatically. Flickr30k test remained sealed.",
        "",
        "| Arm | validation mean R@1 | gain vs parent (pp) | mean Q3 (ms) | parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {100*row['validation_mean_R1']:.3f}% | "
            f"{row['gain_vs_no_merge_parent_pp']:+.3f} | "
            f"{row['latency_q3_mean_ms']:.4f} | "
            f"{row['inference_trainable_parameters']:,} |"
        )
    atomic_text("\n".join(lines) + "\n", STUDY / "results/report/report.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "smoke",
            "train",
            "verify",
            "evaluate-seed",
            "report-arm",
            "profile",
            "report",
        ),
    )
    parser.add_argument("--pipeline-index", type=int, default=0)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate()
    elif args.command == "smoke":
        result = smoke()
    elif args.command == "train":
        result = train(args.pipeline_index, args.seed_index, not args.no_resume)
    elif args.command == "verify":
        result = verify(args.pipeline_index)
    elif args.command == "evaluate-seed":
        result = evaluate_seed(args.pipeline_index, args.seed_index)
    elif args.command == "report-arm":
        result = report_arm(args.pipeline_index)
    elif args.command == "profile":
        result = profile()
    else:
        result = final_report()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

