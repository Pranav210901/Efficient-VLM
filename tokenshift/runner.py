from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from freezeshift.model_adaptation import build_model, merge_lora_
from freezeshift.runner import _config as freezeshift_config
from freezeshift.runner import _load_selected
from src.alignment_v3 import efficiency_frontier as frontier
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from tokenshift.token_reduction import (
    expected_layout,
    install_internal_token_reduction,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "tokenshift"
DUAL_PIPELINE = "freezeshift/configs/pipeline_dual.yaml"
CEILING_MS = 9.480
ARMS = (
    ("no_merge", None),
    ("merge_after_block_8", 8),
    ("merge_after_block_6", 6),
)


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


def validate() -> dict[str, Any]:
    predictions = json.loads((STUDY / "predictions.json").read_text())
    if not predictions.get("written_before_any_tokenshift_runs"):
        raise RuntimeError("TokenShift predictions were not frozen before profiling")
    if not predictions.get("flickr_test_sealed"):
        raise RuntimeError("Flickr test seal is absent")
    final = json.loads((ROOT / "freezeshift/results/report/report.json").read_text())
    dual = next(row for row in final["rows"] if row["arm"] == "dual")
    if abs(float(dual["validation_mean_R1"]) - 0.6341551641623179) > 1e-12:
        raise RuntimeError("FreezeShift dual result changed unexpectedly")
    pipeline, config = freezeshift_config(DUAL_PIPELINE, 0)
    signature = {
        "image_size": int(config["data"]["image_size"]),
        "vision_encoder": config["model"]["vision_encoder"],
        "text_encoder": config["model"]["text_encoder"],
        "image_token_aggregation": config["recipe"]["image_token_aggregation"],
        "text_token_aggregation": config["recipe"]["text_token_aggregation"],
    }
    expected = {
        "image_size": 224,
        "vision_encoder": "dinov3_vits16",
        "text_encoder": "all_minilm_l6_v2",
        "image_token_aggregation": "transformer_128",
        "text_token_aggregation": "learned_query_attention",
    }
    if signature != expected:
        raise RuntimeError(f"TokenShift parent signature mismatch: {signature}")
    model = build_model(config, pretrained=False)
    before_names = tuple(model.state_dict())
    before_params = sum(parameter.numel() for parameter in model.parameters())
    install_internal_token_reduction(model.vision_encoder, merge_after_block=8)
    after_names = tuple(model.state_dict())
    after_params = sum(parameter.numel() for parameter in model.parameters())
    if before_names != after_names or before_params != after_params:
        raise RuntimeError("TokenShift unexpectedly changed the state dict or parameter count")
    receipt = {
        "status": "READY",
        "package_sha256": _package_hash(),
        "parent_pipeline": DUAL_PIPELINE,
        "parent_signature": signature,
        "parent_validation_mean_R1": dual["validation_mean_R1"],
        "parent_inference_trainable_parameters": dual[
            "inference_trainable_parameters"
        ],
        "latency_ceiling_q3_ms": CEILING_MS,
        "arms": [
            {
                "arm": arm,
                "merge_after_block": block,
                **expected_layout(block).__dict__,
            }
            for arm, block in ARMS
        ],
        "parameter_delta": after_params - before_params,
        "new_training_runs": 0,
        "flickr_test_sealed": True,
    }
    atomic_json(receipt, STUDY / "results/manifests/design.json")
    return receipt


def _assert_frozen() -> dict[str, Any]:
    path = STUDY / "results/manifests/design.json"
    if not path.is_file():
        raise RuntimeError("run TokenShift validate before profiling")
    receipt = json.loads(path.read_text())
    if receipt["package_sha256"] != _package_hash():
        raise RuntimeError("TokenShift package changed after validation")
    return receipt


def smoke() -> dict[str, Any]:
    _assert_frozen()
    _, config = freezeshift_config(DUAL_PIPELINE, 0)
    model = build_model(config, pretrained=False).eval()
    controller = install_internal_token_reduction(
        model.vision_encoder, merge_after_block=None
    )
    images = torch.randn(2, 3, 224, 224)
    rows = []
    with torch.inference_mode():
        controller.set_merge_after_block(None)
        baseline = model.vision_encoder.forward_features(images)
        for arm, block in ARMS:
            controller.set_merge_after_block(block)
            features = model.vision_encoder.forward_features(images)
            expected_patches = 196 if block is None else 49
            expected_spatial = (14, 14) if block is None else (7, 7)
            if features.local_tokens.shape != (2, expected_patches, 384):
                raise RuntimeError(
                    f"{arm} local-token shape is {tuple(features.local_tokens.shape)}"
                )
            if features.spatial_shape != expected_spatial:
                raise RuntimeError(f"{arm} spatial shape is {features.spatial_shape}")
            image_embedding = model.encode_image(images)
            if image_embedding.shape != (2, 384):
                raise RuntimeError(f"{arm} embedding shape is {image_embedding.shape}")
            rows.append(
                {
                    "arm": arm,
                    "merge_after_block": block,
                    "patch_tokens": expected_patches,
                    "total_tokens": expected_patches + 5,
                    "embedding_shape": list(image_embedding.shape),
                    "baseline_feature_shape": list(baseline.local_tokens.shape),
                }
            )
    result = {"status": "COMPLETE", "rows": rows}
    atomic_json(result, STUDY / "results/manifests/smoke.json")
    return result


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _summarize(raw: pd.DataFrame, provenance: dict[str, Any]) -> dict[str, Any]:
    rows = []
    baseline = raw[raw["arm"] == "no_merge"].set_index("repetition")
    paired_rows = []
    for arm, block in ARMS:
        frame = raw[raw["arm"] == arm].sort_values("repetition")
        q3 = frame["q3_ms"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(q3, seed=20260808 + (block or 0))
        row = {
            "arm": arm,
            "merge_after_block": block,
            "independent_runs": int(len(frame)),
            "q3_mean_ms": float(q3.mean()),
            "q3_median_ms": float(np.median(q3)),
            "q3_sd_ms": float(q3.std(ddof=1)),
            "q3_min_ms": float(q3.min()),
            "q3_max_ms": float(q3.max()),
            "q3_mean_bootstrap_ci95_low_ms": ci_low,
            "q3_mean_bootstrap_ci95_high_ms": ci_high,
            "image_q3_mean_ms": float(frame["image_q3_ms"].mean()),
            "median_repetition_q3_below_9p48": bool(np.median(q3) < CEILING_MS),
            "all_repetitions_q3_below_9p48": bool((q3 < CEILING_MS).all()),
        }
        if arm != "no_merge":
            paired = frame.set_index("repetition")["q3_ms"] - baseline["q3_ms"]
            values = paired.to_numpy(dtype=float)
            delta_low, delta_high = _bootstrap_mean_ci(
                values, seed=20260818 + int(block)
            )
            row.update(
                {
                    "paired_q3_delta_mean_ms": float(values.mean()),
                    "paired_q3_delta_sd_ms": float(values.std(ddof=1)),
                    "paired_q3_delta_ci95_low_ms": delta_low,
                    "paired_q3_delta_ci95_high_ms": delta_high,
                }
            )
            paired_rows.extend(
                {
                    "arm": arm,
                    "repetition": int(index),
                    "q3_delta_vs_no_merge_ms": float(value),
                }
                for index, value in paired.items()
            )
        rows.append(row)
    summary = {
        "status": "COMPLETE",
        "stage": "PROFILE_ONLY",
        "automatic_training_decision": False,
        "protocol": {
            "batch_size": 64,
            "warmup_per_independent_run": 20,
            "timed_iterations_per_independent_run": 100,
            "independent_runs_per_arm": 10,
            "order_randomized_within_repetition": True,
            "precision": "native_bf16",
            "input_resolution": 224,
            "latency_ceiling_q3_ms": CEILING_MS,
        },
        "rows": rows,
        "paired_rows": paired_rows,
        "hardware": provenance,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(rows), STUDY / "results/profile/summary.csv")
    atomic_csv(pd.DataFrame(paired_rows), STUDY / "results/profile/paired_deltas.csv")
    atomic_json(summary, STUDY / "results/profile/report.json")
    lines = [
        "# TokenShift latency profile",
        "",
        "This is a profile-only result. No training decision was applied automatically.",
        "The input remained 224×224; 2×2 internal merging reduced 196 patches to 49.",
        "",
        "| Arm | mean Q3 (ms) | 95% bootstrap CI | paired Δ vs control (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        delta = row.get("paired_q3_delta_mean_ms")
        lines.append(
            f"| {row['arm']} | {row['q3_mean_ms']:.4f} | "
            f"[{row['q3_mean_bootstrap_ci95_low_ms']:.4f}, "
            f"{row['q3_mean_bootstrap_ci95_high_ms']:.4f}] | "
            f"{'—' if delta is None else f'{delta:+.4f}'} |"
        )
    atomic_text("\n".join(lines) + "\n", STUDY / "results/profile/report.md")
    return summary


def profile() -> dict[str, Any]:
    _assert_frozen()
    smoke_path = STUDY / "results/manifests/smoke.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("TokenShift smoke test must complete before profiling")
    hardware = frontier._hardware_guard({})
    device = hardware.pop("device")
    model, config, parameter_summary, selected_epoch = _load_selected(
        DUAL_PIPELINE, 42, device
    )
    controller = install_internal_token_reduction(
        model.vision_encoder, merge_after_block=None
    )
    merged_modules = merge_lora_(model)
    model.eval()
    frontier._assert_fully_eval(model, "TokenShift profile")

    pil_images, captions = frontier._sample_batch(
        ROOT / "data/flickr30k/validation.csv", 64
    )
    transform = frontier._student_transform(config)
    images = torch.stack([transform(image) for image in pil_images]).to(device)
    tokens = model.text_encoder.tokenize(captions).to(device)
    raw_path = STUDY / "results/profile/raw_runs.csv"
    completed = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame()
    records = completed.to_dict("records") if not completed.empty else []
    completed_keys = {
        (int(row["repetition"]), str(row["arm"])) for row in records
    }
    rng = random.Random(20260808)
    orders: list[list[tuple[str, int | None]]] = []
    for _ in range(10):
        order = list(ARMS)
        rng.shuffle(order)
        orders.append(order)

    for repetition, order in enumerate(orders):
        for order_position, (arm, block) in enumerate(order):
            if (repetition, arm) in completed_keys:
                continue
            controller.set_merge_after_block(block)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                features = model.vision_encoder.forward_features(images[:2])
                expected_patches = 196 if block is None else 49
                if features.local_tokens.shape[1] != expected_patches:
                    raise RuntimeError(f"runtime patch-token guard failed for {arm}")
                full = frontier._quartiles(
                    frontier._cuda_times(
                        lambda: (
                            model.encode_image(images),
                            model.encode_text_tokens(tokens),
                        ),
                        20,
                        100,
                        device,
                    )
                )
                image = frontier._quartiles(
                    frontier._cuda_times(
                        lambda: model.encode_image(images), 20, 100, device
                    )
                )
            records.append(
                {
                    "repetition": repetition,
                    "order_position": order_position,
                    "arm": arm,
                    "merge_after_block": block,
                    "input_resolution": 224,
                    "patch_tokens_before": 196,
                    "patch_tokens_after": expected_patches,
                    "total_tokens_after": expected_patches + 5,
                    "median_ms": full["median"],
                    "q1_ms": full["q1"],
                    "q3_ms": full["q3"],
                    "image_median_ms": image["median"],
                    "image_q1_ms": image["q1"],
                    "image_q3_ms": image["q3"],
                    "selected_seed": 42,
                    "selected_epoch": selected_epoch,
                    "merged_lora_modules": len(merged_modules),
                    "parent_inference_trainable_parameters": parameter_summary[
                        "params_trainable_inference"
                    ],
                }
            )
            atomic_csv(pd.DataFrame(records), raw_path)
    raw = pd.DataFrame(records)
    expected_count = 10 * len(ARMS)
    if len(raw) != expected_count or raw.groupby("arm").size().to_dict() != {
        arm: 10 for arm, _ in ARMS
    }:
        raise RuntimeError(f"incomplete TokenShift profile: {len(raw)}/{expected_count}")
    provenance = {
        **hardware,
        "slurm_job_id": str(__import__("os").environ.get("SLURM_JOB_ID", "local")),
        "absolute_raw_path": str(raw_path.resolve()),
    }
    return _summarize(raw, provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "smoke", "profile"))
    args = parser.parse_args()
    if args.command == "validate":
        result = validate()
    elif args.command == "smoke":
        result = smoke()
    else:
        result = profile()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

