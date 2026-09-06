from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _assert_fully_eval,
    _cuda_times,
    _hardware_guard,
    _quartiles,
    _sample_batch,
    _student_transform,
    _tokenize_native,
    parameter_counts_student,
)
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.alignment_v3.token_projection_profile import (
    BASELINE_HISTORICAL_MS,
    OPENCLIP_CEILING_MS,
    _load_export,
    _safe_profile_flops,
)
from src.phase15.io_utils import atomic_csv, atomic_json


OUTPUT = ROOT / "results/token_aggregator_scale_profile"
RANDOM_SEED = 20260730
VARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "baseline_cls", "aggregation": "cls"},
    {
        "id": "C1_d128_h4_b1_ff256",
        "aggregation": "transformer_128",
        "dim": 128,
        "heads": 4,
        "blocks": 1,
        "ff_dim": 256,
        "native_identity": False,
    },
    {
        "id": "C2_d256_h8_b1_ff512",
        "aggregation": "transformer_128",
        "dim": 256,
        "heads": 8,
        "blocks": 1,
        "ff_dim": 512,
        "native_identity": False,
    },
    {
        "id": "C3_d128_h4_b2_ff256",
        "aggregation": "transformer_128",
        "dim": 128,
        "heads": 4,
        "blocks": 2,
        "ff_dim": 256,
        "native_identity": False,
    },
    {
        "id": "C4_d256_h8_b2_ff512",
        "aggregation": "transformer_128",
        "dim": 256,
        "heads": 8,
        "blocks": 2,
        "ff_dim": 512,
        "native_identity": False,
    },
    {
        "id": "C5_d384_h6_b1_ff768_native",
        "aggregation": "transformer_128",
        "dim": 384,
        "heads": 6,
        "blocks": 1,
        "ff_dim": 768,
        "native_identity": True,
    },
)


def _build(
    export: dict[str, Any],
    base_config: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    config = deepcopy(base_config)
    recipe = config["recipe"]
    recipe["image_token_aggregation"] = spec["aggregation"]
    if spec["aggregation"] != "cls":
        recipe.update(
            {
                "token_pooler_dim": int(spec["dim"]),
                "token_pooler_heads": int(spec["heads"]),
                "token_pooler_blocks": int(spec["blocks"]),
                "token_pooler_ff_dim": int(spec["ff_dim"]),
                "token_pooler_native_width_identity": bool(
                    spec["native_identity"]
                ),
            }
        )
    torch.manual_seed(RANDOM_SEED)
    model = build_model(config).to(device).eval()
    missing, unexpected = model.load_state_dict(export["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"{spec['id']} unexpected state keys: {unexpected}")
    expected_missing = (
        []
        if spec["aggregation"] == "cls"
        else [
            name
            for name, _ in model.named_parameters()
            if name.startswith("token_aggregator.")
        ]
    )
    if sorted(missing) != sorted(expected_missing):
        raise RuntimeError(
            f"{spec['id']} initialization mismatch: {missing} != {expected_missing}"
        )
    base_params = int(export["parameter_summary"]["params_total_inference"])
    added = sum(parameter.numel() for parameter in model.parameters()) - base_params
    return model, int(added)


def profile() -> dict[str, Any]:
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    export, base_config, fingerprint = _load_export(device)
    images, captions = _sample_batch(
        ROOT / "data/flickr30k/validation.csv", 64
    )
    transform = _student_transform(base_config)
    image_tensor = torch.stack([transform(image) for image in images]).to(device)
    rows = []
    for spec in VARIANTS:
        model, added = _build(export, base_config, spec, device)
        _assert_fully_eval(model, spec["id"])
        tokens = _tokenize_native(
            model, captions, pad_to_native_context=False
        ).to(device)
        image_times = _cuda_times(
            lambda: model.encode_image(image_tensor), 20, 100, device
        )
        text_times = _cuda_times(
            lambda: model.encode_text_tokens(tokens), 20, 100, device
        )
        full_times = _cuda_times(
            lambda: (
                model.encode_image(image_tensor),
                model.encode_text_tokens(tokens),
            ),
            20,
            100,
            device,
        )
        latency = _quartiles(full_times)
        parameters = parameter_counts_student(model)
        flops = _safe_profile_flops(model, image_tensor, tokens, device)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable >= 5_000_000:
            raise RuntimeError(f"{spec['id']} exceeds the sub-5M budget")
        payload = {
            "status": "COMPLETE",
            **spec,
            "random_aggregator_seed": (
                None if spec["aggregation"] == "cls" else RANDOM_SEED
            ),
            "training_performed": False,
            "batch_size": 64,
            "warmup_iterations": 20,
            "timed_repeats": 100,
            "synchronize_each_repeat": True,
            "precision": provenance["precision"],
            "gpu_model": provenance["gpu_model"],
            "node": provenance["node"],
            "checkpoint_fingerprint": fingerprint.digest,
            "register_tokens_included": False,
            "patch_tokens": 196,
            "added_inference_parameters": added,
            "total_inference_trainable_parameters": trainable,
            **parameters,
            **flops,
            "full_stack_neural_latency_ms": latency,
            "image_neural_latency_ms": _quartiles(image_times),
            "text_neural_latency_ms": _quartiles(text_times),
            "historical_baseline_median_ms": BASELINE_HISTORICAL_MS,
            "openclip_ceiling_ms": OPENCLIP_CEILING_MS,
            "median_below_ceiling": latency["median"] < OPENCLIP_CEILING_MS,
            "upper_quartile_below_ceiling": latency["q3"] < OPENCLIP_CEILING_MS,
        }
        atomic_json(payload, OUTPUT / f"{spec['id']}.json")
        rows.append(payload)
        del model, tokens
        torch.cuda.empty_cache()

    baseline = rows[0]["full_stack_neural_latency_ms"]["median"]
    summary = pd.DataFrame(
        [
            {
                "candidate": row["id"],
                "added_inference_parameters": row["added_inference_parameters"],
                "total_inference_trainable_parameters": row[
                    "total_inference_trainable_parameters"
                ],
                "median_ms": row["full_stack_neural_latency_ms"]["median"],
                "q1_ms": row["full_stack_neural_latency_ms"]["q1"],
                "q3_ms": row["full_stack_neural_latency_ms"]["q3"],
                "iqr_ms": row["full_stack_neural_latency_ms"]["iqr"],
                "delta_from_contemporaneous_cls_ms": (
                    row["full_stack_neural_latency_ms"]["median"] - baseline
                ),
                "upper_quartile_below_9p48": row[
                    "upper_quartile_below_ceiling"
                ],
            }
            for row in rows
        ]
    )
    atomic_csv(summary, OUTPUT / "profiles.csv")
    result = {
        "status": "COMPLETE",
        "training_performed": False,
        "protocol": {
            "same_allocation": True,
            "batch_size": 64,
            "warmup": 20,
            "timed_repeats": 100,
            "cuda_synchronize": True,
            **provenance,
        },
        "contemporaneous_cls": rows[0],
        "candidates": rows[1:],
        "openclip_ceiling_ms": OPENCLIP_CEILING_MS,
        "flickr_test_used": False,
    }
    atomic_json(result, OUTPUT / "report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("profile",))
    args = parser.parse_args()
    print(json.dumps(profile(), indent=2))


if __name__ == "__main__":
    main()

