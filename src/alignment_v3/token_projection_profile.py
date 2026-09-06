from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _assert_fully_eval,
    _cuda_times,
    _hardware_guard,
    _profile_flops,
    _quartiles,
    _sample_batch,
    _student_transform,
    _token_shape_and_utilization,
    _tokenize_native,
    parameter_counts_student,
)
from src.alignment_v3.fingerprint import Fingerprint, read_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.phase15.io_utils import atomic_csv, atomic_json


CHECKPOINT_DIR = (
    ROOT
    / "checkpoints/resolution_distillation_224_long/"
    "sensitivity/distill_strength_1p0__seed_42"
)
OUTPUT = ROOT / "results/token_projection_profile"
BASELINE_HISTORICAL_MS = 8.546178694814444
OPENCLIP_CEILING_MS = 9.48
EXPECTED_ADDITIONS = {
    "cls_mean_concat": 295_296,
    "learned_query_attention": 165_761,
    "transformer_128": 232_065,
}
VARIANTS = (
    ("baseline_cls", "cls"),
    ("A_cls_mean_concat", "cls_mean_concat"),
    ("B_learned_query_attention", "learned_query_attention"),
    ("C_transformer_128", "transformer_128"),
)


def _load_export(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], Fingerprint]:
    path = CHECKPOINT_DIR / "inference.pt"
    fingerprint = read_fingerprint(CHECKPOINT_DIR / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError("selected 224px checkpoint fingerprint is missing")
    export = torch.load(path, map_location=device, weights_only=False)
    exported = Fingerprint.from_dict(export["fingerprint"])
    if exported.digest != fingerprint.digest:
        raise RuntimeError("inference export fingerprint mismatch")
    config = deepcopy(export["config"])
    if bool(config["recipe"].get("distillation", False)):
        raise RuntimeError("profiling source is not a strict inference export")
    return export, config, fingerprint


def _build_variant(
    export: dict[str, Any],
    base_config: dict[str, Any],
    aggregation: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], int]:
    config = deepcopy(base_config)
    config["recipe"]["image_token_aggregation"] = aggregation
    config["recipe"]["token_pooler_dim"] = 128
    config["recipe"]["token_pooler_heads"] = 4
    torch.manual_seed(20260730)
    model = build_model(config).to(device).eval()
    missing, unexpected = model.load_state_dict(export["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected baseline state keys: {unexpected}")
    expected_missing = (
        []
        if aggregation == "cls"
        else [
            name
            for name, _ in model.named_parameters()
            if name.startswith("token_aggregator.")
        ]
    )
    if sorted(missing) != sorted(expected_missing):
        raise RuntimeError(
            f"aggregator initialization mismatch: missing={missing}, "
            f"expected={expected_missing}"
        )
    base_params = int(export["parameter_summary"]["params_total_inference"])
    loaded_params = sum(parameter.numel() for parameter in model.parameters())
    added = loaded_params - base_params
    if aggregation != "cls" and added != EXPECTED_ADDITIONS[aggregation]:
        raise RuntimeError(
            f"{aggregation} parameter mismatch: {added} != "
            f"{EXPECTED_ADDITIONS[aggregation]}"
        )
    return model, config, added


def _safe_profile_flops(
    model: torch.nn.Module,
    images: torch.Tensor,
    tokens: object,
    device: torch.device,
) -> dict[str, Any]:
    """Keep optional FLOP instrumentation from invalidating latency profiling.

    PyTorch 2.8's FlopCounterMode module tracker can fail inside
    MultiheadAttention under inference_mode even though the same graph executes
    and times correctly. Latency is the pre-registered output of this stage;
    unavailable FLOPs are recorded explicitly rather than guessed.
    """
    try:
        return {
            **_profile_flops(model, images, tokens, device),
            "flop_measurement_status": "COMPLETE",
            "flop_measurement_error": None,
        }
    except (AttributeError, RuntimeError) as exc:
        return {
            "image_flops": None,
            "caption_flops": None,
            "normalization_flops_added_per_embedding": None,
            "multiply_add_flops": 2,
            "flop_source": None,
            "flop_measurement_version": "operator_level_v2",
            "flop_measurement_status": "UNAVAILABLE_PROFILER_INCOMPATIBILITY",
            "flop_measurement_error": f"{type(exc).__name__}: {exc}",
        }


def profile() -> dict[str, Any]:
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    export, base_config, fingerprint = _load_export(device)
    transform = _student_transform(base_config)
    batch_size, warmup, repeats = 64, 20, 100
    images, captions = _sample_batch(
        ROOT / "data/flickr30k/validation.csv", batch_size
    )
    image_tensor = torch.stack([transform(image) for image in images]).to(device)

    rows = []
    for label, aggregation in VARIANTS:
        model, config, added = _build_variant(
            export, base_config, aggregation, device
        )
        _assert_fully_eval(model, label)
        cpu_tokens = _tokenize_native(
            model, captions, pad_to_native_context=False
        )
        token_stats = _token_shape_and_utilization(cpu_tokens)
        tokens = cpu_tokens.to(device)
        image_times = _cuda_times(
            lambda: model.encode_image(image_tensor), warmup, repeats, device
        )
        text_times = _cuda_times(
            lambda: model.encode_text_tokens(tokens), warmup, repeats, device
        )
        full_times = _cuda_times(
            lambda: (
                model.encode_image(image_tensor),
                model.encode_text_tokens(tokens),
            ),
            warmup,
            repeats,
            device,
        )
        parameters = parameter_counts_student(model)
        flops = _safe_profile_flops(model, image_tensor, tokens, device)
        full = _quartiles(full_times)
        payload = {
            "status": "COMPLETE",
            "variant": label,
            "image_token_aggregation": aggregation,
            "random_aggregator_seed": 20260730 if aggregation != "cls" else None,
            "training_performed": False,
            "register_tokens_included": False,
            "register_token_scope_reason": (
                "DINOv3 register tokens may contain global storage, but they are "
                "excluded so this experiment isolates treatment of the 196 patch "
                "tokens. Including registers is a plausible untested extension."
            ),
            "batch_size": batch_size,
            "warmup_iterations": warmup,
            "timed_repeats": repeats,
            "synchronize_each_repeat": True,
            "precision": "native_bf16",
            "gpu_model": provenance["gpu_model"],
            "node": provenance["node"],
            "checkpoint_fingerprint": fingerprint.digest,
            "image_resolution": 224,
            "patch_tokens": 196,
            "total_dino_tokens": 201,
            "added_inference_trainable_parameters": added,
            "inference_trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            **parameters,
            **flops,
            **token_stats,
            "full_stack_neural_latency_ms": full,
            "image_neural_latency_ms": _quartiles(image_times),
            "text_neural_latency_ms": _quartiles(text_times),
            "historical_baseline_median_ms": BASELINE_HISTORICAL_MS,
            "openclip_ceiling_ms": OPENCLIP_CEILING_MS,
            "median_delta_from_historical_baseline_ms": (
                full["median"] - BASELINE_HISTORICAL_MS
            ),
            "median_below_openclip_ceiling": full["median"] < OPENCLIP_CEILING_MS,
        }
        atomic_json(payload, OUTPUT / f"{label}.json")
        rows.append(payload)
        del model, tokens
        torch.cuda.empty_cache()

    frame = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "image_token_aggregation": row["image_token_aggregation"],
                "added_inference_trainable_parameters": row[
                    "added_inference_trainable_parameters"
                ],
                "inference_trainable_parameters": row[
                    "inference_trainable_parameters"
                ],
                "image_flops": row["image_flops"],
                "full_stack_median_ms": row["full_stack_neural_latency_ms"]["median"],
                "full_stack_q1_ms": row["full_stack_neural_latency_ms"]["q1"],
                "full_stack_q3_ms": row["full_stack_neural_latency_ms"]["q3"],
                "full_stack_iqr_ms": row["full_stack_neural_latency_ms"]["iqr"],
                "delta_from_contemporaneous_baseline_ms": (
                    row["full_stack_neural_latency_ms"]["median"]
                    - rows[0]["full_stack_neural_latency_ms"]["median"]
                ),
                "median_below_openclip_ceiling": row[
                    "median_below_openclip_ceiling"
                ],
            }
            for row in rows
        ]
    )
    atomic_csv(frame, OUTPUT / "profiles.csv")
    result = {
        "status": "COMPLETE",
        "training_performed": False,
        "protocol": {
            "batch_size": batch_size,
            "warmup_iterations": warmup,
            "timed_repeats": repeats,
            "synchronize_each_repeat": True,
            "precision": "native_bf16",
            "gpu_model": provenance["gpu_model"],
            "node": platform.node(),
        },
        "historical_baseline_median_ms": BASELINE_HISTORICAL_MS,
        "contemporaneous_baseline": rows[0]["full_stack_neural_latency_ms"],
        "openclip_ceiling_ms": OPENCLIP_CEILING_MS,
        "candidates": rows[1:],
        "register_token_scope": rows[0]["register_token_scope_reason"],
        "flickr_test_used": False,
    }
    atomic_json(result, OUTPUT / "report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("profile",))
    args = parser.parse_args()
    if args.command == "profile":
        print(json.dumps(profile(), indent=2))


if __name__ == "__main__":
    main()
