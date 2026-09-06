from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from src.common.configuration_ids import canonical_configuration_id
from src.multitask.checkpoint_selection import VARIANTS, load_frozen_model, resolve_best_checkpoints
from src.multitask.config import VISION_ENCODERS


VISION_FAMILIES = {
    "efficientnet_b0": "efficient_cnn",
    "convnext_tiny": "convnext_cnn",
    "convnextv2_tiny": "convnext_cnn",
    "dinov2_vits14": "plain_vit",
    "swin_tiny": "hierarchical_vit",
}

TEXT_FAMILIES = {
    "all_minilm_l6_v2": "minilm_sentence_transformer",
    "bge_small_en": "bge_embedding_model",
    "e5_small_v2": "e5_embedding_model",
    "distilbert": "masked_language_model",
}


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parameter_stats(module: torch.nn.Module) -> tuple[int, int, int]:
    parameters = list(module.parameters())
    return (
        sum(value.numel() for value in parameters),
        sum(value.numel() for value in parameters if value.requires_grad),
        sum(value.numel() * value.element_size() for value in parameters),
    )


def _flops(fn: Callable[[], object]) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        fn()
    return float(counter.get_total_flops())


def profile_callable(
    fn: Callable[[], object],
    device: str = "cpu",
    warmup: int = 2,
    repeats: int = 5,
    flop_counter: Callable[[], float] | None = None,
    batch_size: int = 1,
) -> dict[str, float | str]:
    if warmup < 0 or repeats < 1 or batch_size < 1:
        raise ValueError("warmup must be non-negative and repeats/batch_size must be positive")
    target = torch.device(device)
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
            torch.cuda.reset_peak_memory_stats(target)
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = (time.perf_counter() - start) / repeats
    result: dict[str, float | str] = {
        "batch_latency_seconds": elapsed,
        "latency_ms": elapsed * 1000.0,
        "throughput_batches_per_second": 1.0 / max(elapsed, 1e-12),
        "throughput_samples_per_second": batch_size / max(elapsed, 1e-12),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0.0,
        "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0.0,
    }
    try:
        value = float(flop_counter()) if flop_counter else None
        result["flops"] = value if value is not None else "unavailable"
        result["flops_per_batch"] = value if value is not None else "unavailable"
        result["macs_estimate_per_batch"] = value / 2.0 if value is not None else "unavailable"
    except Exception as exc:
        result["flops"] = "unavailable"
        result["flops_per_batch"] = "unavailable"
        result["macs_estimate_per_batch"] = "unavailable"
        result["flop_error"] = str(exc)
    return result


def _token_statistics(text_encoder, captions: list[str]) -> tuple[float | str, int | str]:
    if getattr(text_encoder, "is_openclip", False):
        return "unavailable", "unavailable"
    prepared = text_encoder._prepare_captions(captions)
    tokens = text_encoder.tokenizer(prepared, padding=True, truncation=True, return_tensors="pt")
    mask = tokens.get("attention_mask")
    if mask is None:
        return "unavailable", "unavailable"
    lengths = mask.sum(dim=1)
    return float(lengths.float().mean()), int(lengths.max())


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _upsert(rows: list[dict[str, object]], row: dict[str, object], key: str) -> None:
    rows[:] = [value for value in rows if value.get(key) != row.get(key)]
    rows.append(row)


def profile_phase15_efficiency(
    project_root: str | Path,
    device: str | None = None,
    batch_size: int = 8,
    warmup: int = 2,
    repeats: int = 5,
    measure_flops: bool = True,
    resume: bool = True,
    config_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Profile requested aligned pairs plus each distinct frozen encoder.

    Models are loaded from the canonical best checkpoints, one at a time. Rows
    are committed after every model so an interrupted run can resume safely.
    With no explicit ``config_ids``, the original 20-baseline scope is kept for
    backward compatibility. Explicit IDs may include any supported BLF variant.
    """

    root = Path(project_root).resolve()
    output = root / "results/phase15/efficiency"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "vision": output / "vision_encoders.csv",
        "text": output / "text_encoders.csv",
        "pairs": output / "full_pairs.csv",
        "errors": output / "profiling_errors.csv",
        "manifest": output / "profiling_manifest.json",
    }
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested but no CUDA device is visible")
    device_name = torch.cuda.get_device_name(target) if target.type == "cuda" else platform.processor() or "cpu"
    signature_payload = {
        "device": str(target),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "measure_flops": measure_flops,
        "version": 1,
    }
    profile_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()[:20]
    requested = {canonical_configuration_id(value) for value in config_ids} if config_ids is not None else None
    variants = ("baseline",)
    if requested:
        requested_variants = {value.rsplit("__", 1)[-1] for value in requested}
        unknown_variants = requested_variants - set(VARIANTS)
        if unknown_variants:
            raise ValueError(f"Unknown variants in configuration IDs: {sorted(unknown_variants)}")
        variants = tuple(value for value in VARIANTS if value in requested_variants)
    specs = resolve_best_checkpoints(
        root / "checkpoints",
        list(VISION_ENCODERS),
        variants,
        config_ids=requested,
        load_epochs=False,
    )
    if requested is not None:
        missing = requested - {value.config_id for value in specs}
        if missing:
            raise KeyError(f"Unknown or unavailable configuration IDs: {sorted(missing)}")
    if not specs:
        raise FileNotFoundError("No requested best checkpoints are available for efficiency profiling")

    vision_rows = _read_rows(paths["vision"]) if resume else []
    text_rows = _read_rows(paths["text"]) if resume else []
    pair_rows = _read_rows(paths["pairs"]) if resume else []
    error_rows = _read_rows(paths["errors"]) if resume else []
    complete_visions = {
        str(row["vision_encoder"]) for row in vision_rows
        if row.get("status") == "complete" and row.get("profile_signature") == profile_signature
    }
    complete_texts = {
        str(row["text_encoder"]) for row in text_rows
        if row.get("status") == "complete" and row.get("profile_signature") == profile_signature
    }
    complete_pairs = {
        str(row["config_id"]) for row in pair_rows
        if row.get("status") == "complete" and row.get("profile_signature") == profile_signature
    }
    captions = [
        "a photo of a dog running through grass",
        "two people standing beside a red car",
        "an aerial image of agricultural fields",
        "a small bird sitting on a wooden branch",
        "a train passing through a mountain landscape",
        "a close-up photograph of a household object",
        "several boats floating on blue water",
        "a plate of food on a dining table",
    ]
    captions = (captions * ((batch_size + len(captions) - 1) // len(captions)))[:batch_size]

    for position, spec in enumerate(specs, 1):
        need_vision = spec.vision_encoder not in complete_visions
        need_text = spec.text_encoder not in complete_texts
        need_pair = spec.config_id not in complete_pairs
        if not (need_vision or need_text or need_pair):
            print(f"Efficiency {position}/{len(specs)} already complete: {spec.config_id}", flush=True)
            continue
        print(f"Efficiency {position}/{len(specs)} loading: {spec.config_id} on {target}", flush=True)
        model = None
        try:
            model, config, best_epoch = load_frozen_model(spec, str(target))
            image_size = int(config.get("data", {}).get("image_size", 224))
            images = torch.randn(batch_size, 3, image_size, image_size, device=target)
            common = {
                "device": str(target),
                "device_name": device_name,
                "torch_version": torch.__version__,
                "batch_size": batch_size,
                "warmup": warmup,
                "repeats": repeats,
                "profile_signature": profile_signature,
                "status": "complete",
            }
            if need_vision:
                fn = lambda: model.vision_encoder(images)
                stats = profile_callable(
                    fn, str(target), warmup, repeats, (lambda: _flops(fn)) if measure_flops else None, batch_size
                )
                parameters, trainable_parameters, parameter_bytes = _parameter_stats(model.vision_encoder)
                row = {
                    "vision_encoder": spec.vision_encoder,
                    "architecture_family": VISION_FAMILIES.get(spec.vision_encoder, "other"),
                    "representative_config": spec.config_id,
                    "output_dim": int(model.vision_encoder.output_dim),
                    "image_size": image_size,
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "parameter_bytes": parameter_bytes,
                    **common,
                    **stats,
                }
                _upsert(vision_rows, row, "vision_encoder")
                _atomic_to_csv(pd.DataFrame(vision_rows).sort_values("vision_encoder"), paths["vision"])
                complete_visions.add(spec.vision_encoder)
            if need_text:
                fn = lambda: model.text_encoder(captions)
                stats = profile_callable(
                    fn, str(target), warmup, repeats, (lambda: _flops(fn)) if measure_flops else None, batch_size
                )
                parameters, trainable_parameters, parameter_bytes = _parameter_stats(model.text_encoder)
                mean_tokens, max_tokens = _token_statistics(model.text_encoder, captions)
                row = {
                    "text_encoder": spec.text_encoder,
                    "architecture_family": TEXT_FAMILIES.get(spec.text_encoder, "other"),
                    "representative_config": spec.config_id,
                    "output_dim": int(model.text_encoder.output_dim),
                    "mean_tokens": mean_tokens,
                    "max_tokens": max_tokens,
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "parameter_bytes": parameter_bytes,
                    **common,
                    **stats,
                }
                _upsert(text_rows, row, "text_encoder")
                _atomic_to_csv(pd.DataFrame(text_rows).sort_values("text_encoder"), paths["text"])
                complete_texts.add(spec.text_encoder)
            if need_pair:
                fn = lambda: model(images, captions)
                stats = profile_callable(
                    fn, str(target), warmup, repeats, (lambda: _flops(fn)) if measure_flops else None, batch_size
                )
                parameters, trainable_parameters, parameter_bytes = _parameter_stats(model)
                row = {
                    "config_id": spec.config_id,
                    "vision_encoder": spec.vision_encoder,
                    "text_encoder": spec.text_encoder,
                    "variant": spec.variant,
                    "best_epoch": best_epoch,
                    "checkpoint": str(spec.checkpoint),
                    "checkpoint_bytes": spec.checkpoint.stat().st_size,
                    "shared_dim": int(config.get("model", {}).get("shared_dim", 256)),
                    "image_size": image_size,
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "parameter_bytes": parameter_bytes,
                    **common,
                    **stats,
                }
                _upsert(pair_rows, row, "config_id")
                _atomic_to_csv(pd.DataFrame(pair_rows).sort_values("config_id"), paths["pairs"])
                complete_pairs.add(spec.config_id)
            if spec.config_id in complete_pairs:
                remaining_errors = [row for row in error_rows if row.get("config_id") != spec.config_id]
                if len(remaining_errors) != len(error_rows):
                    error_rows = remaining_errors
                    _atomic_to_csv(pd.DataFrame(error_rows), paths["errors"])
        except Exception as exc:
            error = {
                "config_id": spec.config_id,
                "vision_encoder": spec.vision_encoder,
                "text_encoder": spec.text_encoder,
                "device": str(target),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _upsert(error_rows, error, "config_id")
            _atomic_to_csv(pd.DataFrame(error_rows).sort_values("config_id"), paths["errors"])
            print(f"Efficiency profiling failed for {spec.config_id}: {exc}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if target.type == "cuda":
                torch.cuda.empty_cache()

    expected_pair_ids = {spec.config_id for spec in specs}
    completed_expected_pairs = complete_pairs & expected_pair_ids
    summary: dict[str, object] = {
        "status": "complete" if completed_expected_pairs == expected_pair_ids else "incomplete",
        "device": str(target),
        "device_name": device_name,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "measure_flops": measure_flops,
        "profile_signature": profile_signature,
        "vision_encoders_complete": len(complete_visions),
        "text_encoders_complete": len(complete_texts),
        "pairs_complete": len(completed_expected_pairs),
        "pairs_expected": len(specs),
        "pairs_total_in_file": len(complete_pairs),
        "config_ids": sorted(expected_pair_ids),
        "errors": len(error_rows),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    _atomic_json(summary, paths["manifest"])
    return summary
