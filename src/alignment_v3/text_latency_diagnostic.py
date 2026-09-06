from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _assert_fully_eval,
    _cpu_times,
    _cuda_times,
    _hardware_guard,
    _quartiles,
    _sample_batch,
    _student_transform,
    _token_shape_and_utilization,
    _tokenize_native,
    parameter_counts_student,
)
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.alignment_v3.text_aggregation_study import CEILING_MS
from src.phase15.io_utils import atomic_csv, atomic_json
from src.utils.config import load_config


OUT = ROOT / "results/text_aggregation_study/diagnostics/e5_latency"
CELLS = {
    "M_T0": "configs/token_aggregator_scale_training/base_c4.yaml",
    "E_T0": "configs/text_aggregation_study/base_e_t0.yaml",
}


def _tensor_dtypes(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        found.add(str(value.dtype).replace("torch.", ""))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.update(_tensor_dtypes(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_tensor_dtypes(item))
    elif hasattr(value, "to_tuple"):
        found.update(_tensor_dtypes(value.to_tuple()))
    return sorted(found)


def _dtype_trace(model: torch.nn.Module, tokens: object, device: torch.device) -> list[dict[str, Any]]:
    encoder = model.text_encoder.encoder
    modules: list[tuple[str, torch.nn.Module]] = []
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is not None:
        modules.append(("embeddings", embeddings))
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is not None:
        modules.extend((f"encoder.layer.{index}", layer) for index, layer in enumerate(layers))
    final_norm = getattr(encoder, "LayerNorm", None)
    if final_norm is not None:
        modules.append(("final_layer_norm", final_norm))

    records: list[dict[str, Any]] = []
    handles = []
    for name, module in modules:
        def hook(_module: torch.nn.Module, inputs: Any, output: Any, *, label: str = name) -> None:
            records.append({
                "module": label,
                "input_dtypes": _tensor_dtypes(inputs),
                "output_dtypes": _tensor_dtypes(output),
            })
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            states, mask = model.text_encoder.forward_token_features(tokens)
            pooled = (states * mask.unsqueeze(-1).to(states.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            projected = model.text_projection(pooled)
        records.extend([
            {"module": "text_encoder_output", "input_dtypes": [], "output_dtypes": _tensor_dtypes(states)},
            {"module": "masked_mean_pool", "input_dtypes": _tensor_dtypes(states), "output_dtypes": _tensor_dtypes(pooled)},
            {"module": "text_projection", "input_dtypes": _tensor_dtypes(pooled), "output_dtypes": _tensor_dtypes(projected)},
        ])
    finally:
        for handle in handles:
            handle.remove()
    for row in records:
        row["bf16_output"] = row["output_dtypes"] == ["bfloat16"]
    return records


def _profile_cell(cell: str, config_path: str, images: list[Any], captions: list[str], device: torch.device) -> dict[str, Any]:
    config = load_config(ROOT / config_path)
    config["recipe"]["distillation"] = False
    config["recipe"].pop("teacher_dims", None)
    torch.manual_seed(20260801)
    model = build_model(config).to(device).eval()
    _assert_fully_eval(model, cell)
    transform = _student_transform(config)
    image_tensor = torch.stack([transform(image) for image in images]).to(device)

    tokenize = lambda: _tokenize_native(model, captions, pad_to_native_context=False)
    tokenization = _quartiles(_cpu_times(tokenize, 20, 100))
    cpu_tokens = tokenize()
    token_stats = _token_shape_and_utilization(cpu_tokens)
    tokens = cpu_tokens.to(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vision = model.vision_encoder.forward_features(image_tensor)
        states, mask = model.text_encoder.forward_token_features(tokens)
        pooled = (states * mask.unsqueeze(-1).to(states.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    stages = {
        "vision_encoder": _quartiles(_cuda_times(lambda: model.vision_encoder.forward_features(image_tensor), 20, 100, device)),
        "image_aggregation": _quartiles(_cuda_times(lambda: model.token_aggregator(vision.local_tokens, vision.global_feature, vision.spatial_shape), 20, 100, device)),
        "image_projection": _quartiles(_cuda_times(lambda: model.image_projection(vision.global_feature), 20, 100, device)),
        "text_encoder": _quartiles(_cuda_times(lambda: model.text_encoder.forward_token_features(tokens), 20, 100, device)),
        "text_aggregation": _quartiles(_cuda_times(lambda: (states * mask.unsqueeze(-1).to(states.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1), 20, 100, device)),
        "text_projection": _quartiles(_cuda_times(lambda: model.text_projection(pooled), 20, 100, device)),
    }
    full = _quartiles(_cuda_times(lambda: (model.encode_image(image_tensor), model.encode_text_tokens(tokens)), 20, 100, device))
    for values in stages.values():
        values["percent_of_full_stack_q3"] = 100.0 * values["q3"] / full["q3"]
    tokenization["percent_of_full_stack_q3"] = 100.0 * tokenization["q3"] / full["q3"]

    result = {
        "cell": cell,
        "config": config_path,
        "full_stack_neural_latency_ms": full,
        "cpu_tokenization_latency_ms": tokenization,
        "component_latency_ms": stages,
        "component_values_are_isolated_and_non_additive": True,
        "tokenizer": {
            "native_max_sequence_length": int(model.text_encoder.native_context_length),
            "padding_policy": "dynamic_batch_max",
            **token_stats,
        },
        "precision_trace": _dtype_trace(model, tokens, device),
        "parameter_counts": parameter_counts_student(model),
    }
    result["non_bf16_boundaries"] = [
        row for row in result["precision_trace"] if not row["bf16_output"]
    ]
    return result


def _markdown(payload: dict[str, Any]) -> str:
    cells = {row["cell"]: row for row in payload["cells"]}
    lines = [
        "# E5 baseline component-level latency diagnostic",
        "",
        "Diagnostic only: no training, checkpoint mutation, re-gating, or ceiling change.",
        "",
        "## 1. Component latency breakdown",
        "",
        "Isolated component timings use cached module boundaries and are not additive. Percentages use each cell's full-stack neural Q3 denominator.",
        "",
        "| Stage | MiniLM M-T0 Q3 (ms) | M-T0 % | E5 E-T0 Q3 (ms) | E-T0 % |",
        "|---|---:|---:|---:|---:|",
    ]
    names = ["cpu_tokenization"] + list(cells["M_T0"]["component_latency_ms"])
    for name in names:
        key = "cpu_tokenization_latency_ms" if name == "cpu_tokenization" else "component_latency_ms"
        m = cells["M_T0"][key] if name == "cpu_tokenization" else cells["M_T0"][key][name]
        e = cells["E_T0"][key] if name == "cpu_tokenization" else cells["E_T0"][key][name]
        lines.append(f"| {name} | {m['q3']:.4f} | {m['percent_of_full_stack_q3']:.2f}% | {e['q3']:.4f} | {e['percent_of_full_stack_q3']:.2f}% |")
    lines.extend([
        f"| **Full-stack neural** | **{cells['M_T0']['full_stack_neural_latency_ms']['q3']:.4f}** | **100%** | **{cells['E_T0']['full_stack_neural_latency_ms']['q3']:.4f}** | **100%** |",
        "", "## 2. Padding and sequence-length audit", "",
        "| Item | MiniLM M-T0 | E5 E-T0 |", "|---|---:|---:|",
    ])
    for label, key in [
        ("Configured maximum", "native_max_sequence_length"),
        ("Padded batch length", "padded_sequence_length"),
        ("Real-token mean", "raw_token_length_mean"),
        ("Real-token median", "raw_token_length_median"),
        ("Real-token maximum", "raw_token_length_max"),
    ]:
        lines.append(f"| {label} | {cells['M_T0']['tokenizer'][key]} | {cells['E_T0']['tokenizer'][key]} |")
    lines.append(f"| Padding policy | {cells['M_T0']['tokenizer']['padding_policy']} | {cells['E_T0']['tokenizer']['padding_policy']} |")
    lines.extend(["", "## 3. Precision-path audit", ""])
    for cell in ("M_T0", "E_T0"):
        lines.extend([f"### {cell}", "", "| Boundary | Input dtype(s) | Output dtype(s) | BF16 output |", "|---|---|---|---:|"])
        for row in cells[cell]["precision_trace"]:
            lines.append(f"| {row['module']} | {', '.join(row['input_dtypes']) or 'non-floating'} | {', '.join(row['output_dtypes']) or 'non-floating'} | {row['bf16_output']} |")
        lines.append("")
    lines.extend(["## 4. Side-by-side summary", "", f"Frozen ceiling: **{payload['latency_ceiling_ms']:.3f} ms Q3**.", ""])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    images, captions = _sample_batch(ROOT / "data/flickr30k/validation.csv", 64)
    cells = [_profile_cell(cell, path, images, captions, device) for cell, path in CELLS.items()]
    payload = {
        "status": "COMPLETE",
        "diagnostic_only": True,
        "latency_ceiling_ms": CEILING_MS,
        "protocol": {"batch_size": 64, "warmup": 20, "timed_repeats": 100, "dynamic_padding": True},
        **provenance,
        "cells": cells,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, OUT / "report.json")
    rows = []
    for cell in cells:
        for stage, timing in cell["component_latency_ms"].items():
            rows.append({"cell": cell["cell"], "stage": stage, **timing})
        rows.append({"cell": cell["cell"], "stage": "cpu_tokenization", **cell["cpu_tokenization_latency_ms"]})
    atomic_csv(pd.DataFrame(rows), OUT / "component_latency.csv")
    (OUT / "report.md").write_text(_markdown(payload) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
