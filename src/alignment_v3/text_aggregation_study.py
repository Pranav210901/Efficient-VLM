from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
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
    _token_shape_and_utilization,
    _tokenize_native,
    parameter_counts_student,
)
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.phase15.io_utils import atomic_csv, atomic_json
from src.utils.config import load_config


OUT = ROOT / "results/text_aggregation_study"
PREREG = OUT / "preregistration.json"
CEILING_MS = 9.48
PARAM_CEILING = 5_000_000
CELLS = (
    ("M_T0", "configs/token_aggregator_scale_training/base_c4.yaml", "masked_mean", 128, 4),
    ("M_T1", "configs/text_aggregation_study/base_m_t1.yaml", "learned_query_attention", 128, 4),
    ("M_T1_small", "configs/text_aggregation_study/base_m_t1_small.yaml", "learned_query_attention", 96, 3),
    ("E_T0", "configs/text_aggregation_study/base_e_t0.yaml", "masked_mean", 128, 4),
    ("E_T1", "configs/text_aggregation_study/base_e_t1.yaml", "learned_query_attention", 128, 4),
    ("E_T1_small", "configs/text_aggregation_study/base_e_t1_small.yaml", "learned_query_attention", 96, 3),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_preregistered() -> str:
    payload = json.loads(PREREG.read_text())
    if payload.get("status") != "FROZEN_BEFORE_PROFILING":
        raise RuntimeError("text aggregation preregistration is not frozen")
    if not bool(payload.get("flickr_test_sealed")):
        raise RuntimeError("Flickr test seal is absent")
    return _sha(PREREG)


def _token_lengths(tokens: object) -> list[int]:
    mask = getattr(tokens, "attention_mask", None)
    if mask is None and isinstance(tokens, dict):
        mask = tokens.get("attention_mask")
    if mask is None:
        raise RuntimeError("tokenizer output lacks attention_mask")
    return [int(value) for value in mask.sum(dim=1).tolist()]


def _component_times(model: torch.nn.Module, images: torch.Tensor, tokens: object, device: torch.device) -> dict[str, Any]:
    with torch.inference_mode():
        vision = model.vision_encoder.forward_features(images)
        states, mask = model.text_encoder.forward_token_features(tokens)
        baseline = (states * mask.unsqueeze(-1).to(states.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    timings: dict[str, Any] = {}
    timings["vision_encoder_ms"] = _quartiles(_cuda_times(lambda: model.vision_encoder.forward_features(images), 20, 100, device))
    timings["image_aggregation_ms"] = _quartiles(_cuda_times(lambda: model.token_aggregator(vision.local_tokens, vision.global_feature, vision.spatial_shape), 20, 100, device))
    timings["image_projection_ms"] = _quartiles(_cuda_times(lambda: model.image_projection(vision.global_feature), 20, 100, device))
    timings["text_encoder_ms"] = _quartiles(_cuda_times(lambda: model.text_encoder.forward_token_features(tokens), 20, 100, device))
    if model.text_token_aggregator is None:
        timings["text_aggregation_ms"] = _quartiles(_cuda_times(lambda: (states * mask.unsqueeze(-1).to(states.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1), 20, 100, device))
    else:
        timings["text_aggregation_ms"] = _quartiles(_cuda_times(lambda: model.text_token_aggregator(states, mask, baseline), 20, 100, device))
    timings["text_projection_ms"] = _quartiles(_cuda_times(lambda: model.text_projection(baseline), 20, 100, device))
    timings["warning"] = "Isolated cached-boundary diagnostics; component medians are not additive."
    return timings


def validate() -> dict[str, Any]:
    prereg_sha = _assert_preregistered()
    rows = []
    for cell, config_path, aggregation, dim, heads in CELLS:
        config = load_config(ROOT / config_path)
        recipe = config["recipe"]
        observed = (str(recipe.get("text_token_aggregation", "masked_mean")), int(recipe.get("text_pooler_dim", 128)), int(recipe.get("text_pooler_heads", 4)))
        if observed != (aggregation, dim, heads):
            raise RuntimeError(f"{cell} resolved identity mismatch: {observed}")
        if int(config["data"]["image_size"]) != 224 or int(config["training"]["epochs"]) != 24:
            raise RuntimeError(f"{cell} changed the locked resolution/schedule")
        rows.append({"cell": cell, "config": config_path, "aggregation": aggregation, "pool_dim": dim, "heads": heads})
    result = {"status": "READY", "preregistration_sha256": prereg_sha, "profile_cells": 6, "flickr_test_sealed": True, "rows": rows}
    atomic_json(result, OUT / "manifests/validation.json")
    return result


def profile() -> dict[str, Any]:
    prereg_sha = _assert_preregistered()
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    images, captions = _sample_batch(ROOT / "data/flickr30k/validation.csv", 64)
    rows = []
    component_rows = []
    for index, (cell, config_path, aggregation, dim, heads) in enumerate(CELLS):
        config = load_config(ROOT / config_path)
        # Profile the deployable graph only. Teacher heads are training-only and
        # must never enter either the latency path or inference parameter field.
        config["recipe"]["distillation"] = False
        config["recipe"].pop("teacher_dims", None)
        torch.manual_seed(20260801)
        model = build_model(config).to(device).eval()
        _assert_fully_eval(model, cell)
        transform = _student_transform(config)
        image_tensor = torch.stack([transform(image) for image in images]).to(device)
        cpu_tokens = _tokenize_native(model, captions, pad_to_native_context=False)
        lengths = _token_lengths(cpu_tokens)
        tokens = cpu_tokens.to(device)
        full = _quartiles(_cuda_times(lambda: (model.encode_image(image_tensor), model.encode_text_tokens(tokens)), 20, 100, device))
        image_latency = _quartiles(_cuda_times(lambda: model.encode_image(image_tensor), 20, 100, device))
        text_latency = _quartiles(_cuda_times(lambda: model.encode_text_tokens(tokens), 20, 100, device))
        counts = parameter_counts_student(model)
        trainable = int(model.parameter_summary()["params_trainable_inference"])
        row = {
            "index": index, "cell": cell, "encoder": cell[0], "config": config_path,
            "text_token_aggregation": aggregation, "text_pooler_dim": dim, "text_pooler_heads": heads,
            "full_stack_median_ms": full["median"], "full_stack_q1_ms": full["q1"], "full_stack_q3_ms": full["q3"], "full_stack_iqr_ms": full["iqr"],
            "image_median_ms": image_latency["median"], "text_median_ms": text_latency["median"],
            "inference_trainable_parameters": trainable, "parameter_gate_pass": trainable < PARAM_CEILING,
            "latency_gate_pass": full["q3"] < CEILING_MS, "efficiency_pass": full["q3"] < CEILING_MS and trainable < PARAM_CEILING,
            "token_length_mean": float(pd.Series(lengths).mean()), "token_length_sd": float(pd.Series(lengths).std(ddof=1)),
            "token_length_median": float(pd.Series(lengths).median()), "token_length_p5": float(pd.Series(lengths).quantile(.05)), "token_length_p95": float(pd.Series(lengths).quantile(.95)),
            "token_length_min": min(lengths), "token_length_max": max(lengths),
            "params_total_inference": int(counts["full_stack_inference_parameters"]),
            "query_side_inference_parameters": int(counts["query_side_inference_parameters"]),
            "node": provenance["node"], "gpu_model": provenance["gpu_model"], "precision": "native_bf16",
        }
        rows.append(row)
        component_rows.append({"cell": cell, **_component_times(model, image_tensor, tokens, device)})
        del model, image_tensor, tokens
        torch.cuda.empty_cache()
    report = {
        "status": "COMPLETE", "preregistration_sha256": prereg_sha,
        "protocol": {"batch_size": 64, "warmup": 20, "timed_repeats": 100, "cuda_synchronize": True, "same_allocation": True, "native_bf16": True},
        "latency_ceiling_ms": CEILING_MS, "parameter_ceiling": PARAM_CEILING,
        "node": provenance["node"], "gpu_model": provenance["gpu_model"], "cells": rows,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(rows), OUT / "profiling/profile.csv")
    atomic_json(component_rows, OUT / "profiling/component_latency.json")
    atomic_json({"python": platform.python_version(), "torch": torch.__version__, **provenance}, OUT / "profiling/environment.json")
    atomic_json(report, OUT / "profiling/profile_report.json")
    return report


def select_efficiency_cell(prefix: str, rows: dict[str, dict[str, Any]]) -> str:
    """Frozen primary -> small fallback -> baseline hierarchy."""
    if bool(rows[f"{prefix}_T1"]["efficiency_pass"]):
        return f"{prefix}_T1"
    if bool(rows[f"{prefix}_T1_small"]["efficiency_pass"]):
        return f"{prefix}_T1_small"
    return f"{prefix}_T0"


def gate(encoder: str) -> dict[str, Any]:
    prereg_sha = _assert_preregistered()
    profile_path = OUT / "profiling/profile_report.json"
    payload = json.loads(profile_path.read_text())
    if payload.get("preregistration_sha256") != prereg_sha:
        raise RuntimeError("profile/preregistration hash mismatch")
    prefix = "M" if encoder == "minilm" else "E"
    rows = {row["cell"]: row for row in payload["cells"] if row["cell"].startswith(prefix + "_")}
    primary, fallback, baseline = rows[f"{prefix}_T1"], rows[f"{prefix}_T1_small"], rows[f"{prefix}_T0"]
    selected = select_efficiency_cell(prefix, rows)
    result = {
        "status": "PASS_AGGREGATOR" if selected != f"{prefix}_T0" else "NO_AGGREGATOR_PASSED",
        "encoder": encoder, "selected_cell": selected, "baseline_pass": bool(baseline["efficiency_pass"]),
        "preregistration_sha256": prereg_sha, "profile_sha256": _sha(profile_path),
        "strict_gate": {"q3_lt_ms": CEILING_MS, "inference_trainable_params_lt": PARAM_CEILING},
        "cells": [baseline, primary, fallback],
    }
    atomic_json(result, OUT / f"gates/{encoder}/gate_result.json")
    return result


def verify_gate(encoder: str, purpose: str) -> dict[str, Any]:
    path = OUT / f"gates/{encoder}/gate_result.json"
    value = json.loads(path.read_text())
    if value["preregistration_sha256"] != _assert_preregistered() or value["profile_sha256"] != _sha(OUT / "profiling/profile_report.json"):
        raise RuntimeError("gate receipt is stale or tampered")
    selected = str(value["selected_cell"])
    if purpose == "aggregation" and selected.endswith("T0"):
        raise RuntimeError(f"{encoder}: no aggregation passed; branch intentionally blocked")
    if purpose == "baseline" and not bool(value["baseline_pass"]):
        raise RuntimeError(f"{encoder}: T0 failed the independent efficiency gate")
    return {"status": "VERIFIED", "encoder": encoder, "purpose": purpose, "selected_cell": selected}


def _arm_report(cell: str) -> dict[str, Any]:
    path = OUT / f"training/{cell}/report/report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE" or bool(value.get("flickr_test_used")):
        raise RuntimeError(f"invalid or test-contaminated report: {path}")
    return value


def final_report() -> dict[str, Any]:
    """Apply the frozen accuracy hierarchy after all eligible branches close."""
    prereg_sha = _assert_preregistered()
    mgate = json.loads((OUT / "gates/minilm/gate_result.json").read_text())
    egate = json.loads((OUT / "gates/e5/gate_result.json").read_text())
    m_cell = str(mgate["selected_cell"])
    e_agg_cell = str(egate["selected_cell"])
    m_t0 = {"mean": 0.5383629191, "sd": 0.0028394832, "source": "Probe-1 deterministic baseline"}

    if m_cell == "M_T0":
        m_selected = {**m_t0, "cell": "M_T0", "aggregation_promoted": False}
    else:
        arm = _arm_report(m_cell)
        candidate = {"mean": float(arm["flickr_mean_under_flickr_validation_epoch_selection"]), "sd": float(arm["flickr_validation_selected_sd"])}
        threshold = ((m_t0["sd"] ** 2 + candidate["sd"] ** 2) / 2.0) ** 0.5
        promoted = candidate["mean"] - m_t0["mean"] > threshold
        m_selected = {**(candidate if promoted else m_t0), "cell": m_cell if promoted else "M_T0", "aggregation_promoted": promoted, "gain": candidate["mean"] - m_t0["mean"], "pooled_sd_threshold": threshold}

    e_selected: dict[str, Any] | None = None
    if bool(egate["baseline_pass"]):
        e0_report = _arm_report("E_T0")
        e0 = {"mean": float(e0_report["flickr_mean_under_flickr_validation_epoch_selection"]), "sd": float(e0_report["flickr_validation_selected_sd"]), "cell": "E_T0"}
        e_selected = {**e0, "aggregation_promoted": False}
        if e_agg_cell != "E_T0":
            ea_report = _arm_report(e_agg_cell)
            ea = {"mean": float(ea_report["flickr_mean_under_flickr_validation_epoch_selection"]), "sd": float(ea_report["flickr_validation_selected_sd"]), "cell": e_agg_cell}
            threshold = ((e0["sd"] ** 2 + ea["sd"] ** 2) / 2.0) ** 0.5
            if ea["mean"] - e0["mean"] > threshold:
                e_selected = {**ea, "aggregation_promoted": True, "gain": ea["mean"] - e0["mean"], "pooled_sd_threshold": threshold}
            else:
                e_selected.update({"gain": ea["mean"] - e0["mean"], "pooled_sd_threshold": threshold})

    final = "MiniLM"
    between = None
    if e_selected is not None:
        between = ((m_selected["sd"] ** 2 + e_selected["sd"] ** 2) / 2.0) ** 0.5
        if e_selected["mean"] - m_selected["mean"] > between:
            final = "E5"
    result = {
        "status": "COMPLETE", "preregistration_sha256": prereg_sha,
        "primary_metric": "Flickr30k validation mean bidirectional R@1",
        "minilm_selected": m_selected, "e5_selected": e_selected,
        "between_encoder_pooled_sd_threshold": between,
        "selected_extension": final,
        "main_study_pair_changed": False,
        "directional_regressions_must_be_read_from_arm_reports": True,
        "flickr_test_used": False,
    }
    atomic_json(result, OUT / "report/final_selection.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "profile", "gate", "verify-gate", "final-report"))
    parser.add_argument("--encoder", choices=("minilm", "e5"))
    parser.add_argument("--purpose", choices=("aggregation", "baseline"), default="aggregation")
    args = parser.parse_args()
    if args.command == "validate": result = validate()
    elif args.command == "profile": result = profile()
    elif args.command == "gate":
        if not args.encoder: parser.error("gate requires --encoder")
        result = gate(args.encoder)
    elif args.command == "verify-gate":
        if not args.encoder: parser.error("verify-gate requires --encoder")
        result = verify_gate(args.encoder, args.purpose)
    else: result = final_report()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
