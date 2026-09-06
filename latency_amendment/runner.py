from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from freezeshift.model_adaptation import merge_lora_
from freezeshift.runner import _load_selected
from src.alignment_v3 import efficiency_frontier as frontier
from src.alignment_v3.references import REFERENCE_CHECKPOINTS, build_reference
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "latency_amendment"
REFERENCE_ID = "openclip_vit_b32_quickgelu_openai"
PIPELINES = {
    "M_T1": "configs/text_aggregation_study/pipeline_m_t1.yaml",
    "vision": "freezeshift/configs/pipeline_vision.yaml",
    "text": "freezeshift/configs/pipeline_text.yaml",
    "dual": "freezeshift/configs/pipeline_dual.yaml",
}
ROSTER = (*PIPELINES, REFERENCE_ID)
REPETITIONS = 10


def _package_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(STUDY.rglob("*")):
        if not path.is_file() or any(
            part in {"results", "logs", "__pycache__"} for part in path.parts
        ):
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    import open_clip
    import yaml

    amendment = yaml.safe_load((STUDY / "amendment.yaml").read_text())
    if not amendment.get("declared_after_observation"):
        raise RuntimeError("post-observation amendment disclosure is missing")
    if not amendment.get("historical_result", {}).get("preserved"):
        raise RuntimeError("original FreezeShift latency verdict was not preserved")
    if not amendment.get("flickr_test_sealed"):
        raise RuntimeError("Flickr test seal is absent")
    expected_reference = ("ViT-B-32-quickgelu", "openai")
    if REFERENCE_CHECKPOINTS.get(REFERENCE_ID) != expected_reference:
        raise RuntimeError(
            f"OpenCLIP registry changed: {REFERENCE_CHECKPOINTS.get(REFERENCE_ID)}"
        )
    if str(open_clip.__version__) != "3.3.0":
        raise RuntimeError(f"expected open_clip 3.3.0, got {open_clip.__version__}")
    old = json.loads((ROOT / "freezeshift/results/report/latency.json").read_text())
    if not old.get("same_allocation"):
        raise RuntimeError("original FreezeShift latency provenance is incomplete")
    old_rows = {row["arm"]: row for row in old["rows"]}
    if abs(float(old_rows["M_T1"]["q3_ms"]) - 9.497033664956689) > 1e-12:
        raise RuntimeError("amendment trigger no longer matches the recorded M_T1 value")
    final = json.loads((ROOT / "freezeshift/results/report/report.json").read_text())
    if final.get("status") != "COMPLETE" or final.get("flickr_test_used") is not False:
        raise RuntimeError("FreezeShift is incomplete or its test seal is invalid")
    receipt = {
        "status": "READY",
        "package_sha256": _package_hash(),
        "declared_after_observation": True,
        "measurement_invalidity_rationale": amendment["observation_trigger"]["defect"],
        "original_verdict_preserved": amendment["historical_result"]["statement"],
        "roster": list(ROSTER),
        "independent_repetitions": REPETITIONS,
        "new_training_runs": 0,
        "reference_registry": {
            "id": REFERENCE_ID,
            "model_name": expected_reference[0],
            "pretrained": expected_reference[1],
            "open_clip_version": open_clip.__version__,
        },
        "original_rows": old["rows"],
        "flickr_test_sealed": True,
    }
    atomic_json(receipt, STUDY / "results/manifests/design.json")
    return receipt


def _assert_frozen() -> dict[str, Any]:
    path = STUDY / "results/manifests/design.json"
    if not path.is_file():
        raise RuntimeError("run latency-amendment validate first")
    receipt = json.loads(path.read_text())
    if receipt["package_sha256"] != _package_hash():
        raise RuntimeError("latency-amendment package changed after validation")
    return receipt


def _load_models(device: torch.device):
    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, dict[str, Any] | None] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for arm, pipeline in PIPELINES.items():
        model, config, summary, epoch = _load_selected(pipeline, 42, device)
        merged = merge_lora_(model)
        model.eval()
        frontier._assert_fully_eval(model, f"latency amendment {arm}")
        models[arm] = model
        configs[arm] = config
        metadata[arm] = {
            "selected_seed": 42,
            "selected_epoch": epoch,
            "merged_lora_modules": len(merged),
            "inference_trainable_parameters": summary[
                "params_trainable_inference"
            ],
        }
    reference = build_reference(REFERENCE_ID).to(device).eval()
    frontier._assert_fully_eval(reference, "latency amendment OpenCLIP")
    models[REFERENCE_ID] = reference
    configs[REFERENCE_ID] = None
    metadata[REFERENCE_ID] = {
        "selected_seed": None,
        "selected_epoch": None,
        "merged_lora_modules": 0,
        "inference_trainable_parameters": 0,
    }
    return models, configs, metadata


def _prepare_workloads(models, configs, device: torch.device):
    pil_images, captions = frontier._sample_batch(
        ROOT / "data/flickr30k/validation.csv", 64
    )
    workloads = {}
    for arm in ROSTER:
        model = models[arm]
        config = configs[arm]
        transform = model.preprocess if config is None else frontier._student_transform(config)
        images = torch.stack([transform(image) for image in pil_images]).to(device)
        cpu_tokens = frontier._tokenize_native(
            model, captions, pad_to_native_context=False
        )
        tokens = cpu_tokens.to(device)
        workloads[arm] = {
            "images": images,
            "tokens": tokens,
            "token_stats": frontier._token_shape_and_utilization(cpu_tokens),
            "image_resolution": int(model.image_size) if config is None else int(config["data"]["image_size"]),
        }
    return workloads


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _classification(low: float, high: float) -> str:
    if high < 0:
        return "FASTER"
    if low > 0:
        return "SLOWER"
    return "LATENCY_EQUIVALENT"


def _summarize(raw: pd.DataFrame, hardware: dict[str, Any]) -> dict[str, Any]:
    indexed = {
        arm: raw[raw["arm"] == arm].set_index("repetition").sort_index()
        for arm in ROSTER
    }
    reference = indexed[REFERENCE_ID]
    baseline = indexed["M_T1"]
    rows = []
    paired_rows = []
    for arm in ROSTER:
        frame = indexed[arm]
        q3 = frame["q3_ms"].to_numpy(dtype=float)
        own_low, own_high = _bootstrap_mean_ci(q3, 20260808 + ROSTER.index(arm))
        row: dict[str, Any] = {
            "arm": arm,
            "independent_repetitions": len(q3),
            "q3_mean_ms": float(q3.mean()),
            "q3_median_ms": float(np.median(q3)),
            "q3_sd_ms": float(q3.std(ddof=1)),
            "q3_min_ms": float(q3.min()),
            "q3_max_ms": float(q3.max()),
            "q3_mean_ci95_low_ms": own_low,
            "q3_mean_ci95_high_ms": own_high,
            "image_q3_mean_ms": float(frame["image_q3_ms"].mean()),
            "text_q3_mean_ms": float(frame["text_q3_ms"].mean()),
        }
        if arm != REFERENCE_ID:
            delta_ref = (
                frame["q3_ms"] - reference["q3_ms"]
            ).to_numpy(dtype=float)
            ref_low, ref_high = _bootstrap_mean_ci(
                delta_ref, 20260908 + ROSTER.index(arm)
            )
            row.update(
                {
                    "paired_delta_vs_openclip_mean_ms": float(delta_ref.mean()),
                    "paired_delta_vs_openclip_ci95_low_ms": ref_low,
                    "paired_delta_vs_openclip_ci95_high_ms": ref_high,
                    "point_estimate_latency_eligible": bool(delta_ref.mean() <= 0),
                    "uncertainty_classification_vs_openclip": _classification(
                        ref_low, ref_high
                    ),
                }
            )
            for repetition, value in zip(frame.index, delta_ref):
                paired_rows.append(
                    {
                        "arm": arm,
                        "reference": REFERENCE_ID,
                        "repetition": int(repetition),
                        "paired_q3_delta_ms": float(value),
                    }
                )
        if arm not in {"M_T1", REFERENCE_ID}:
            delta_base = (
                frame["q3_ms"] - baseline["q3_ms"]
            ).to_numpy(dtype=float)
            base_low, base_high = _bootstrap_mean_ci(
                delta_base, 20261008 + ROSTER.index(arm)
            )
            row.update(
                {
                    "paired_delta_vs_m_t1_mean_ms": float(delta_base.mean()),
                    "paired_delta_vs_m_t1_ci95_low_ms": base_low,
                    "paired_delta_vs_m_t1_ci95_high_ms": base_high,
                    "adaptation_overhead_classification": _classification(
                        base_low, base_high
                    ),
                }
            )
        rows.append(row)
    result = {
        "status": "COMPLETE",
        "amendment_status": "POST_OBSERVATION_MEASUREMENT_REPAIR",
        "original_verdict_preserved": True,
        "primary_reference": REFERENCE_ID,
        "protocol": {
            "same_allocation": True,
            "independent_repetitions": REPETITIONS,
            "warmup_iterations_per_repetition": 20,
            "timed_iterations_per_repetition": 100,
            "batch_size": 64,
            "order_randomized_within_repetition": True,
            "precision": "native_bf16",
        },
        "rows": rows,
        "hardware": hardware,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(rows), STUDY / "results/report/summary.csv")
    atomic_csv(pd.DataFrame(paired_rows), STUDY / "results/report/paired_deltas.csv")
    atomic_json(result, STUDY / "results/report/report.json")
    lines = [
        "# FreezeShift same-allocation latency amendment",
        "",
        "This corrective measurement was declared after the original gate anomaly. ",
        "The original negative verdict remains preserved; this report repairs the ",
        "cross-session comparison rather than silently replacing it.",
        "",
        "| Model | mean Q3 (ms) | paired Δ vs OpenCLIP (ms) | 95% CI | classification |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        delta = row.get("paired_delta_vs_openclip_mean_ms")
        if delta is None:
            delta_text = ci_text = "—"
            classification = "REFERENCE"
        else:
            delta_text = f"{delta:+.4f}"
            ci_text = (
                f"[{row['paired_delta_vs_openclip_ci95_low_ms']:+.4f}, "
                f"{row['paired_delta_vs_openclip_ci95_high_ms']:+.4f}]"
            )
            classification = row["uncertainty_classification_vs_openclip"]
        lines.append(
            f"| {row['arm']} | {row['q3_mean_ms']:.4f} | {delta_text} | "
            f"{ci_text} | {classification} |"
        )
    atomic_text("\n".join(lines) + "\n", STUDY / "results/report/report.md")
    return result


def profile() -> dict[str, Any]:
    _assert_frozen()
    provenance = frontier._hardware_guard({})
    device = provenance.pop("device")
    models, configs, metadata = _load_models(device)
    workloads = _prepare_workloads(models, configs, device)
    job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
    # A partial profile may never be resumed into another allocation: doing so
    # would recreate the exact cross-session defect this amendment repairs.
    raw_path = STUDY / "results/runs" / f"job_{job_id}" / "raw_runs.csv"
    existing = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame()
    records = existing.to_dict("records") if not existing.empty else []
    completed = {(int(row["repetition"]), str(row["arm"])) for row in records}
    rng = random.Random(20260808)
    orders = []
    for _ in range(REPETITIONS):
        order = list(ROSTER)
        rng.shuffle(order)
        orders.append(order)
    for repetition, order in enumerate(orders):
        for order_position, arm in enumerate(order):
            if (repetition, arm) in completed:
                continue
            model = models[arm]
            workload = workloads[arm]
            images = workload["images"]
            tokens = workload["tokens"]
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
                frontier._cuda_times(
                    lambda: model.encode_image(images), 20, 100, device
                )
            )
            text = frontier._quartiles(
                frontier._cuda_times(
                    lambda: frontier._encode_tokens(model, tokens),
                    20,
                    100,
                    device,
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
                    "image_resolution": workload["image_resolution"],
                    "padded_sequence_length": workload["token_stats"][
                        "padded_sequence_length"
                    ],
                    **metadata[arm],
                }
            )
            atomic_csv(pd.DataFrame(records), raw_path)
    raw = pd.DataFrame(records)
    counts = raw.groupby("arm").size().to_dict()
    if len(raw) != REPETITIONS * len(ROSTER) or counts != {
        arm: REPETITIONS for arm in ROSTER
    }:
        raise RuntimeError(f"incomplete latency amendment: rows={len(raw)}, counts={counts}")
    hardware = {
        **provenance,
        "slurm_job_id": job_id,
        "absolute_raw_path": str(raw_path.resolve()),
    }
    reference_weights = frontier._reference_weight_manifest(
        models[REFERENCE_ID], REFERENCE_ID
    )
    atomic_json(
        reference_weights,
        STUDY / "results/manifests/openclip_weights.json",
    )
    return _summarize(raw, hardware)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "profile"))
    args = parser.parse_args()
    result = validate() if args.command == "validate" else profile()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
