from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from freezeshift.model_adaptation import build_model, lora_parameter_count, merge_lora_
from src.alignment_v3 import efficiency_frontier as frontier
from src.alignment_v3 import resolution_distillation_trajectory as trajectory
from src.alignment_v3 import runner as alignment_runner
from src.alignment_v3 import training as alignment_training
from src.alignment_v3.fingerprint import read_fingerprint
from src.alignment_v3.training import load_training_checkpoint
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "freezeshift"
PIPELINES = (
    "freezeshift/configs/pipeline_vision.yaml",
    "freezeshift/configs/pipeline_text.yaml",
    "freezeshift/configs/pipeline_dual.yaml",
)
BASELINE_PIPELINE = "configs/text_aggregation_study/pipeline_m_t1.yaml"
BASELINE_REPORT = ROOT / "results/text_aggregation_study/training/M_T1/report/report.json"
PREDICTIONS = STUDY_ROOT / "predictions.json"
CEILING_MS = 9.480
OPENCLIP_VALIDATION = 0.6994082840236686


def _patch_builders() -> None:
    alignment_training.build_model = build_model
    alignment_runner.build_model = build_model
    trajectory.build_model = build_model
    frontier.build_model = build_model


def _package_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in STUDY_ROOT.rglob("*")
        if path.is_file()
        and not any(part in {"results", "checkpoints", "logs", "__pycache__"} for part in path.parts)
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config(path: str, seed_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    _, pipeline = alignment_runner.load_pipeline(path)
    job = alignment_runner.sensitivity_jobs(pipeline)[seed_index]
    return pipeline, alignment_runner.build_job_config(pipeline, job, "sensitivity")


def validate() -> dict[str, Any]:
    predictions = json.loads(PREDICTIONS.read_text())
    if not predictions.get("written_before_any_runs") or not predictions.get("flickr_test_sealed"):
        raise RuntimeError("FreezeShift predictions are absent or test is unsealed")
    baseline = json.loads(BASELINE_REPORT.read_text())
    if baseline.get("status") != "COMPLETE" or baseline.get("flickr_test_used") is not False:
        raise RuntimeError("M_T1 baseline is incomplete or test-contaminated")
    rows = []
    frozen = None
    for path in PIPELINES:
        pipeline, config = _config(path)
        if "test" in str(pipeline["optional_transfer"]["flickr30k_csv"]).lower():
            raise RuntimeError(f"Flickr test path entered {path}")
        model = build_model(config, pretrained=False)
        summary = model.parameter_summary()
        observed_lora = lora_parameter_count(model)
        declared = pipeline["freezeshift"]
        if observed_lora != int(declared["expected_lora_parameters"]):
            raise RuntimeError(f"LoRA count mismatch in {path}: {observed_lora}")
        if summary["params_trainable_inference"] != int(
            declared["expected_inference_trainable_parameters"]
        ):
            raise RuntimeError(f"trainable count mismatch in {path}: {summary}")
        contract = config.get("provenance", {}).get("parameter_contract", {})
        required_contract = {
            "params_trainable_inference",
            "params_trainable_training",
            "params_training_only",
        }
        if not required_contract.issubset(contract):
            raise RuntimeError(
                f"incomplete runtime parameter contract in {path}: {contract}"
            )
        for key, expected in contract.items():
            if key not in summary or int(expected) != int(summary[key]):
                raise RuntimeError(
                    f"runtime parameter contract mismatch in {path} for {key}: "
                    f"contract={expected}, loaded={summary.get(key)}"
                )
        if summary["params_trainable_inference"] >= 5_000_000:
            raise RuntimeError(f"5M budget exceeded in {path}: {summary}")
        base_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (name.startswith("vision_encoder.model.") or name.startswith("text_encoder.encoder."))
            and ".lora_" not in name
        ]
        if base_trainable:
            raise RuntimeError(f"base encoder weights unfrozen in {path}: {base_trainable[:3]}")
        signature = {
            "vision": config["model"]["vision_encoder"],
            "text": config["model"]["text_encoder"],
            "image_size": int(config["data"]["image_size"]),
            "batch": int(config["training"]["batch_size"]),
            "epochs": int(config["training"]["epochs"]),
            "lr": float(config["training"]["lr"]),
            "loss": config["recipe"]["loss_type"],
            "queue": int(config["training"]["memory_queue_size"]),
            "vision_aggregation": config["recipe"]["image_token_aggregation"],
            "text_aggregation": config["recipe"]["text_token_aggregation"],
            "teacher": config["distillation"]["teacher_id"],
        }
        if frozen is None:
            frozen = signature
        elif signature != frozen:
            raise RuntimeError(f"non-adaptation field differs in {path}")
        rows.append(
            {
                "arm": declared["arm"],
                "pipeline": path,
                "lora_parameters": observed_lora,
                **summary,
                "lora_targets": len(model.lora_targets),
                "base_encoder_weights_frozen": True,
            }
        )
    manifest = {
        "status": "READY",
        "package_sha256": _package_hash(),
        "new_training_runs": 9,
        "seeds": [42, 43, 44],
        "frozen_training_signature": frozen,
        "baseline_validation_mean_R1": baseline[
            "flickr_mean_under_flickr_validation_epoch_selection"
        ],
        "openclip_validation_mean_R1": OPENCLIP_VALIDATION,
        "latency_ceiling_q3_ms": CEILING_MS,
        "rows": rows,
        "flickr_test_sealed": True,
    }
    atomic_csv(pd.DataFrame(rows), STUDY_ROOT / "results/manifests/design.csv")
    atomic_json(manifest, STUDY_ROOT / "results/manifests/design.json")
    return manifest


def _assert_frozen_package() -> None:
    path = STUDY_ROOT / "results/manifests/design.json"
    if not path.is_file():
        raise RuntimeError("run validate before executing FreezeShift")
    receipt = json.loads(path.read_text())
    if receipt["package_sha256"] != _package_hash():
        raise RuntimeError("FreezeShift package changed after validation")


def smoke() -> dict[str, Any]:
    _assert_frozen_package()
    rows = []
    for path in PIPELINES:
        pipeline, config = _config(path)
        model = build_model(config, pretrained=False).eval()
        images = torch.randn(2, 3, 224, 224)
        captions = ["a person riding a bicycle", "two dogs in a field"]
        with torch.no_grad():
            before_image = model.encode_image(images)
            before_text = model.encode_text(captions)
        merged = merge_lora_(model)
        with torch.no_grad():
            after_image = model.encode_image(images)
            after_text = model.encode_text(captions)
        image_delta = float((before_image - after_image).abs().max())
        text_delta = float((before_text - after_text).abs().max())
        if image_delta > 1e-6 or text_delta > 1e-6:
            raise RuntimeError(f"LoRA merge equivalence failed for {path}")
        rows.append(
            {
                "arm": pipeline["freezeshift"]["arm"],
                "merged_modules": len(merged),
                "image_max_abs_delta": image_delta,
                "text_max_abs_delta": text_delta,
            }
        )
    result = {"status": "COMPLETE", "rows": rows}
    atomic_json(result, STUDY_ROOT / "results/manifests/smoke.json")
    return result


def train(pipeline_index: int, seed_index: int, resume: bool = True) -> dict[str, Any]:
    _assert_frozen_package()
    _patch_builders()
    return alignment_runner.run_training_stage(
        PIPELINES[pipeline_index], "sensitivity", seed_index, resume=resume
    )


def evaluate_epoch(pipeline_index: int, epoch_index: int) -> dict[str, Any]:
    _assert_frozen_package()
    _patch_builders()
    return trajectory.evaluate_epoch(PIPELINES[pipeline_index], epoch_index)


def verify(pipeline_index: int) -> dict[str, Any]:
    _assert_frozen_package()
    return trajectory.verify_snapshots(PIPELINES[pipeline_index])


def report_arm(pipeline_index: int) -> dict[str, Any]:
    _assert_frozen_package()
    return trajectory.report(PIPELINES[pipeline_index])


def _load_selected(path: str, seed: int, device: torch.device):
    _, pipeline = alignment_runner.load_pipeline(path)
    report = json.loads((ROOT / pipeline["output_root"] / "report/report.json").read_text())
    row = next(value for value in report["per_seed"] if int(value["seed"]) == seed)
    epoch = int(row["flickr_selected_epoch"])
    checkpoint_dir = ROOT / pipeline["checkpoint_root"] / "sensitivity" / f"distill_strength_1p0__seed_{seed}"
    config = load_config(checkpoint_dir / "config.yaml")
    model = build_model(config).to(device).eval()
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError(f"missing fingerprint in {checkpoint_dir}")
    load_training_checkpoint(
        checkpoint_dir / f"epoch_{epoch:02d}.pt",
        model,
        device=device,
        expected_fingerprint=fingerprint,
    )
    summary = model.parameter_summary()
    return model, config, summary, epoch


def _profile_model(model, config, device: torch.device) -> dict[str, Any]:
    csv_path = ROOT / "data/flickr30k/validation.csv"
    pil_images, captions = frontier._sample_batch(csv_path, 64)
    transform = frontier._student_transform(config)
    images = torch.stack([transform(image) for image in pil_images]).to(device)
    tokens = model.text_encoder.tokenize(captions).to(device)
    model.eval()
    frontier._assert_fully_eval(model, "FreezeShift profile")
    full = frontier._quartiles(
        frontier._cuda_times(
            lambda: (model.encode_image(images), model.encode_text_tokens(tokens)),
            20,
            100,
            device,
        )
    )
    image = frontier._quartiles(
        frontier._cuda_times(lambda: model.encode_image(images), 20, 100, device)
    )
    text = frontier._quartiles(
        frontier._cuda_times(lambda: model.encode_text_tokens(tokens), 20, 100, device)
    )
    return {"full_stack": full, "image_side": image, "text_side": text}


def profile() -> dict[str, Any]:
    _assert_frozen_package()
    provenance = frontier._hardware_guard({})
    device = provenance.pop("device")
    entries = [("M_T1", BASELINE_PIPELINE), *[(Path(path).stem.removeprefix("pipeline_"), path) for path in PIPELINES]]
    rows = []
    for arm, path in entries:
        model, config, summary, epoch = _load_selected(path, 42, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sample_image = torch.randn(2, 3, 224, 224, device=device)
            sample_text = model.text_encoder.tokenize(["a dog", "a bicycle"]).to(device)
            before_image = model.encode_image(sample_image)
            before_text = model.encode_text_tokens(sample_text)
        merged = merge_lora_(model)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            after_image = model.encode_image(sample_image)
            after_text = model.encode_text_tokens(sample_text)
        max_delta = max(
            float((before_image - after_image).abs().max()),
            float((before_text - after_text).abs().max()),
        )
        if max_delta > 5e-3:
            raise RuntimeError(f"BF16 merge deviation too large for {arm}: {max_delta}")
        timing = _profile_model(model, config, device)
        rows.append(
            {
                "arm": arm,
                "seed": 42,
                "selected_epoch": epoch,
                "merged_lora_modules": len(merged),
                "merge_max_abs_delta_bf16": max_delta,
                "adaptation_parameters": summary["params_trainable_inference"],
                "median_ms": timing["full_stack"]["median"],
                "q1_ms": timing["full_stack"]["q1"],
                "q3_ms": timing["full_stack"]["q3"],
                "image_median_ms": timing["image_side"]["median"],
                "text_median_ms": timing["text_side"]["median"],
                "q3_below_9p48": timing["full_stack"]["q3"] < CEILING_MS,
                **provenance,
            }
        )
    result = {
        "status": "COMPLETE",
        "same_allocation": True,
        "lora_merged_before_profiling": True,
        "protocol": {"batch": 64, "warmup": 20, "repeats": 100, "native_bf16": True},
        "rows": rows,
    }
    atomic_csv(pd.DataFrame(rows), STUDY_ROOT / "results/report/latency.csv")
    atomic_json(result, STUDY_ROOT / "results/report/latency.json")
    return result


def final_report() -> dict[str, Any]:
    _assert_frozen_package()
    baseline = json.loads(BASELINE_REPORT.read_text())
    baseline_mean = float(baseline["flickr_mean_under_flickr_validation_epoch_selection"])
    baseline_sd = float(baseline["flickr_validation_selected_sd"])
    latency = json.loads((STUDY_ROOT / "results/report/latency.json").read_text())
    latency_rows = {row["arm"]: row for row in latency["rows"]}
    rows = []
    for path in PIPELINES:
        _, pipeline = alignment_runner.load_pipeline(path)
        arm = pipeline["freezeshift"]["arm"]
        report = json.loads((ROOT / pipeline["output_root"] / "report/report.json").read_text())
        mean = float(report["flickr_mean_under_flickr_validation_epoch_selection"])
        sd = float(report["flickr_validation_selected_sd"])
        pooled = math.sqrt((baseline_sd**2 + sd**2) / 2.0)
        gain = mean - baseline_mean
        timing = latency_rows[arm]
        eligible = (
            int(pipeline["freezeshift"]["expected_inference_trainable_parameters"]) < 5_000_000
            and bool(timing["q3_below_9p48"])
        )
        rows.append(
            {
                "arm": arm,
                "validation_mean_R1": mean,
                "validation_sd": sd,
                "gain_vs_M_T1_pp": gain * 100.0,
                "pooled_sd_pp": pooled * 100.0,
                "exceeds_pooled_sd": gain > pooled,
                "beats_openclip_validation": mean > OPENCLIP_VALIDATION,
                "inference_trainable_parameters": pipeline["freezeshift"]["expected_inference_trainable_parameters"],
                "latency_q3_ms": timing["q3_ms"],
                "eligible": eligible,
            }
        )
    candidates = [row for row in rows if row["eligible"]]
    winner = max(candidates, key=lambda row: row["validation_mean_R1"]) if candidates else None
    promoted = bool(winner and winner["exceeds_pooled_sd"])
    result = {
        "status": "COMPLETE",
        "baseline": {"arm": "M_T1", "validation_mean_R1": baseline_mean, "validation_sd": baseline_sd},
        "winner": winner,
        "promoted": promoted,
        "final_arm": winner["arm"] if promoted else "M_T1",
        "openclip_validation_mean_R1": OPENCLIP_VALIDATION,
        "openclip_beaten_on_validation": bool(winner and winner["beats_openclip_validation"]),
        "flickr_test_used": False,
        "hard_stop_applies": True,
        "rows": rows,
    }
    atomic_csv(pd.DataFrame(rows), STUDY_ROOT / "results/report/summary.csv")
    atomic_json(result, STUDY_ROOT / "results/report/report.json")
    atomic_text(
        "# FreezeShift result\n\n"
        f"Final arm: **{result['final_arm']}**. Flickr test remained sealed.\n",
        STUDY_ROOT / "results/report/report.md",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "smoke", "train", "evaluate", "verify", "report-arm", "profile", "report"))
    parser.add_argument("--pipeline-index", type=int, default=0)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.command == "validate": value = validate()
    elif args.command == "smoke": value = smoke()
    elif args.command == "train": value = train(args.pipeline_index, args.seed_index, not args.no_resume)
    elif args.command == "evaluate": value = evaluate_epoch(args.pipeline_index, args.epoch_index)
    elif args.command == "verify": value = verify(args.pipeline_index)
    elif args.command == "report-arm": value = report_arm(args.pipeline_index)
    elif args.command == "profile": value = profile()
    else: value = final_report()
    print(json.dumps(value, indent=2, default=str))


if __name__ == "__main__":
    main()
