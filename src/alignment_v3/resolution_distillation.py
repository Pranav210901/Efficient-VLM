from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _hardware_guard,
    _load_student,
    _metrics_from_embeddings,
    _student_loader,
)
from src.alignment_v3.resolution_arm import _guard, _select
from src.alignment_v3.runner import ROOT, checkpoint_root, load_pipeline, sensitivity_jobs
from src.phase15.io_utils import atomic_csv, atomic_json
from src.training.evaluate import extract_embeddings


def _checkpoint_job(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "student",
        "entry_id": "mobileclip2_distilled_224",
        "label": "DINOv3 ViT-S/16 + MiniLM, MobileCLIP2-distilled, 224px",
        "role": "selected_resolution_distillation",
        "seed": int(job["seed"]),
        "checkpoint_dir": str(
            (
                checkpoint_root(pipeline)
                / "sensitivity"
                / str(job["run_id"])
            ).relative_to(ROOT)
        ),
    }


def validate(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    jobs = sensitivity_jobs(pipeline)
    expected = {(42, True), (43, True), (44, True)}
    observed = {(int(job["seed"]), bool(job["distillation"])) for job in jobs}
    if observed != expected or len(jobs) != 3:
        raise RuntimeError(f"expected exactly three distilled jobs, got {jobs}")
    if int(pipeline["experiment"]["image_size"]) != 224:
        raise RuntimeError("combination arm must be fixed at 224px")
    if str(pipeline["optional_transfer"]["flickr30k_csv"]) != "data/flickr30k/validation.csv":
        raise RuntimeError("combination arm may evaluate Flickr30k validation only")
    if "test" in str(pipeline["optional_transfer"]["flickr30k_csv"]).lower():
        raise RuntimeError("Flickr30k test is sealed")
    cache = ROOT / str(
        pipeline.get("distillation", {}).get(
            "cache_path",
            "results/alignment_wave1/mobileclip2/teacher_cache/mobileclip2_s0_dfndr2b.pt",
        )
    )
    # The cache path lives in the resolved base config rather than the pipeline.
    from src.alignment_v3.runner import build_job_config

    rows = []
    for job in jobs:
        config = build_job_config(pipeline, job, "sensitivity")
        identity = _guard({"image_size": 224}, config)
        if not bool(config["recipe"]["distillation"]):
            raise RuntimeError("non-distilled job found in combination arm")
        cache = ROOT / str(config["distillation"]["cache_path"])
        if not cache.is_file():
            raise FileNotFoundError(f"verified Wave 1 teacher cache missing: {cache}")
        rows.append(
            {
                **job,
                **identity,
                "resolved_lr": float(config["training"]["lr"]),
                "teacher_cache": str(cache.relative_to(ROOT)),
            }
        )
    atomic_csv(
        pd.DataFrame(rows),
        ROOT / "results/resolution_distillation_224/manifests/jobs_verified.csv",
    )
    return {
        "status": "READY",
        "jobs": len(rows),
        "flickr_test_sealed": True,
        "rows": rows,
    }


def evaluate_validation(pipeline_path: str, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(sensitivity_jobs(pipeline), index)
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    model, config, fingerprint = _load_student(
        _checkpoint_job(pipeline, job), device
    )
    identity = _guard({"image_size": 224}, config, model)
    csv_path = ROOT / "data/flickr30k/validation.csv"
    loader = _student_loader(config, csv_path, 256)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, [1, 5, 10])
    payload = {
        "status": "COMPLETE",
        **job,
        **identity,
        **metrics,
        **provenance,
        "dataset": "Flickr30k Karpathy validation",
        "test_split_used": False,
        "checkpoint_fingerprint": fingerprint.digest,
    }
    destination = (
        ROOT / "results/resolution_distillation_224/validation" / str(job["run_id"])
    )
    atomic_json(payload, destination / "metrics.json")
    return payload


def report(pipeline_path: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows = []
    for job in sensitivity_jobs(pipeline):
        path = (
            ROOT
            / "results/resolution_distillation_224/validation"
            / str(job["run_id"])
            / "metrics.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    baseline_path = ROOT / "results/resolution_arm/report/resolution_curve.csv"
    baseline = pd.read_csv(baseline_path)
    baseline_224 = float(
        baseline.loc[baseline["resolution"] == 224, "flickr_validation_mean_R1"].iloc[0]
    )
    distilled_mean = float(frame["mean_R@1"].mean())
    result = {
        "status": "COMPLETE",
        "selected_resolution": 224,
        "teacher": "MobileCLIP2-S0:dfndr2b",
        "seeds": [42, 43, 44],
        "flickr_validation_mean_R1": distilled_mean,
        "flickr_validation_sd_R1": float(frame["mean_R@1"].std(ddof=1)),
        "flickr_validation_min_R1": float(frame["mean_R@1"].min()),
        "flickr_validation_max_R1": float(frame["mean_R@1"].max()),
        "matched_224_baseline_mean_R1": baseline_224,
        "distillation_gain_pp": (distilled_mean - baseline_224) * 100.0,
        "test_split_used": False,
        "flickr_test_sealed": True,
    }
    destination = ROOT / "results/resolution_distillation_224/report"
    atomic_csv(frame, destination / "per_seed.csv")
    atomic_json(result, destination / "report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "evaluate", "report"))
    parser.add_argument(
        "--pipeline", default="configs/resolution_distillation_224/pipeline.yaml"
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.pipeline)
    elif args.command == "evaluate":
        result = evaluate_validation(args.pipeline, args.index)
    else:
        result = report(args.pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
