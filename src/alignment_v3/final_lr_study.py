from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.alignment_v3.resolution_distillation_trajectory import report as report_arm
from src.alignment_v3.runner import ROOT, build_job_config, load_pipeline, sensitivity_jobs
from src.phase15.io_utils import atomic_csv, atomic_json


CURRENT_PIPELINE = "configs/text_aggregation_study/pipeline_m_t1.yaml"
CANDIDATE_PIPELINES = (
    "configs/final_lr_study/pipeline_lr_0p5.yaml",
    "configs/final_lr_study/pipeline_lr_0p75.yaml",
    "configs/final_lr_study/pipeline_lr_1p25.yaml",
    "configs/final_lr_study/pipeline_lr_1p5.yaml",
)
ALL_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
CURRENT_LR = 0.002545584412271571
OUTPUT = ROOT / "results/final_lr_study"
PREREG = ROOT / "predictions/final_lr_study.json"
SELECTION = OUTPUT / "selection/seed42_selection.json"
CURRENT_REPORT = ROOT / "results/text_aggregation_study/training/M_T1/report/report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_for_factor(factor: float) -> str:
    if factor == 1.0:
        return CURRENT_PIPELINE
    for path in CANDIDATE_PIPELINES:
        _, pipeline = load_pipeline(path)
        if math.isclose(float(pipeline["lr_study"]["factor_vs_current"]), factor):
            return path
    raise KeyError(f"No pipeline for factor {factor}")


def _seed_peak(pipeline_path: str, seed: int) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    root = ROOT / str(pipeline["output_root"]) / "trajectory" / f"seed_{seed}"
    rows = []
    for epoch in range(1, 25):
        path = root / f"epoch_{epoch:02d}.json"
        if not path.exists():
            raise RuntimeError(f"Missing trajectory metric: {path}")
        payload = json.loads(path.read_text())
        if payload.get("test_split_used") is not False:
            raise RuntimeError(f"Test-seal violation in {path}")
        rows.append(payload)
    winner = max(rows, key=lambda row: (float(row["mean_R@1"]), -int(row["epoch"])))
    return {
        "seed": seed,
        "selected_epoch": int(winner["epoch"]),
        "mean_R1": float(winner["mean_R@1"]),
        "i2t_R1": float(winner["i2t_R@1"]),
        "t2i_R1": float(winner["t2i_R@1"]),
        "checkpoint_fingerprint": winner["checkpoint_fingerprint"],
        "checkpoint_path": winner["checkpoint_path"],
    }


def validate() -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text())
    if not prereg.get("written_before_any_new_runs") or not prereg.get("flickr_test_sealed"):
        raise RuntimeError("Invalid or unsealed preregistration")
    current_report = json.loads(CURRENT_REPORT.read_text())
    if current_report.get("status") != "COMPLETE" or current_report.get("flickr_test_used") is not False:
        raise RuntimeError("Current M_T1 baseline is incomplete or unsealed")
    current_seed42 = _seed_peak(CURRENT_PIPELINE, 42)
    rows = []
    frozen_signature = None
    for factor in ALL_FACTORS:
        path = _pipeline_for_factor(factor)
        _, pipeline = load_pipeline(path)
        jobs = sensitivity_jobs(pipeline)
        if [int(job["seed"]) for job in jobs] != [42, 43, 44]:
            raise RuntimeError(f"Seed contract changed in {path}")
        config = build_job_config(pipeline, jobs[0], "sensitivity")
        signature = {
            "vision": config["model"]["vision_encoder"],
            "text": config["model"]["text_encoder"],
            "teacher": config["distillation"]["teacher_id"],
            "image_size": int(config["data"]["image_size"]),
            "batch_size": int(config["training"]["batch_size"]),
            "epochs": int(config["training"]["epochs"]),
            "loss_type": config["recipe"]["loss_type"],
            "queue": int(config["recipe"]["memory_queue_size"]),
            "image_aggregation": config["recipe"]["image_token_aggregation"],
            "text_aggregation": config["recipe"]["text_token_aggregation"],
        }
        if frozen_signature is None:
            frozen_signature = signature
        elif signature != frozen_signature:
            raise RuntimeError(f"Non-LR field changed in {path}: {signature}")
        expected = CURRENT_LR * factor
        observed = float(config["training"]["lr"])
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"Resolved LR mismatch in {path}: {observed} != {expected}")
        rows.append({"factor": factor, "pipeline": path, "resolved_lr": observed, "reused": factor == 1.0})
    result = {
        "status": "READY",
        "preregistration_sha256": _sha256(PREREG),
        "new_screen_training_jobs": 4,
        "new_confirmation_training_jobs_maximum": 2,
        "current_seed42_reuse": current_seed42,
        "frozen_signature": frozen_signature,
        "rows": rows,
        "flickr_test_sealed": True,
    }
    atomic_csv(pd.DataFrame(rows), OUTPUT / "manifests/candidates.csv")
    atomic_json(result, OUTPUT / "manifests/design.json")
    return result


def select() -> dict[str, Any]:
    design = json.loads((OUTPUT / "manifests/design.json").read_text())
    if design["preregistration_sha256"] != _sha256(PREREG):
        raise RuntimeError("Preregistration changed after validation")
    rows = []
    for factor in ALL_FACTORS:
        peak = _seed_peak(_pipeline_for_factor(factor), 42)
        rows.append({"factor": factor, "pipeline": _pipeline_for_factor(factor), **peak})
    # Exact score ties prefer current, then distance from the current LR.
    winner = max(rows, key=lambda row: (row["mean_R1"], row["factor"] == 1.0, -abs(row["factor"] - 1.0)))
    result = {
        "status": "SELECTED",
        "selection_metric": "Flickr30k validation mean bidirectional R@1",
        "screen_seed": 42,
        "selected_factor": winner["factor"],
        "selected_pipeline": winner["pipeline"],
        "current_lr_retained_at_screen": winner["factor"] == 1.0,
        "confirmation_required": winner["factor"] != 1.0,
        "preregistration_sha256": _sha256(PREREG),
        "rows": rows,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(rows), OUTPUT / "selection/seed42_screen.csv")
    atomic_json(result, SELECTION)
    return result


def selected_pipeline() -> str:
    payload = json.loads(SELECTION.read_text())
    if payload["preregistration_sha256"] != _sha256(PREREG):
        raise RuntimeError("Selection/preregistration hash mismatch")
    return "SKIP" if not payload["confirmation_required"] else str(payload["selected_pipeline"])


def final_report() -> dict[str, Any]:
    selection = json.loads(SELECTION.read_text())
    current = json.loads(CURRENT_REPORT.read_text())
    current_mean = float(current["flickr_mean_under_flickr_validation_epoch_selection"])
    current_sd = float(current["flickr_validation_selected_sd"])
    if not selection["confirmation_required"]:
        result = {
            "status": "COMPLETE",
            "promoted": False,
            "final_factor": 1.0,
            "final_pipeline": CURRENT_PIPELINE,
            "reason": "The current LR won the preregistered seed-42 screen; no confirmation training was required.",
            "current_mean": current_mean,
            "current_sd": current_sd,
            "hard_stop_applies": True,
            "flickr_test_used": False,
        }
        atomic_json(result, OUTPUT / "report/report.json")
        return result
    candidate_report = report_arm(str(selection["selected_pipeline"]))
    candidate_mean = float(candidate_report["flickr_mean_under_flickr_validation_epoch_selection"])
    candidate_sd = float(candidate_report["flickr_validation_selected_sd"])
    pooled_sd = math.sqrt((current_sd ** 2 + candidate_sd ** 2) / 2.0)
    gain = candidate_mean - current_mean
    promoted = gain > pooled_sd
    per_seed = []
    for seed in (42, 43, 44):
        current_peak = _seed_peak(CURRENT_PIPELINE, seed)
        candidate_peak = _seed_peak(str(selection["selected_pipeline"]), seed)
        per_seed.append({
            "seed": seed,
            "current_mean_R1": current_peak["mean_R1"],
            "candidate_mean_R1": candidate_peak["mean_R1"],
            "paired_difference_pp": (candidate_peak["mean_R1"] - current_peak["mean_R1"]) * 100.0,
            "candidate_i2t_R1": candidate_peak["i2t_R1"],
            "candidate_t2i_R1": candidate_peak["t2i_R1"],
            "candidate_selected_epoch": candidate_peak["selected_epoch"],
        })
    result = {
        "status": "COMPLETE",
        "selected_screen_factor": selection["selected_factor"],
        "current_mean": current_mean,
        "current_sd": current_sd,
        "candidate_mean": candidate_mean,
        "candidate_sd": candidate_sd,
        "gain_pp": gain * 100.0,
        "pooled_sd_pp": pooled_sd * 100.0,
        "promoted": promoted,
        "final_factor": selection["selected_factor"] if promoted else 1.0,
        "final_pipeline": selection["selected_pipeline"] if promoted else CURRENT_PIPELINE,
        "rule": "gain > one pooled SD",
        "rule_is_heuristic_not_significance_test": True,
        "per_seed": per_seed,
        "hard_stop_applies": True,
        "flickr_test_used": False,
    }
    atomic_csv(pd.DataFrame(per_seed), OUTPUT / "report/per_seed.csv")
    atomic_json(result, OUTPUT / "report/report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "select", "selected-pipeline", "report"))
    args = parser.parse_args()
    if args.command == "validate":
        value: Any = validate()
    elif args.command == "select":
        value = select()
    elif args.command == "selected-pipeline":
        print(selected_pipeline())
        return
    else:
        value = final_report()
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
