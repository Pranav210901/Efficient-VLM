from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.alignment_v3.fingerprint import hash_config, read_fingerprint, write_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.training import load_training_checkpoint, train
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_csv, atomic_json
from src.training.evaluate import extract_embeddings
from src.utils.config import deep_update, load_config, save_config
from src.utils.device import get_device


ROOT = Path(__file__).resolve().parents[2]


def load_pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return load_config(value)


def capacity(batch_size: int, target_age: int) -> int:
    return int(batch_size) * int(target_age)


def canonical_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    spec = pipeline["factorial"]
    jobs: dict[tuple[int, int, int, str], dict[str, Any]] = {}

    def add(batch: int, age: int, seed: int, mode: str, membership: str) -> None:
        cap = 0 if mode == "none" or age == 0 else capacity(batch, age)
        actual_mode = "none" if cap == 0 else mode
        key = (batch, cap, seed, actual_mode)
        if key not in jobs:
            run_id = f"b{batch}__age_{age}__{actual_mode}__seed_{seed}"
            jobs[key] = {
                "index": -1,
                "run_id": run_id,
                "experiment_id": "queue_factorial",
                "batch_size": batch,
                "target_age_steps": age,
                "memory_queue_size": cap,
                "queue_mode": actual_mode,
                "seed": seed,
                "memberships": [],
            }
        if membership not in jobs[key]["memberships"]:
            jobs[key]["memberships"].append(membership)

    for batch in spec["batches"]:
        for age in spec["target_ages"]:
            for seed in spec["seeds"]:
                add(int(batch), int(age), int(seed), "both", "grid1_matched_age")
    for mode in spec["queue_modes"]:
        ages = [0] if mode == "none" else spec["modality_ages"]
        for age in ages:
            for seed in spec["seeds"]:
                add(1024, int(age), int(seed), str(mode), "grid2_modality")
    for batch in spec["batches"]:
        for seed in spec["seeds"]:
            add(int(batch), 0, int(seed), "none", "control_batch")
    values = sorted(
        jobs.values(),
        key=lambda row: (
            row["batch_size"],
            row["target_age_steps"],
            row["queue_mode"],
            row["seed"],
        ),
    )
    for index, job in enumerate(values):
        job["index"] = index
    if len(values) != 66:
        raise AssertionError(f"canonical queue factorial must contain 66 cells, got {len(values)}")
    return values


def smoke_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for index, cell in enumerate(pipeline["smoke"]["cells"]):
        values.append(
            {
                "index": index,
                "run_id": (
                    f"smoke_timing_v2__b{cell['batch_size']}"
                    f"__queue_{cell['memory_queue_size']}"
                ),
                "experiment_id": "queue_factorial_smoke",
                "batch_size": int(cell["batch_size"]),
                "target_age_steps": int(cell["target_age_steps"]),
                "memory_queue_size": int(cell["memory_queue_size"]),
                "queue_mode": "both",
                "seed": 42,
                "memberships": ["smoke"],
            }
        )
    return values


def build_config(pipeline: dict[str, Any], job: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    config = load_config(ROOT / pipeline["base_config"])
    pinned = float(pipeline["factorial"]["pinned_resolved_lr"])
    batch = int(job["batch_size"])
    counterfactual = float(
        pipeline["factorial"]["counterfactual_sqrt_lrs"][str(batch)]
        if str(batch) in pipeline["factorial"]["counterfactual_sqrt_lrs"]
        else pipeline["factorial"]["counterfactual_sqrt_lrs"][batch]
    )
    stage = "smoke" if smoke else "runs"
    save_dir = ROOT / pipeline["checkpoint_root"] / stage / job["run_id"]
    config = deep_update(
        config,
        {
            "seed": int(job["seed"]),
            "experiment_id": job["experiment_id"],
            "run_id": job["run_id"],
            "model": pipeline["student"],
            "recipe": {"loss_type": "infonce_no_queue"},
            "data": {
                "train_csv": pipeline["split"]["train_csv"],
                "val_csv": pipeline["split"]["dev_csv"],
                "train_captions_per_image": None,
                "num_workers": int(pipeline["resources"]["dataloader_workers"]),
            },
            "training": {
                "batch_size": batch,
                "memory_queue_size": int(job["memory_queue_size"]),
                "queue_mode": job["queue_mode"],
                "lr": pinned,
                "lr_scale_rule": "none",
                "pinned_resolved_lr": pinned,
                "counterfactual_sqrt_lr": counterfactual,
                "precision": "bf16",
                "save_dir": str(save_dir.relative_to(ROOT)),
            },
            "queue_diagnostics": {
                "probe_manifest": str(
                    (
                        ROOT
                        / pipeline["output_root"]
                        / stage
                        / job["run_id"]
                        / "projector_probe.csv"
                    ).relative_to(ROOT)
                )
            },
            "provenance": {
                "split_manifest": pipeline["split"]["manifest"],
                "factorial_memberships": job["memberships"],
                "target_age_steps": int(job["target_age_steps"]),
            },
        },
    )
    if smoke:
        # The training loop supports a fixed optimizer-step cap for smoke jobs.
        config["training"]["max_optimizer_steps"] = int(pipeline["smoke"]["optimizer_steps"])
        config["training"]["select_on_dev"] = False
        config["training"]["step_timing"] = True
    return config


def prediction_gate(pipeline: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / pipeline["predictions_path"]
    if not path.is_file():
        raise RuntimeError(f"missing prediction file: {path}")
    payload = json.loads(path.read_text())
    values = payload.get("unseen_predictions", [])
    if not values or any(not str(value.get("statement", "")).strip() for value in values):
        raise RuntimeError("prediction file contains a missing/empty statement")
    return payload


def hardware_guard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("queue factorial requires CUDA")
    name = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in name or "Blackwell" not in name:
        raise RuntimeError(f"refusing non-RTX-PRO-6000-Blackwell GPU: {name}")
    major, minor = torch.cuda.get_device_capability(0)
    if major < 10 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"native BF16 Blackwell required; got capability {major}.{minor}"
        )
    return {"gpu": name, "compute_capability": f"{major}.{minor}", "precision": "native_bf16"}


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    predictions = prediction_gate(pipeline)
    jobs = canonical_jobs(pipeline)
    output = ROOT / pipeline["output_root"] / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(jobs, output / "cells.json")
    atomic_csv(pd.DataFrame(jobs), output / "cells.csv")
    reuse = reuse_audit(pipeline)
    pending = [job for job in jobs if str(job["index"]) not in reuse["by_index"]]
    atomic_json(pending, output / "pending_cells.json")
    report = {
        "status": "READY",
        "unique_cells": len(jobs),
        "verified_reuse": len(reuse["by_index"]),
        "expected_new_runs": len(pending),
        "prediction_ids": [row["id"] for row in predictions["unseen_predictions"]],
        "pinned_resolved_lr": pipeline["factorial"]["pinned_resolved_lr"],
    }
    atomic_json(report, output / "validation.json")
    return report


def reuse_audit(pipeline: dict[str, Any]) -> dict[str, Any]:
    jobs = canonical_jobs(pipeline)
    sources = {
        42: (
            "wave0-lr-control",
            "infonce_no_queue__captions_all__b1024__lr_3p0__seed_42",
        ),
        43: (
            "wave0-winner-confirmation",
            "infonce_no_queue__captions_all__b1024__lr_3p0__seed_43",
        ),
        44: (
            "wave0-winner-confirmation",
            "infonce_no_queue__captions_all__b1024__lr_3p0__seed_44",
        ),
    }
    by_index: dict[str, Any] = {}
    rows = []
    for seed, (stage, run_id) in sources.items():
        job = next(
            row
            for row in jobs
            if row["batch_size"] == 1024
            and row["target_age_steps"] == 0
            and row["seed"] == seed
        )
        checkpoint_dir = ROOT / "checkpoints/alignment_v4_wave0" / stage / run_id
        result_dir = ROOT / "results/alignment_v4_wave0" / stage / run_id
        config_path = checkpoint_dir / "config.yaml"
        metrics_path = result_dir / "metrics.json"
        summary_path = checkpoint_dir / "run_summary.json"
        fp = read_fingerprint(checkpoint_dir / "fingerprint.json")
        reasons = []
        if not all(
            path.is_file()
            for path in (
                config_path,
                metrics_path,
                summary_path,
                checkpoint_dir / "best.pt",
            )
        ) or fp is None:
            reasons.append("incomplete_artifact_bundle")
        else:
            config = load_config(config_path)
            metrics = json.loads(metrics_path.read_text())
            run_summary = json.loads(summary_path.read_text())
            historical_report = ROOT / "results/alignment_v4_wave0/report.md"
            report_text = (
                historical_report.read_text() if historical_report.is_file() else ""
            )
            checks = {
                "seed": int(config["seed"]) == seed,
                "pair": config["model"]["vision_encoder"] == "dinov3_vits16"
                and config["model"]["text_encoder"] == "all_minilm_l6_v2",
                "loss": config["recipe"]["loss_type"] == "infonce_no_queue",
                "captions": config["data"]["train_captions_per_image"] is None,
                "batch": int(config["training"]["batch_size"]) == 1024,
                "queue": int(config["training"]["memory_queue_size"]) == 0,
                "lr": math.isclose(
                    float(config["training"]["lr"]),
                    float(pipeline["factorial"]["pinned_resolved_lr"]),
                    rel_tol=0,
                    abs_tol=1e-15,
                ),
                "precision": str(config["training"]["precision"]).lower() == "bf16",
                "epochs_configured": int(config["training"]["epochs"]) == 12,
                "twelve_epochs_completed": int(
                    run_summary.get("completed_epoch", 0)
                )
                == 12,
                "fingerprint_config": hash_config(config) == fp.config_hash,
                "gpu": "RTX PRO 6000 Blackwell" in str(metrics.get("gpu", "")),
                "partition_and_native_bf16_provenance": (
                    "array 2223439 ran on the teaching partition's RTX PRO 6000 Blackwell GPUs with native BF16"
                    in report_text
                    if seed == 42
                    else (
                        config.get("provenance", {}).get(
                            "slurm_partition_profile"
                        )
                        == "teaching"
                        and "RTX PRO 6000 Blackwell"
                        in str(metrics.get("gpu", ""))
                    )
                ),
            }
            reasons.extend(name for name, passed in checks.items() if not passed)
        accepted = not reasons
        row = {
            "factorial_index": job["index"],
            "seed": seed,
            "source_stage": stage,
            "source_run_id": run_id,
            "accepted": accepted,
            "reasons": reasons,
            "link_only_no_copy": True,
        }
        rows.append(row)
        if accepted:
            by_index[str(job["index"])] = row
    result = {"status": "COMPLETE", "by_index": by_index, "rows": rows}
    destination = ROOT / pipeline["output_root"] / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(result, destination / "reuse_audit.json")
    return result


def pending_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    # Completed submissions retain the signed-off reuse audit but deliberately
    # omit the heavyweight source checkpoints. Prefer that immutable evidence
    # when it contains the three declared age-zero rows; live runs without a
    # closed audit still perform the full artifact verification above.
    audit_path = ROOT / pipeline["output_root"] / "manifests/reuse_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    if audit.get("status") != "COMPLETE" or len(audit.get("by_index", {})) != 3:
        audit = reuse_audit(pipeline)
    reused = {int(value) for value in audit["by_index"]}
    return [job for job in canonical_jobs(pipeline) if int(job["index"]) not in reused]


def first_cell_job(pipeline: dict[str, Any]) -> dict[str, Any]:
    matches = [
        job
        for job in pending_jobs(pipeline)
        if job["batch_size"] == 1024
        and job["memory_queue_size"] == 65536
        and job["queue_mode"] == "both"
        and job["seed"] == 42
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one first-cell gate job, found {len(matches)}")
    return matches[0]


def remaining_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    gate_run_id = first_cell_job(pipeline)["run_id"]
    values = [job for job in pending_jobs(pipeline) if job["run_id"] != gate_run_id]
    if len(values) != 62:
        raise RuntimeError(f"remaining factorial must contain 62 jobs, got {len(values)}")
    return values


def _select(values: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else index
    return values[int(selected)]


def _run_train_job(
    pipeline: dict[str, Any],
    job: dict[str, Any],
    *,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    prediction_gate(pipeline)
    provenance = hardware_guard()
    config = build_config(pipeline, job, smoke=smoke)
    from src.alignment_v3.runner import _fingerprint

    fingerprint = _fingerprint(config)
    result = train(config, fingerprint, resume=resume)
    return {"run_id": job["run_id"], **provenance, **result}


def run_train(pipeline: dict[str, Any], index: int | None, *, smoke: bool, resume: bool) -> dict[str, Any]:
    job = _select(smoke_jobs(pipeline) if smoke else pending_jobs(pipeline), index)
    return _run_train_job(pipeline, job, smoke=smoke, resume=resume)


def run_first_cell(pipeline: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    return _run_train_job(
        pipeline, first_cell_job(pipeline), smoke=False, resume=resume
    )


def run_remaining(
    pipeline: dict[str, Any], index: int | None, *, resume: bool
) -> dict[str, Any]:
    gate = ROOT / pipeline["output_root"] / "gate" / "report.json"
    if not gate.is_file() or json.loads(gate.read_text()).get("verdict") != "PASS":
        raise RuntimeError("remaining factorial is blocked until first-cell verdict is PASS")
    return _run_train_job(
        pipeline, _select(remaining_jobs(pipeline), index), smoke=False, resume=resume
    )


def run_eval(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = hardware_guard()
    job = _select(pending_jobs(pipeline), index)
    config = build_config(pipeline, job)
    checkpoint_dir = ROOT / config["training"]["save_dir"]
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError(f"missing checkpoint fingerprint: {checkpoint_dir}")
    device = get_device("auto")
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint_dir / "best.pt", model, device=device, expected_fingerprint=fingerprint
    )
    eval_config = deep_update(
        config, {"training": {"batch_size": int(config["evaluation"]["batch_size"])}}
    )
    _, loader = build_dataloaders(eval_config)
    embeddings = extract_embeddings(model, loader, device)
    from src.alignment_v3.runner import _metrics_from_embeddings

    metrics = _metrics_from_embeddings(embeddings, list(config["evaluation"]["k_values"]))
    destination = ROOT / pipeline["output_root"] / "runs" / job["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    row = {
        "status": "COMPLETE",
        **job,
        **metrics,
        **provenance,
        "pinned_resolved_lr": config["training"]["pinned_resolved_lr"],
        "counterfactual_sqrt_lr": config["training"]["counterfactual_sqrt_lr"],
        "fingerprint_digest": fingerprint.digest,
    }
    atomic_json(row, destination / "metrics.json")
    write_fingerprint(destination / "fingerprint.json", fingerprint)
    append_ledger(pipeline, job, row, checkpoint_dir)
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def append_ledger(
    pipeline: dict[str, Any],
    job: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    diagnostic_path = checkpoint_dir / "queue_diagnostics.jsonl"
    diagnostic_rows = [
        json.loads(line)
        for line in diagnostic_path.read_text().splitlines()
        if line.strip()
    ] if diagnostic_path.is_file() else []
    full = [row for row in diagnostic_rows if row.get("phase") == "full_queue"]
    drift_path = checkpoint_dir / "projector_drift.jsonl"
    drift = [
        json.loads(line)
        for line in drift_path.read_text().splitlines()
        if line.strip()
    ] if drift_path.is_file() else []
    config = load_config(checkpoint_dir / "config.yaml")

    def nested(rows: list[dict[str, Any]], outer: str, inner: str) -> list[float]:
        return [
            float(row[outer][inner])
            for row in rows
            if isinstance(row.get(outer), dict) and row[outer].get(inner) is not None
        ]

    positive = []
    for row in full:
        for key in (
            "image_queue_positive_fraction",
            "text_queue_positive_fraction",
        ):
            if row.get(key) is not None:
                positive.append(float(row[key]))
    entry = {
        "wave": "queue_factorial",
        "run_id": job["run_id"],
        "question": "Does queue damage track feature age or entry count, and is it modality-specific?",
        "config_fingerprint": metrics["fingerprint_digest"],
        "seed": job["seed"],
        "git_sha": metrics["fingerprint_digest"],
        "image_queue_capacity": (
            job["memory_queue_size"] if job["queue_mode"] in {"image_only", "both"} else 0
        ),
        "text_queue_capacity": (
            job["memory_queue_size"] if job["queue_mode"] in {"text_only", "both"} else 0
        ),
        "queue_mode": job["queue_mode"],
        "target_age_steps": job["target_age_steps"],
        "measured_image_age_steps": _mean(nested(full, "image_age", "mean")),
        "measured_text_age_steps": _mean(nested(full, "text_age", "mean")),
        "image_exposures": _mean(nested(full, "image_exposures", "mean")),
        "queue_fill_fraction": _mean(
            [
                float(value)
                for row in full
                for value in (
                    row.get("image_fill_fraction"),
                    row.get("text_fill_fraction"),
                )
                if value is not None
            ]
        ),
        "queue_positive_fraction": _mean(positive),
        "image_projector_drift": drift[-1]["image_from_step0"] if drift else None,
        "text_projector_drift": drift[-1]["text_from_step0"] if drift else None,
        "i2t_loss": _mean([float(row["i2t_loss"]) for row in full]),
        "t2i_loss": _mean([float(row["t2i_loss"]) for row in full]),
        "peak_memory_gib": None,
        "images_per_second": None,
        "dev_r1": metrics["mean_R@1"],
        "pinned_resolved_lr": config["training"]["pinned_resolved_lr"],
        "counterfactual_sqrt_lr": config["training"]["counterfactual_sqrt_lr"],
        "status": "COMPLETE",
    }
    training_csv = checkpoint_dir / "metrics.csv"
    if training_csv.is_file():
        training = pd.read_csv(training_csv)
        entry["peak_memory_gib"] = float(training["peak_training_memory_bytes"].max()) / 1024**3
        entry["images_per_second"] = float(training["images_per_second"].mean())
    ledger = ROOT / pipeline["output_root"] / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        if any(
            json.loads(line).get("config_fingerprint") == entry["config_fingerprint"]
            for line in handle
            if line.strip()
        ):
            return
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    jobs = canonical_jobs(pipeline)
    audit = reuse_audit(pipeline)
    rows = []
    for job in jobs:
        if str(job["index"]) in audit["by_index"]:
            source = audit["by_index"][str(job["index"])]
            path = (
                ROOT
                / "results/alignment_v4_wave0"
                / source["source_stage"]
                / source["source_run_id"]
                / "metrics.json"
            )
        else:
            path = ROOT / pipeline["output_root"] / "runs" / job["run_id"] / "metrics.json"
        if path.is_file():
            values = json.loads(path.read_text())
            rows.append({**job, "dev_R@1": float(values["mean_R@1"])})
    frame = pd.DataFrame(rows)
    destination = ROOT / pipeline["output_root"] / "report"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, destination / "measured_results.csv")
    if len(frame) != 66:
        result = {"status": "INCOMPLETE", "completed": len(frame), "expected": 66}
        atomic_json(result, destination / "report.json")
        return result
    summary = (
        frame.groupby(["batch_size", "target_age_steps", "queue_mode"], as_index=False)
        .agg(
            mean=("dev_R@1", "mean"),
            sd=("dev_R@1", "std"),
            minimum=("dev_R@1", "min"),
            maximum=("dev_R@1", "max"),
        )
    )
    atomic_csv(summary, destination / "summary.csv")
    matched = frame[
        frame["memberships"].apply(lambda values: "grid1_matched_age" in values)
    ]
    verdicts = []
    for age in pipeline["factorial"]["target_ages"]:
        pivot = matched[matched["target_age_steps"] == age].pivot(
            index="seed", columns="batch_size", values="dev_R@1"
        )
        difference_pp = float((pivot[512] - pivot[1024]).mean() * 100)
        verdicts.append(
            {
                "target_age_steps": age,
                "paired_difference_pp_b512_minus_b1024": difference_pp,
                "within_equivalence_margin": abs(difference_pp) <= 1.0,
            }
        )
    equivalent = sum(row["within_equivalence_margin"] for row in verdicts)
    count_higher = sum(
        row["paired_difference_pp_b512_minus_b1024"] > 1.0 for row in verdicts
    )
    harm_thresholds: dict[str, int | None] = {}
    for batch in (512, 1024):
        batch_summary = {
            int(row.target_age_steps): float(row.mean)
            for row in summary[
                (summary["batch_size"] == batch)
                & (summary["queue_mode"].isin(["none", "both"]))
            ].itertuples()
        }
        baseline = batch_summary[0]
        harm_thresholds[str(batch)] = next(
            (
                age
                for age in pipeline["factorial"]["target_ages"][1:]
                if (baseline - batch_summary[int(age)]) * 100 > 1.0
            ),
            None,
        )

    condition_means = (
        frame.groupby(
            ["batch_size", "target_age_steps", "queue_mode"], as_index=False
        )
        .agg(mean_dev_r1=("dev_R@1", "mean"))
        .sort_values("mean_dev_r1", ascending=False)
    )
    ordering_checks = []
    conditions = list(condition_means.itertuples(index=False))
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1 :]:
            gap_pp = (float(left.mean_dev_r1) - float(right.mean_dev_r1)) * 100
            if gap_pp <= 1.0:
                continue
            left_rows = frame[
                (frame["batch_size"] == left.batch_size)
                & (frame["target_age_steps"] == left.target_age_steps)
                & (frame["queue_mode"] == left.queue_mode)
            ].set_index("seed")
            right_rows = frame[
                (frame["batch_size"] == right.batch_size)
                & (frame["target_age_steps"] == right.target_age_steps)
                & (frame["queue_mode"] == right.queue_mode)
            ].set_index("seed")
            shared = sorted(set(left_rows.index) & set(right_rows.index))
            consistent = len(shared) == 3 and all(
                float(left_rows.loc[seed, "dev_R@1"])
                > float(right_rows.loc[seed, "dev_R@1"])
                for seed in shared
            )
            ordering_checks.append(
                {
                    "higher": [
                        left.batch_size,
                        left.target_age_steps,
                        left.queue_mode,
                    ],
                    "lower": [
                        right.batch_size,
                        right.target_age_steps,
                        right.queue_mode,
                    ],
                    "mean_gap_pp": gap_pp,
                    "consistent_all_three_seeds": consistent,
                }
            )
    result = {
        "status": "COMPLETE",
        "age_dominates": equivalent >= 6,
        "count_matters": count_higher >= 3,
        "equivalent_ages": equivalent,
        "batch512_higher_over_1pp_ages": count_higher,
        "harm_threshold_by_batch": harm_thresholds,
        "seed_ordering_consistent": all(
            row["consistent_all_three_seeds"] for row in ordering_checks
        ),
        "seed_ordering_checks": ordering_checks,
        "paired_contrasts": verdicts,
    }
    atomic_json(result, destination / "report.json")
    return result


def smoke_report(pipeline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for job in smoke_jobs(pipeline):
        path = ROOT / pipeline["checkpoint_root"] / "smoke" / job["run_id"] / "metrics.csv"
        if not path.is_file():
            raise RuntimeError(f"missing smoke metrics: {path}")
        frame = pd.read_csv(path)
        checkpoint_dir = path.parent
        diagnostics = [
            json.loads(line)
            for line in (checkpoint_dir / "queue_diagnostics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        warmup = [row for row in diagnostics if row["phase"] == "warmup"]
        full = [row for row in diagnostics if row["phase"] == "full_queue"]

        def age_summary(modality: str) -> dict[str, float]:
            values = [
                row[f"{modality}_age"]
                for row in full
                if isinstance(row.get(f"{modality}_age"), dict)
            ]
            return {
                key: sum(float(value[key]) for value in values) / len(values)
                for key in ("mean", "p50", "p95", "max")
            }

        peak = float(frame["peak_training_memory_bytes"].max())
        total = float(frame["gpu_total_memory_bytes"].max())
        rows.append(
            {
                **job,
                "peak_memory_gib": peak / 1024**3,
                "total_memory_gib": total / 1024**3,
                "headroom_gib": (total - peak) / 1024**3,
                "images_per_second": float(frame["images_per_second"].mean()),
                "warmup_images_per_second": _mean(
                    [float(row["images_per_second"]) for row in warmup]
                ),
                "full_queue_images_per_second": _mean(
                    [float(row["images_per_second"]) for row in full]
                ),
                "measured_image_age_steps": age_summary("image"),
                "measured_text_age_steps": age_summary("text"),
                "steps_to_full_queue": min(
                    int(row["optimizer_step"]) for row in full
                )
                - 1,
                "warmup_fill_series": [
                    {
                        "optimizer_step": row["optimizer_step"],
                        "image_fill_fraction": row["image_fill_fraction"],
                        "text_fill_fraction": row["text_fill_fraction"],
                    }
                    for row in warmup
                ],
                "classification": "NON_OOM",
            }
        )
    destination = ROOT / pipeline["output_root"] / "smoke"
    atomic_json(
        {
            "status": "COMPLETE",
            "instrumentation_version": "timing_v2",
            "classification": "NON_OOM",
            "cells": rows,
        },
        destination / "report.json",
    )
    return {"status": "COMPLETE", "cells": rows}


def first_cell_gate_report(pipeline: dict[str, Any]) -> dict[str, Any]:
    job = first_cell_job(pipeline)
    checkpoint_dir = ROOT / pipeline["checkpoint_root"] / "runs" / job["run_id"]
    summary_path = checkpoint_dir / "run_summary.json"
    drift_path = checkpoint_dir / "projector_drift.jsonl"
    queue_path = checkpoint_dir / "queue_diagnostics.jsonl"
    if not all(path.is_file() for path in (summary_path, drift_path, queue_path)):
        raise RuntimeError(f"incomplete first-cell diagnostic bundle: {checkpoint_dir}")
    training_summary = json.loads(summary_path.read_text())
    if int(training_summary.get("completed_epoch", 0)) != 12:
        raise RuntimeError("first-cell gate requires a completed 12-epoch run")
    drift = [json.loads(line) for line in drift_path.read_text().splitlines() if line.strip()]
    queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    if not drift or any("epoch" not in row for row in drift):
        raise RuntimeError("first-cell drift rows are missing epoch provenance")

    per_epoch = []
    for epoch in range(1, 13):
        values = [row for row in drift if int(row["epoch"]) == epoch]
        if not values:
            raise RuntimeError(f"no projector-drift measurements in epoch {epoch}")
        per_epoch.append(
            {
                "epoch": epoch,
                "image_mean_drift_from_previous": _mean(
                    [float(row["image_from_previous"]) for row in values]
                ),
                "text_mean_drift_from_previous": _mean(
                    [float(row["text_from_previous"]) for row in values]
                ),
                "image_mean_drift_from_step0": _mean(
                    [float(row["image_from_step0"]) for row in values]
                ),
                "text_mean_drift_from_step0": _mean(
                    [float(row["text_from_step0"]) for row in values]
                ),
                "measurements": len(values),
            }
        )
    first, final = per_epoch[0], per_epoch[-1]
    image_ratio = float(final["image_mean_drift_from_previous"]) / float(
        first["image_mean_drift_from_previous"]
    )
    text_ratio = float(final["text_mean_drift_from_previous"]) / float(
        first["text_mean_drift_from_previous"]
    )
    image_pass = image_ratio < 0.20
    text_pass = text_ratio < 0.20
    verdict = gate_verdict(image_ratio, text_ratio)

    def crossing(modality: str) -> dict[str, Any] | None:
        found = next(
            (
                row
                for row in drift
                if float(row[f"{modality}_from_step0"]) >= 0.95
            ),
            None,
        )
        return (
            {
                "optimizer_step": int(found["optimizer_step"]),
                "epoch": int(found["epoch"]),
                "distance": float(found[f"{modality}_from_step0"]),
            }
            if found is not None
            else None
        )

    boundary_windows = []
    for epoch in range(1, 12):
        values = [
            row
            for row in queue
            if (
                int(row["epoch"]) == epoch
                and int(row["optimizer_step_in_epoch"])
                >= max(
                    int(value["optimizer_step_in_epoch"])
                    for value in queue
                    if int(value["epoch"]) == epoch
                )
                - 2
            )
            or (
                int(row["epoch"]) == epoch + 1
                and int(row["optimizer_step_in_epoch"]) <= 3
            )
        ]
        boundary_windows.append(
            {
                "after_epoch": epoch,
                "steps": [
                    {
                        "optimizer_step": int(row["optimizer_step"]),
                        "epoch": int(row["epoch"]),
                        "optimizer_step_in_epoch": int(row["optimizer_step_in_epoch"]),
                        "image_queue_positive_fraction": row.get(
                            "image_queue_positive_fraction"
                        ),
                        "text_queue_positive_fraction": row.get(
                            "text_queue_positive_fraction"
                        ),
                    }
                    for row in values
                ],
            }
        )
    positive_values = [
        float(value)
        for row in queue
        for value in (
            row.get("image_queue_positive_fraction"),
            row.get("text_queue_positive_fraction"),
        )
        if value is not None
    ]
    positive_seen = any(value > 0 for value in positive_values)
    result = {
        "status": "COMPLETE",
        "verdict": verdict,
        "run_id": job["run_id"],
        "cell": job,
        "threshold": 0.20,
        "combined_mean_used": False,
        "image": {
            "first_epoch_mean_drift_from_previous": first[
                "image_mean_drift_from_previous"
            ],
            "final_epoch_mean_drift_from_previous": final[
                "image_mean_drift_from_previous"
            ],
            "ratio": image_ratio,
            "passes": image_pass,
            "first_crossing_0p95": crossing("image"),
        },
        "text": {
            "first_epoch_mean_drift_from_previous": first[
                "text_mean_drift_from_previous"
            ],
            "final_epoch_mean_drift_from_previous": final[
                "text_mean_drift_from_previous"
            ],
            "ratio": text_ratio,
            "passes": text_pass,
            "first_crossing_0p95": crossing("text"),
        },
        "passing_modalities": [
            modality
            for modality, passed in (("image", image_pass), ("text", text_pass))
            if passed
        ],
        "failing_modalities": [
            modality
            for modality, passed in (("image", image_pass), ("text", text_pass))
            if not passed
        ],
        "per_epoch_drift": per_epoch,
        "queue_positive_fraction_nonzero_observed": positive_seen,
        "epoch_boundary_windows": boundary_windows,
        "remaining_cells_unblocked": verdict == "PASS" and positive_seen,
    }
    # A missing positive is an instrumentation/data-ID audit stop even if drift passes.
    if verdict == "PASS" and not positive_seen:
        result["verdict"] = "BLOCKED_POSITIVE_FRACTION_AUDIT"
        result["remaining_cells_unblocked"] = False
    destination = ROOT / pipeline["output_root"] / "gate"
    atomic_json(result, destination / "report.json")
    atomic_csv(pd.DataFrame(per_epoch), destination / "per_epoch_drift.csv")
    return result


def gate_verdict(image_ratio: float, text_ratio: float) -> str:
    image_pass = float(image_ratio) < 0.20
    text_pass = float(text_ratio) < 0.20
    if image_pass and text_pass:
        return "PASS"
    if image_pass or text_pass:
        return "PARTIAL"
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "train",
            "eval",
            "smoke",
            "smoke-report",
            "first-cell",
            "gate-report",
            "remaining",
            "manifest",
            "report",
        ),
    )
    parser.add_argument("--pipeline", default="configs/queue_factorial/pipeline.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    pipeline = load_pipeline(args.pipeline)
    if args.command in {"validate", "manifest"}:
        result = validate(pipeline)
    elif args.command == "smoke":
        result = run_train(pipeline, args.index, smoke=True, resume=not args.no_resume)
    elif args.command == "train":
        result = run_train(pipeline, args.index, smoke=False, resume=not args.no_resume)
    elif args.command == "smoke-report":
        result = smoke_report(pipeline)
    elif args.command == "first-cell":
        result = run_first_cell(pipeline, resume=not args.no_resume)
    elif args.command == "gate-report":
        result = first_cell_gate_report(pipeline)
    elif args.command == "remaining":
        result = run_remaining(pipeline, args.index, resume=not args.no_resume)
    elif args.command == "eval":
        result = run_eval(pipeline, args.index)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
