"""Queue identification control: is staleness itself the cause of the damage?

The queue factorial established a dissociation -- conditions matched on
measured projector drift differed in damage -- but not an identification,
because every queue-enabled arm confounds two things: the residents are stale,
and the residents are a particular set of negatives that the queue-free control
does not see at all.

This control separates them.  ``both_fresh`` keeps the negative set, its
ordering, its capacity, its eviction schedule and its positive masks exactly as
``both`` has them, and changes one quantity: residents are re-projected from
their stored pre-projection features through the *current* projector before
entering the denominator, so their staleness is zero.

  both  vs  both_fresh  ->  isolates staleness
  both_fresh vs none    ->  isolates the extra negatives

If both_fresh recovers the queue-free score, staleness is the mechanism.  If
both_fresh stays damaged, the harm was never staleness and the chapter's
headline needs rewording.  Either outcome is publishable; only the first is the
one the drift argument predicts.

The matched ``both`` and ``none`` arms already exist in the queue factorial at
these settings and are not re-run.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from src.alignment_v3.fingerprint import read_fingerprint, write_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.queue_factorial import (
    ROOT,
    build_config,
    capacity,
    hardware_guard,
    load_pipeline,
    prediction_gate,
)
from src.alignment_v3.training import load_training_checkpoint, train
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_json
from src.training.evaluate import extract_embeddings
from src.utils.config import deep_update
from src.utils.device import get_device


# Queue-free batch-1024 controls, reused rather than re-run.  Mirrors the
# per-seed source mapping asserted in queue_factorial.reuse_audit.
WAVE0_CONTROLS = {
    42: "results/alignment_v4_wave0/wave0-lr-control/infonce_no_queue__captions_all__b1024__lr_3p0__seed_42/metrics.json",
    43: "results/alignment_v4_wave0/wave0-winner-confirmation/infonce_no_queue__captions_all__b1024__lr_3p0__seed_43/metrics.json",
    44: "results/alignment_v4_wave0/wave0-winner-confirmation/infonce_no_queue__captions_all__b1024__lr_3p0__seed_44/metrics.json",
}


def identification_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    """One fresh-queue cell per (age, seed).  Batch is pinned by the pipeline."""
    spec = pipeline["identification"]
    batch = int(spec["batch_size"])
    jobs: list[dict[str, Any]] = []
    for age in spec["target_ages"]:
        for seed in spec["seeds"]:
            age = int(age)
            jobs.append(
                {
                    "index": len(jobs),
                    "run_id": f"b{batch}__age_{age}__both_fresh__seed_{seed}",
                    "experiment_id": "queue_identification",
                    "batch_size": batch,
                    "target_age_steps": age,
                    "memory_queue_size": capacity(batch, age),
                    "queue_mode": "both_fresh",
                    "seed": int(seed),
                    "memberships": ["identification_fresh"],
                    # The already-completed factorial arms this is read against.
                    "matched_stale_run_id": f"b{batch}__age_{age}__both__seed_{seed}",
                    # The batch-1024 queue-free control was never re-run inside
                    # the factorial; it is reused from wave 0, per the pipeline
                    # reuse block and queue_factorial.reuse_audit.
                    "matched_control_path": WAVE0_CONTROLS[int(seed)],
                }
            )
    expected = len(spec["target_ages"]) * len(spec["seeds"])
    if len(jobs) != expected:
        raise AssertionError(f"expected {expected} identification cells, got {len(jobs)}")
    return jobs


def _select(jobs: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    if index is None:
        raise SystemExit("--index is required")
    if not 0 <= index < len(jobs):
        raise SystemExit(f"--index must be in [0,{len(jobs) - 1}]")
    return jobs[index]


def run_train(pipeline: dict[str, Any], index: int | None, *, resume: bool) -> dict[str, Any]:
    prediction_gate(pipeline)
    provenance = hardware_guard()
    job = _select(identification_jobs(pipeline), index)
    config = build_config(pipeline, job)
    from src.alignment_v3.runner import _fingerprint

    fingerprint = _fingerprint(config)
    result = train(config, fingerprint, resume=resume)
    return {"run_id": job["run_id"], **provenance, **result}


def run_eval(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = hardware_guard()
    job = _select(identification_jobs(pipeline), index)
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
        "fingerprint_digest": fingerprint.digest,
    }
    atomic_json(row, destination / "metrics.json")
    write_fingerprint(destination / "fingerprint.json", fingerprint)
    return row


def _read_metric(path) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    for key in ("dev_mean_r_at_1", "mean_r_at_1", "mean_R@1"):
        if key in payload:
            return float(payload[key])
    return None


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Three-way read: none (control), both (stale), both_fresh (this control)."""
    factorial_runs = ROOT / pipeline["factorial_output_root"] / "runs"
    fresh_runs = ROOT / pipeline["output_root"] / "runs"
    rows = []
    for job in identification_jobs(pipeline):
        fresh = _read_metric(fresh_runs / job["run_id"] / "metrics.json")
        stale = _read_metric(factorial_runs / job["matched_stale_run_id"] / "metrics.json")
        control = _read_metric(ROOT / job["matched_control_path"])
        row = {
            "run_id": job["run_id"],
            "target_age_steps": job["target_age_steps"],
            "seed": job["seed"],
            "none_control": control,
            "stale_queue": stale,
            "fresh_queue": fresh,
        }
        if None not in (control, stale, fresh):
            # Positive = damage recovered by removing staleness alone.
            row["staleness_attributable_pp"] = 100.0 * (fresh - stale)
            row["residual_damage_pp"] = 100.0 * (control - fresh)
            total = control - stale
            row["fraction_of_damage_from_staleness"] = (
                None if abs(total) < 1e-9 else float((fresh - stale) / total)
            )
        rows.append(row)
    complete = [r for r in rows if "staleness_attributable_pp" in r]
    summary: dict[str, Any] = {"status": "COMPLETE" if len(complete) == len(rows) else "PARTIAL", "rows": rows}
    if complete:
        fractions = [r["fraction_of_damage_from_staleness"] for r in complete if r["fraction_of_damage_from_staleness"] is not None]
        summary["mean_fraction_of_damage_from_staleness"] = (
            sum(fractions) / len(fractions) if fractions else None
        )
        summary["interpretation"] = (
            "fraction near 1.0: staleness identified as the mechanism; "
            "near 0.0: the harm is the extra negatives, not their age; "
            "intermediate: both contribute and the chapter must say so."
        )
    destination = ROOT / pipeline["output_root"] / "report"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, destination / "report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["jobs", "train", "eval", "report"])
    parser.add_argument("--pipeline", default="configs/queue_factorial/pipeline_fresh.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    pipeline = load_pipeline(ROOT / args.pipeline)
    if args.command == "jobs":
        print(json.dumps(identification_jobs(pipeline), indent=2))
        return
    if args.command == "train":
        print(json.dumps(run_train(pipeline, args.index, resume=not args.no_resume), indent=2, default=str))
        return
    if args.command == "eval":
        print(json.dumps(run_eval(pipeline, args.index), indent=2, default=str))
        return
    print(json.dumps(report(pipeline), indent=2))


if __name__ == "__main__":
    main()
