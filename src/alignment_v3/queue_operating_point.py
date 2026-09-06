"""Queue decomposition repeated at the M-T1 operating point.

The queue factorial (Q02), the fresh-reprojection identification control (Q04)
and the mechanism closure (Q05) all ran on the Wave 0 projection-only student,
whose queue-free score is 37.32% mean R@1.  The endpoint the dissertation
actually reports is M-T1 at 54.49%.  Every queue conclusion is therefore stated
at an operating point roughly seventeen points below the shipped model, and the
open question is whether the decomposition survives there.

This module repeats the three-arm decomposition on the M-T1 architecture:

    none        queue-free control
    both        stale queue, as the original recipe used it
    both_fresh  same entries and capacity, re-projected through the current
                projector so that projector staleness is zero

    both vs both_fresh  ->  staleness component
    both_fresh vs none  ->  residual (count / composition / weighting)

Unlike the identification control, no matched arm exists at this operating
point, so all three are trained here: 3 modes x {16, 64} x {42, 43, 44} = 18.

The student config is taken from configs/queue_operating_point/base.yaml, which
extends the M-T1 base unchanged.  In particular the learning rate is NOT pinned
to the factorial's Wave 0 value: it resolves exactly as it does for the shipped
model, because pinning it would change a second variable and defeat the point.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from src.alignment_v3.fingerprint import read_fingerprint, write_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.queue_factorial import (
    ROOT,
    capacity,
    hardware_guard,
    load_pipeline,
    prediction_gate,
)
from src.alignment_v3.runner import build_job_config
from src.alignment_v3.runner import load_pipeline as load_runner_pipeline
from src.alignment_v3.training import load_training_checkpoint, train
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_json
from src.training.evaluate import extract_embeddings
from src.utils.config import deep_update
from src.utils.device import get_device


def operating_point_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    """One cell per (mode, age, seed).  The queue-free arm ignores age."""
    spec = pipeline["operating_point"]
    batch = int(spec["batch_size"])
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for age in spec["target_ages"]:
        for mode in spec["queue_modes"]:
            for seed in spec["seeds"]:
                age_i, seed_i = int(age), int(seed)
                cap = 0 if mode == "none" else capacity(batch, age_i)
                # The queue-free control does not depend on the age axis; train
                # it once per seed and read it against both ages.
                key = (mode, cap, seed_i)
                if key in seen:
                    continue
                seen.add(key)
                run_age = 0 if mode == "none" else age_i
                jobs.append(
                    {
                        "index": len(jobs),
                        "run_id": f"mt1__b{batch}__age_{run_age}__{mode}__seed_{seed_i}",
                        "experiment_id": "queue_operating_point",
                        "batch_size": batch,
                        "target_age_steps": run_age,
                        "memory_queue_size": cap,
                        "queue_mode": mode,
                        "seed": seed_i,
                        "memberships": ["operating_point", f"mode_{mode}"],
                    }
                )
    expected = len(spec["seeds"]) * (1 + (len(spec["queue_modes"]) - 1) * len(spec["target_ages"]))
    if len(jobs) != expected:
        raise AssertionError(f"expected {expected} operating-point cells, got {len(jobs)}")
    return jobs


def _m_t1_pipeline(pipeline: dict[str, Any]) -> dict[str, Any]:
    """The shipped M-T1 pipeline, loaded through the runner's own loader."""
    loaded = load_runner_pipeline(ROOT / str(pipeline["operating_point"]["m_t1_pipeline"]))
    if isinstance(loaded, tuple):
        loaded = next(item for item in loaded if isinstance(item, dict))
    return loaded


def build_config(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Exactly the M-T1 training config, with the queue axis applied on top.

    The config is produced by the same runner entry point that produced the
    shipped M-T1 checkpoints -- ``build_job_config(..., stage="sensitivity")``
    against the M-T1 pipeline -- so the wave-1 locked recipe, the x3 sweep
    multiplier, the MobileCLIP2 distillation strength and the 24-epoch schedule
    are all resolved by the project's own code rather than restated here.  Only
    the queue axis, the drift probe and the output paths are then overridden.
    """
    m_t1 = _m_t1_pipeline(pipeline)
    base_job = {
        "experiment_id": job["experiment_id"],
        "run_id": job["run_id"],
        "strength": float(pipeline["operating_point"]["distillation_strength"]),
        "distillation": True,
        "seed": int(job["seed"]),
    }
    config = build_job_config(m_t1, base_job, "sensitivity")

    save_dir = ROOT / pipeline["checkpoint_root"] / "runs" / job["run_id"]
    probe = (
        ROOT / pipeline["output_root"] / "runs" / job["run_id"] / "projector_probe.csv"
    ).relative_to(ROOT)
    resolved_lr = float(config["training"]["lr"])
    return deep_update(
        config,
        {
            "data": {
                "train_csv": pipeline["split"]["train_csv"],
                "val_csv": pipeline["split"]["dev_csv"],
                "num_workers": int(pipeline["resources"]["dataloader_workers"]),
            },
            # The queue axis is applied after the recipe merge, because the
            # wave-1 locked recipe pins memory_queue_size to 0 for M-T1.
            "recipe": {"memory_queue_size": int(job["memory_queue_size"])},
            "training": {
                "memory_queue_size": int(job["memory_queue_size"]),
                "queue_mode": job["queue_mode"],
                "save_dir": str(save_dir.relative_to(ROOT)),
            },
            "queue_diagnostics": dict(pipeline["queue_diagnostics"], probe_manifest=str(probe)),
            "provenance": {
                "split_manifest": pipeline["split"]["manifest"],
                "factorial_memberships": job["memberships"],
                "target_age_steps": int(job["target_age_steps"]),
                "operating_point": "M-T1 shipped frozen-encoder architecture",
                "operating_point_source_pipeline": str(
                    pipeline["operating_point"]["m_t1_pipeline"]
                ),
                "resolved_lr": resolved_lr,
                "learning_rate_pinned_from_wave0": False,
            },
        },
    )


def _select(jobs: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    if index is None:
        raise SystemExit("--index is required")
    if not 0 <= index < len(jobs):
        raise SystemExit(f"--index must be in [0,{len(jobs) - 1}]")
    return jobs[index]


def run_train(pipeline: dict[str, Any], index: int | None, *, resume: bool) -> dict[str, Any]:
    prediction_gate(pipeline)
    provenance = hardware_guard()
    job = _select(operating_point_jobs(pipeline), index)
    config = build_config(pipeline, job)
    from src.alignment_v3.runner import _fingerprint

    fingerprint = _fingerprint(config)
    result = train(config, fingerprint, resume=resume)
    return {"run_id": job["run_id"], **provenance, **result}


def run_eval(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    provenance = hardware_guard()
    job = _select(operating_point_jobs(pipeline), index)
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
    """Decomposition per (age, seed), read exactly as the Q04 control was."""
    spec = pipeline["operating_point"]
    batch = int(spec["batch_size"])
    runs = ROOT / pipeline["output_root"] / "runs"

    def metric(mode: str, age: int, seed: int) -> float | None:
        run_age = 0 if mode == "none" else age
        return _read_metric(runs / f"mt1__b{batch}__age_{run_age}__{mode}__seed_{seed}" / "metrics.json")

    rows = []
    for age in spec["target_ages"]:
        for seed in spec["seeds"]:
            age_i, seed_i = int(age), int(seed)
            control = metric("none", age_i, seed_i)
            stale = metric("both", age_i, seed_i)
            fresh = metric("both_fresh", age_i, seed_i)
            row = {
                "target_age_steps": age_i,
                "seed": seed_i,
                "none_control": control,
                "stale_queue": stale,
                "fresh_queue": fresh,
            }
            if None not in (control, stale, fresh):
                row["staleness_attributable_pp"] = 100.0 * (fresh - stale)
                row["residual_damage_pp"] = 100.0 * (control - fresh)
                total = control - stale
                row["total_damage_pp"] = 100.0 * total
                row["fraction_of_damage_from_staleness"] = (
                    None if abs(total) < 1e-9 else float((fresh - stale) / total)
                )
            rows.append(row)

    complete = [r for r in rows if "staleness_attributable_pp" in r]
    summary: dict[str, Any] = {
        "status": "COMPLETE" if len(complete) == len(rows) else "PARTIAL",
        "operating_point": spec["reference_endpoint"],
        "prior_study_staleness_share_percent": 31.79,
        "rows": rows,
    }
    if complete:
        fractions = [
            r["fraction_of_damage_from_staleness"]
            for r in complete
            if r["fraction_of_damage_from_staleness"] is not None
        ]
        if fractions:
            summary["mean_fraction_of_damage_from_staleness"] = sum(fractions) / len(fractions)
        summary["interpretation"] = (
            "Compare the staleness share against the 31.79% measured on the Wave 0 "
            "student. A similar share replicates the mechanism at the shipped "
            "operating point; a materially larger share means the earlier result "
            "understated staleness because the weaker projector moved more; a "
            "smaller share means the residual mechanisms dominate even further."
        )
    destination = ROOT / pipeline["output_root"] / "report"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, destination / "report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["jobs", "train", "eval", "report"])
    parser.add_argument("--pipeline", default="configs/queue_operating_point/pipeline.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    pipeline = load_pipeline(ROOT / args.pipeline)
    if args.command == "jobs":
        print(json.dumps(operating_point_jobs(pipeline), indent=2))
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
