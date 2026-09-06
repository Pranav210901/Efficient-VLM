from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.alignment_v3.efficiency_frontier import (
    _assert_fully_eval,
    _cuda_times,
    _hardware_guard,
    _quartiles,
    _sample_batch,
    _student_transform,
    _tokenize_native,
)
from src.alignment_v3.fingerprint import read_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.alignment_v3.training import load_training_checkpoint
from src.phase15.io_utils import atomic_csv, atomic_json
from src.utils.config import load_config


OUT = ROOT / "results/text_aggregation_study/diagnostics/m_t1_repeated_latency"
SEEDS = (42, 43, 44)
REPETITIONS = 10
WARMUP = 20
TIMED_REPEATS = 100
LATENCY_CEILING_MS = 9.480
ORDER_SEED = 20260801
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 10_000
CHECKPOINTS = {
    "M_T0": {
        "root": "checkpoints/token_aggregator_scale_training/C4_d256_h8_b2_ff512/sensitivity",
        "epochs": {42: 14, 43: 22, 44: 20},
    },
    "M_T1": {
        "root": "checkpoints/text_aggregation_study/M_T1/sensitivity",
        "epochs": {42: 22, 43: 22, 44: 20},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_dir(arm: str, seed: int) -> Path:
    return ROOT / str(CHECKPOINTS[arm]["root"]) / f"distill_strength_1p0__seed_{seed}"


def _load(arm: str, seed: int, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    directory = _checkpoint_dir(arm, seed)
    epoch = int(CHECKPOINTS[arm]["epochs"][seed])
    config_path = directory / "config.yaml"
    fingerprint_path = directory / "fingerprint.json"
    checkpoint_path = directory / f"epoch_{epoch:02d}.pt"
    missing = [path for path in (config_path, fingerprint_path, checkpoint_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen latency inputs: {missing}")
    config = load_config(config_path)
    fingerprint = read_fingerprint(fingerprint_path)
    if fingerprint is None:
        raise RuntimeError(f"invalid fingerprint: {fingerprint_path}")
    model = build_model(config).to(device).eval()
    load_training_checkpoint(
        checkpoint_path,
        model,
        device=device,
        expected_fingerprint=fingerprint,
    )
    _assert_fully_eval(model, f"{arm}:seed_{seed}:epoch_{epoch}")
    identity = {
        "arm": arm,
        "seed": seed,
        "selected_epoch": epoch,
        "selection_source": "Flickr30k validation, frozen before this diagnostic",
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "fingerprint": fingerprint.digest,
        "config_path": str(config_path.relative_to(ROOT)),
        "config": config,
    }
    return model, identity


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("bootstrap CI needs at least two scalar observations")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"arm", "seed", "repetition", "q3_ms"}
    if not required.issubset(raw.columns):
        raise ValueError(f"raw measurements missing {sorted(required - set(raw.columns))}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for arm in ("M_T0", "M_T1"):
        arm_rows = raw.loc[raw.arm == arm]
        for seed_label, selected in [
            *((str(seed), arm_rows.loc[arm_rows.seed == seed]) for seed in SEEDS),
            ("pooled", arm_rows),
        ]:
            values = selected.q3_ms.to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(values, rng)
            rows.append({
                "comparison": arm,
                "seed": seed_label,
                "n": len(values),
                "median_q3_ms": float(np.median(values)),
                "mean_q3_ms": float(np.mean(values)),
                "sd_q3_ms": float(np.std(values, ddof=1)),
                "mean_q3_bootstrap_ci95_low_ms": low,
                "mean_q3_bootstrap_ci95_high_ms": high,
                "raw_q3_min_ms": float(np.min(values)),
                "raw_q3_max_ms": float(np.max(values)),
            })

    paired = raw.pivot(index=["seed", "repetition"], columns="arm", values="q3_ms").reset_index()
    if paired[["M_T0", "M_T1"]].isna().any().any() or len(paired) != len(SEEDS) * REPETITIONS:
        raise RuntimeError("paired repeated profile is incomplete")
    paired["M_T1_minus_M_T0_ms"] = paired.M_T1 - paired.M_T0
    for seed_label, selected in [
        *((str(seed), paired.loc[paired.seed == seed]) for seed in SEEDS),
        ("pooled", paired),
    ]:
        values = selected.M_T1_minus_M_T0_ms.to_numpy(dtype=np.float64)
        low, high = _bootstrap_mean_ci(values, rng)
        rows.append({
            "comparison": "M_T1_minus_M_T0_paired",
            "seed": seed_label,
            "n": len(values),
            "median_q3_ms": float(np.median(values)),
            "mean_q3_ms": float(np.mean(values)),
            "sd_q3_ms": float(np.std(values, ddof=1)),
            "mean_q3_bootstrap_ci95_low_ms": low,
            "mean_q3_bootstrap_ci95_high_ms": high,
            "raw_q3_min_ms": float(np.min(values)),
            "raw_q3_max_ms": float(np.max(values)),
        })
    return pd.DataFrame(rows), paired


def _markdown(payload: dict[str, Any], summary: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        return f"{float(value):.6f}"

    lines = [
        "# Paired repeated-measurement latency diagnostic: M_T1 vs M_T0",
        "",
        "Diagnostic only. No retraining, checkpoint/epoch reselection, automatic gate decision, or ceiling modification was performed.",
        "",
        f"- Slurm job ID: `{payload['slurm_job_id']}`",
        f"- Node: `{payload['node']}`",
        f"- GPU: `{payload['gpu_model']}`",
        f"- Frozen ceiling (context only): `{LATENCY_CEILING_MS:.3f} ms Q3`",
        f"- Protocol: batch 64, {WARMUP} warm-up iterations and {TIMED_REPEATS} timed iterations per independent profile",
        f"- Repetitions: {REPETITIONS} per seed per arm; {len(SEEDS) * REPETITIONS * 2} total profiles",
        "- Ordering: deterministically randomized within every seed/repetition pair",
        f"- CI: {BOOTSTRAP_REPLICATES:,}-replicate percentile bootstrap CI for the mean Q3",
        "",
        "## Summary",
        "",
        "| Comparison | Seed | n | Median Q3 (ms) | Mean Q3 (ms) | SD (ms) | Mean Q3 95% CI (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['comparison']} | {row['seed']} | {row['n']} | "
            f"{fmt(row['median_q3_ms'])} | {fmt(row['mean_q3_ms'])} | "
            f"{fmt(row['sd_q3_ms'])} | [{fmt(row['mean_q3_bootstrap_ci95_low_ms'])}, "
            f"{fmt(row['mean_q3_bootstrap_ci95_high_ms'])}] |"
        )
    lines.extend(["", "## Frozen checkpoint identities", ""])
    for row in payload["checkpoints"]:
        lines.append(
            f"- {row['arm']} seed {row['seed']}: epoch {row['selected_epoch']}; "
            f"`{row['checkpoint_path']}`; SHA-256 `{row['checkpoint_sha256']}`"
        )
    lines.extend([
        "", "## Artifacts", "",
        f"- Raw profiles: `{payload['artifacts']['raw_csv']}`",
        f"- Summary statistics: `{payload['artifacts']['summary_csv']}`",
        f"- Paired values: `{payload['artifacts']['paired_csv']}`",
        f"- Machine-readable report: `{payload['artifacts']['report_json']}`",
        f"- This summary: `{payload['artifacts']['report_md']}`",
        "", "No PASS/FAIL verdict is emitted by this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    images, captions = _sample_batch(ROOT / "data/flickr30k/validation.csv", 64)
    models: dict[tuple[str, int], torch.nn.Module] = {}
    identities: list[dict[str, Any]] = []
    image_tensors: dict[str, torch.Tensor] = {}
    token_batches: dict[tuple[str, int], object] = {}
    for arm in ("M_T0", "M_T1"):
        for seed in SEEDS:
            model, identity = _load(arm, seed, device)
            models[(arm, seed)] = model
            identities.append({key: value for key, value in identity.items() if key != "config"})
            if arm not in image_tensors:
                transform = _student_transform(identity["config"])
                image_tensors[arm] = torch.stack([transform(image) for image in images]).to(device)
            token_batches[(arm, seed)] = _tokenize_native(
                model, captions, pad_to_native_context=False
            ).to(device)

    schedule: list[tuple[int, int, str, int]] = []
    order_rng = random.Random(ORDER_SEED)
    order_index = 0
    for seed in SEEDS:
        for repetition in range(1, REPETITIONS + 1):
            arms = ["M_T0", "M_T1"]
            order_rng.shuffle(arms)
            for arm in arms:
                schedule.append((order_index, seed, arm, repetition))
                order_index += 1

    raw_rows: list[dict[str, Any]] = []
    for order_index, seed, arm, repetition in schedule:
        model = models[(arm, seed)]
        images_gpu = image_tensors[arm]
        tokens = token_batches[(arm, seed)]
        values = _cuda_times(
            lambda: (model.encode_image(images_gpu), model.encode_text_tokens(tokens)),
            WARMUP,
            TIMED_REPEATS,
            device,
        )
        timing = _quartiles(values)
        raw_rows.append({
            "order_index": order_index,
            "arm": arm,
            "seed": seed,
            "selected_epoch": int(CHECKPOINTS[arm]["epochs"][seed]),
            "repetition": repetition,
            "median_ms": timing["median"],
            "q1_ms": timing["q1"],
            "q3_ms": timing["q3"],
            "iqr_ms": timing["iqr"],
            "node": provenance["node"],
            "gpu_model": provenance["gpu_model"],
            "precision": provenance["precision"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_under_slurm"),
        })
    raw = pd.DataFrame(raw_rows)
    summary, paired = summarize(raw)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_csv": str((OUT / "raw_profiles.csv").resolve()),
        "summary_csv": str((OUT / "summary_statistics.csv").resolve()),
        "paired_csv": str((OUT / "paired_differences.csv").resolve()),
        "report_json": str((OUT / "report.json").resolve()),
        "report_md": str((OUT / "report.md").resolve()),
    }
    payload = {
        "status": "COMPLETE",
        "diagnostic_only": True,
        "automatic_gate_decision": False,
        "latency_ceiling_ms_context_only": LATENCY_CEILING_MS,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_under_slurm"),
        "node": provenance["node"],
        "gpu_model": provenance["gpu_model"],
        "precision": provenance["precision"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "protocol": {
            "batch_size": 64,
            "warmup_iterations_per_profile": WARMUP,
            "timed_iterations_per_profile": TIMED_REPEATS,
            "independent_profiles_per_seed_per_arm": REPETITIONS,
            "total_independent_profiles": len(raw),
            "order": "deterministically randomized within each seed/repetition pair",
            "order_seed": ORDER_SEED,
            "ci": f"percentile bootstrap CI for mean Q3; {BOOTSTRAP_REPLICATES} replicates",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "same_allocation": True,
            "dynamic_padding": True,
        },
        "checkpoints": identities,
        "summary": summary.to_dict("records"),
        "artifacts": paths,
    }
    atomic_csv(raw, OUT / "raw_profiles.csv")
    atomic_csv(summary, OUT / "summary_statistics.csv")
    atomic_csv(paired, OUT / "paired_differences.csv")
    atomic_json(payload, OUT / "report.json")
    (OUT / "report.md").write_text(_markdown(payload, summary))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
