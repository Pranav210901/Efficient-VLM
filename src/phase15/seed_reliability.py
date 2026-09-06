"""Canonical repeated-seed evidence and conservative BLF reliability labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.configuration_ids import canonical_configuration_id, canonical_variant

from .io_utils import atomic_csv, atomic_json


T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
RELIABILITY_SCORES = {"supported": 1.0, "inconclusive": 0.5, "unsupported": 0.0, "not_tested": 0.25}


def _interval(mean: float, std: float, n: int) -> tuple[float, float]:
    if n < 2 or not np.isfinite(std):
        return float("nan"), float("nan")
    half = T_CRITICAL_95.get(n - 1, 1.96) * std / np.sqrt(n)
    return mean - half, mean + half


def build_seed_reliability(
    seed_results: pd.DataFrame,
    paired_deltas: pd.DataFrame | None = None,
    *,
    metric: str = "coco5_i2t_R@1",
    minimum_effect: float = 0.0,
    minimum_runs: int = 5,
    minimum_wins: int = 4,
) -> pd.DataFrame:
    required = {"comparison", "vision_encoder", "text_encoder", "variant", "seed", metric}
    missing = required - set(seed_results.columns)
    if missing:
        raise ValueError(f"Seed results are missing columns: {sorted(missing)}")
    work = seed_results.copy()
    work["config_id"] = [canonical_configuration_id(row) for row in work.to_dict("records")]
    work["variant"] = work["variant"].map(canonical_variant)
    if work.duplicated(["config_id", "seed"]).any():
        raise ValueError("Seed results contain duplicate canonical configuration/seed rows")
    paired = paired_deltas.copy() if paired_deltas is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for config_id, group in work.groupby("config_id", sort=True):
        group = group.sort_values("seed")
        values = pd.to_numeric(group[metric], errors="raise")
        n = int(group["seed"].nunique())
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        low, high = _interval(mean, std, n)
        variant = str(group["variant"].iloc[0])
        comparison = str(group["comparison"].iloc[0])
        matched_mean = matched_low = matched_high = float("nan")
        wins = ties = losses = 0
        if variant == "baseline":
            status = "supported" if n >= minimum_runs else "inconclusive"
        else:
            candidate = paired[
                paired["comparison"].eq(comparison)
                & paired["variant"].map(canonical_variant).eq(variant)
            ] if not paired.empty else pd.DataFrame()
            if candidate.empty:
                status = "inconclusive" if n >= minimum_runs else "not_tested"
            else:
                value = candidate.iloc[0]
                matched_mean = float(value["mean_delta"])
                half = float(value.get("delta_ci95_half_width", float("nan")))
                matched_low = matched_mean - half
                matched_high = matched_mean + half
                wins, ties, losses = int(value["wins"]), int(value["ties"]), int(value["losses"])
                if n >= minimum_runs and matched_mean > minimum_effect and matched_low > 0 and wins >= minimum_wins:
                    status = "supported"
                elif np.isfinite(matched_high) and matched_high < 0:
                    status = "unsupported"
                else:
                    status = "inconclusive"
        rows.append(
            {
                "config_id": config_id,
                "comparison": comparison,
                "variant": variant,
                "seed_runs": n,
                "seed_values": json.dumps(group["seed"].astype(int).tolist()),
                "mean_metric": mean,
                "standard_deviation": std,
                "confidence_interval_low": low,
                "confidence_interval_high": high,
                "matched_delta_mean": matched_mean,
                "matched_delta_ci_low": matched_low,
                "matched_delta_ci_high": matched_high,
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "reliability_status": status,
                "seed_reliability_score": RELIABILITY_SCORES[status],
                "criterion_minimum_effect": minimum_effect,
                "criterion_minimum_runs": minimum_runs,
                "criterion_minimum_wins": minimum_wins,
            }
        )
    return pd.DataFrame(rows)


def run_seed_reliability_mapping(
    project_root: str | Path,
    *,
    metric: str = "coco5_i2t_R@1",
    minimum_effect: float = 0.0,
    minimum_runs: int = 5,
    minimum_wins: int = 4,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    seed_path = root / "results/seed_sweep_results.csv"
    paired_path = root / "results/seed_sweep_paired_deltas.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed results are missing: {seed_path}")
    frame = build_seed_reliability(
        pd.read_csv(seed_path),
        pd.read_csv(paired_path) if paired_path.exists() else None,
        metric=metric,
        minimum_effect=minimum_effect,
        minimum_runs=minimum_runs,
        minimum_wins=minimum_wins,
    )
    output = root / "results/phase15/expert_selection"
    atomic_csv(frame, output / "seed_reliability.csv")
    atomic_json(
        {
            "metric": metric,
            "criterion": {"minimum_effect": minimum_effect, "minimum_runs": minimum_runs, "minimum_wins": minimum_wins, "confidence_interval_lower_bound_must_exceed_zero": True},
            "configurations": json.loads(frame.to_json(orient="records")),
        },
        output / "seed_reliability.json",
    )
    return frame
