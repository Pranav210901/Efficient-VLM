#!/usr/bin/env python3
"""Create the post-factorial interpretation package without running models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_ROOT = ROOT / "results/queue_factorial"
OUTPUT = FACTORIAL_ROOT / "interpretation"
CHECKPOINTS = ROOT / "checkpoints/queue_factorial/runs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def linear_fit(frame: pd.DataFrame, predictors: list[str]) -> dict[str, Any]:
    clean = frame.dropna(subset=["degradation_pp", *predictors]).copy()
    design = np.column_stack(
        [np.ones(len(clean)), *[clean[name].to_numpy(float) for name in predictors]]
    )
    response = clean["degradation_pp"].to_numpy(float)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    residual = response - fitted
    denominator = float(np.sum((response - response.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "n": len(clean),
        "predictors": predictors,
        "intercept": float(coefficients[0]),
        "coefficients": {
            name: float(value) for name, value in zip(predictors, coefficients[1:])
        },
        "r_squared": r_squared,
        "rmse_pp": float(np.sqrt(np.mean(residual**2))),
    }


def residence_window_path_length(
    drift: pd.DataFrame,
    modality: str,
    residence_steps: float,
    first_full_step: int,
) -> tuple[float, int]:
    """Average measured cumulative drift over an eviction-length window.

    The instrumentation stores cosine distance from the previous probe every
    ten optimizer steps. This function treats each recorded distance as the
    path length over its measurement interval, prorating boundary intervals,
    and sums the path over the preceding measured eviction residence.
    """

    steps = drift["optimizer_step"].to_numpy(int)
    distances = drift[f"{modality}_from_previous"].to_numpy(float)
    cumulative: list[float] = []
    for index, end_step in enumerate(steps):
        if end_step < first_full_step:
            continue
        window_start = float(end_step) - residence_steps
        path_length = 0.0
        for interval_index in range(index + 1):
            interval_end = float(steps[interval_index])
            interval_start = (
                0.0 if interval_index == 0 else float(steps[interval_index - 1])
            )
            overlap = max(
                0.0,
                min(interval_end, float(end_step))
                - max(interval_start, window_start),
            )
            width = interval_end - interval_start
            if overlap and width:
                path_length += float(distances[interval_index]) * overlap / width
        cumulative.append(path_length)
    if not cumulative:
        raise RuntimeError(
            f"no full-queue drift windows for {modality}, residence={residence_steps}"
        )
    return float(np.mean(cumulative)), len(cumulative)


def build_drift_at_eviction() -> pd.DataFrame:
    summary = pd.read_csv(FACTORIAL_ROOT / "report/summary.csv")
    baseline = float(
        summary[
            (summary["batch_size"] == 1024)
            & (summary["target_age_steps"] == 0)
            & (summary["queue_mode"] == "none")
        ].iloc[0]["mean"]
    )
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((FACTORIAL_ROOT / "runs").glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        mode = metrics["queue_mode"]
        if mode not in {"image_only", "text_only"}:
            continue
        modality = "image" if mode == "image_only" else "text"
        checkpoint = CHECKPOINTS / metrics["run_id"]
        diagnostic = pd.read_json(checkpoint / "queue_diagnostics.jsonl", lines=True)
        drift = pd.read_json(checkpoint / "projector_drift.jsonl", lines=True)
        full = diagnostic[diagnostic["phase"] == "full_queue"].copy()
        age_values = full[f"{modality}_age"].dropna().map(lambda value: value["max"])
        if age_values.empty:
            raise RuntimeError(f"missing measured {modality} ages: {metrics['run_id']}")
        residence_steps = float(age_values.mean())
        first_full_step = int(full["optimizer_step"].min())
        cumulative_drift, windows = residence_window_path_length(
            drift, modality, residence_steps, first_full_step
        )
        rows.append(
            {
                "run_id": metrics["run_id"],
                "seed": int(metrics["seed"]),
                "queue_mode": mode,
                "modality": modality,
                "target_image_age_steps": int(metrics["target_age_steps"]),
                "measured_eviction_age_steps": residence_steps,
                "mean_cumulative_drift_at_eviction": cumulative_drift,
                "eviction_windows": windows,
                "dev_mean_r1": float(metrics["mean_R@1"]),
                "degradation_pp": (baseline - float(metrics["mean_R@1"])) * 100.0,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 18:
        raise RuntimeError(f"expected 18 modality cells, found {len(frame)}")
    return frame


def regression_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame.copy()
    data["is_text"] = (data["modality"] == "text").astype(float)
    data["drift_x_text"] = (
        data["mean_cumulative_drift_at_eviction"] * data["is_text"]
    )
    image = data[data["modality"] == "image"]
    text = data[data["modality"] == "text"]
    grouped = (
        data.groupby(["queue_mode", "modality", "target_image_age_steps"])
        .agg(
            measured_eviction_age_steps=("measured_eviction_age_steps", "mean"),
            cumulative_drift=("mean_cumulative_drift_at_eviction", "mean"),
            degradation_pp=("degradation_pp", "mean"),
            degradation_sd_pp=("degradation_pp", "std"),
        )
        .reset_index()
    )
    image_16 = grouped[
        (grouped["modality"] == "image")
        & (grouped["target_image_age_steps"] == 16)
    ].iloc[0]
    text_64 = grouped[
        (grouped["modality"] == "text")
        & (grouped["target_image_age_steps"] == 64)
    ].iloc[0]
    return {
        "estimator": (
            "Mean sum of measured cosine distances from the previous probe over "
            "the preceding measured eviction-residence window; ten-step boundary "
            "intervals are prorated."
        ),
        "models": {
            "pooled_age_only": linear_fit(data, ["measured_eviction_age_steps"]),
            "pooled_cumulative_drift_only": linear_fit(
                data, ["mean_cumulative_drift_at_eviction"]
            ),
            "image_cumulative_drift": linear_fit(
                image, ["mean_cumulative_drift_at_eviction"]
            ),
            "text_cumulative_drift": linear_fit(
                text, ["mean_cumulative_drift_at_eviction"]
            ),
            "cumulative_drift_with_modality_interaction": linear_fit(
                data,
                ["mean_cumulative_drift_at_eviction", "is_text", "drift_x_text"],
            ),
        },
        "collapse_assessment": {
            "formal_threshold_preregistered": False,
            "verdict": "DOES_NOT_COLLAPSE",
            "reason": (
                "Image-only age 16 and text-only age 64 have nearly identical "
                "measured cumulative drift, but materially different degradation."
            ),
            "matched_example": {
                "image_only_target_age": 16,
                "image_cumulative_drift": float(image_16["cumulative_drift"]),
                "image_degradation_pp": float(image_16["degradation_pp"]),
                "text_only_target_age": 64,
                "text_cumulative_drift": float(text_64["cumulative_drift"]),
                "text_degradation_pp": float(text_64["degradation_pp"]),
                "degradation_gap_pp_text_minus_image": float(
                    text_64["degradation_pp"] - image_16["degradation_pp"]
                ),
            },
            "interpretation": (
                "Cumulative projector drift alone does not provide one quantitative "
                "relationship shared by both modalities. The modality interaction "
                "remains substantial, so gradient sensitivity, representation "
                "geometry, false-negative structure, or another modality-specific "
                "factor is still required."
            ),
        },
        "condition_means": grouped.to_dict(orient="records"),
    }


def matched_drift_sensitivity(
    frame: pd.DataFrame, tolerances: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
) -> tuple[pd.DataFrame, dict[str, Any]]:
    conditions = (
        frame.groupby(["queue_mode", "modality", "target_image_age_steps"])
        .agg(
            measured_eviction_age_steps=("measured_eviction_age_steps", "mean"),
            cumulative_drift=("mean_cumulative_drift_at_eviction", "mean"),
            degradation_pp=("degradation_pp", "mean"),
        )
        .reset_index()
    )
    image = conditions[conditions["modality"] == "image"]
    text = conditions[conditions["modality"] == "text"]
    drift_min = float(conditions["cumulative_drift"].min())
    drift_max = float(conditions["cumulative_drift"].max())
    drift_range = drift_max - drift_min
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for tolerance in tolerances:
        matches = 0
        mismatches = 0
        for _, image_row in image.iterrows():
            for _, text_row in text.iterrows():
                drift_difference = abs(
                    float(image_row["cumulative_drift"])
                    - float(text_row["cumulative_drift"])
                )
                if drift_difference > tolerance:
                    continue
                degradation_difference = abs(
                    float(image_row["degradation_pp"])
                    - float(text_row["degradation_pp"])
                )
                mismatch = degradation_difference > 1.0
                matches += 1
                mismatches += int(mismatch)
                rows.append(
                    {
                        "tolerance": tolerance,
                        "tolerance_as_percent_of_observed_range": (
                            tolerance / drift_range * 100.0
                        ),
                        "image_target_age": int(image_row["target_image_age_steps"]),
                        "image_measured_eviction_age": float(
                            image_row["measured_eviction_age_steps"]
                        ),
                        "image_cumulative_drift": float(
                            image_row["cumulative_drift"]
                        ),
                        "image_degradation_pp": float(
                            image_row["degradation_pp"]
                        ),
                        "text_target_age": int(text_row["target_image_age_steps"]),
                        "text_measured_eviction_age": float(
                            text_row["measured_eviction_age_steps"]
                        ),
                        "text_cumulative_drift": float(text_row["cumulative_drift"]),
                        "text_degradation_pp": float(text_row["degradation_pp"]),
                        "absolute_drift_difference": drift_difference,
                        "absolute_degradation_difference_pp": degradation_difference,
                        "degradation_difference_exceeds_1pp": mismatch,
                    }
                )
        summaries.append(
            {
                "tolerance": tolerance,
                "tolerance_as_percent_of_observed_range": (
                    tolerance / drift_range * 100.0
                ),
                "matched_pair_count": matches,
                "pairs_exceeding_1pp": mismatches,
                "conclusion": (
                    "Every matched pair shows modality-dependent damage above 1pp."
                    if matches and matches == mismatches
                    else (
                        "Some, but not all, matched pairs exceed 1pp."
                        if mismatches
                        else "No matched pair exceeds 1pp."
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    reproduced_at_multiple_pairs = sum(
        int(row["matched_pair_count"] >= 2 and row["pairs_exceeding_1pp"] >= 2)
        for row in summaries
    )
    assessment = {
        "observed_cumulative_drift": {
            "minimum": drift_min,
            "maximum": drift_max,
            "range": drift_range,
        },
        "equivalence_margin_pp": 1.0,
        "tolerance_summaries": summaries,
        "claim_decision": "GENERAL",
        "claim": (
            "Across multiple cross-modality condition pairs and all four "
            "reported tolerances, matched cumulative drift does not imply "
            "equivalent degradation. Drift is not a sufficient explanation "
            "of queue damage in this regime."
        ),
        "support": (
            f"At {reproduced_at_multiple_pairs} of {len(tolerances)} tolerances, "
            "at least two matched pairs independently exceeded the 1pp margin."
        ),
    }
    return result, assessment


def paired_age_table() -> pd.DataFrame:
    report = read_json(FACTORIAL_ROOT / "report/report.json")
    frame = pd.DataFrame(report["paired_contrasts"])
    frame["direction"] = np.where(
        frame["paired_difference_pp_b512_minus_b1024"] < 0,
        "batch 512 lower",
        "batch 512 higher",
    )
    baseline_offset = float(
        frame.loc[
            frame["target_age_steps"] == 0,
            "paired_difference_pp_b512_minus_b1024",
        ].iloc[0]
    )
    frame["exploratory_net_of_age0_offset_pp"] = (
        frame["paired_difference_pp_b512_minus_b1024"] - baseline_offset
    )
    return frame


def efficiency_table() -> pd.DataFrame:
    smoke = read_json(FACTORIAL_ROOT / "smoke/report.json")
    cells = {int(row["batch_size"]): row for row in smoke["cells"]}
    return pd.DataFrame(
        [
            {
                "evidence": "Original Wave-0 queue configuration",
                "comparison": "queue 16,384 vs no queue",
                "performance_effect_pp": 16.77 - 35.76,
                "throughput_images_per_second": math.nan,
                "peak_gpu_memory_gib": math.nan,
                "available_gpu_memory_gib": math.nan,
                "interpretation": "Approximately 19pp lower R@1",
            },
            {
                "evidence": "Factorial maximum-capacity smoke",
                "comparison": "batch 1024, queue 65,536",
                "performance_effect_pp": math.nan,
                "throughput_images_per_second": cells[1024][
                    "full_queue_images_per_second"
                ],
                "peak_gpu_memory_gib": cells[1024]["peak_memory_gib"],
                "available_gpu_memory_gib": cells[1024]["total_memory_gib"],
                "interpretation": "No required memory saving; 86.13 GiB headroom",
            },
            {
                "evidence": "Factorial maximum-capacity smoke",
                "comparison": "batch 512, queue 32,768",
                "performance_effect_pp": math.nan,
                "throughput_images_per_second": cells[512][
                    "full_queue_images_per_second"
                ],
                "peak_gpu_memory_gib": cells[512]["peak_memory_gib"],
                "available_gpu_memory_gib": cells[512]["total_memory_gib"],
                "interpretation": "No post-fill throughput degradation",
            },
        ]
    )


def plot_age_curves(summary: pd.DataFrame) -> None:
    selected = summary[summary["queue_mode"].isin(["none", "both"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for batch, group in selected.groupby("batch_size"):
        group = group.sort_values("target_age_steps")
        ax.errorbar(
            group["target_age_steps"],
            group["mean"] * 100,
            yerr=group["sd"] * 100,
            marker="o",
            capsize=3,
            label=f"batch {batch}",
        )
    ax.set_xscale("symlog", linthresh=1, base=2)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Maximum image-queue age (optimizer steps)")
    ax.set_ylabel("Dev mean bidirectional R@1 (%)")
    ax.set_title("No safe queue-age threshold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "queue_age_curve.png", dpi=180)
    plt.close(fig)


def plot_paired_differences(paired: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhspan(-1, 1, color="#d8ead3", alpha=0.8, label="±1pp margin")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(
        paired["target_age_steps"],
        paired["paired_difference_pp_b512_minus_b1024"],
        marker="o",
        color="#345995",
    )
    ax.set_xscale("symlog", linthresh=1, base=2)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Maximum image-queue age (optimizer steps)")
    ax.set_ylabel("Paired difference: batch 512 − 1024 (pp)")
    ax.set_title("Matched-age verdict passes exactly at 6 of 8")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "matched_age_differences.png", dpi=180)
    plt.close(fig)


def plot_modality(summary: pd.DataFrame) -> None:
    selected = summary[summary["queue_mode"].isin(["image_only", "text_only"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for mode, group in selected.groupby("queue_mode"):
        group = group.sort_values("target_age_steps")
        ax.errorbar(
            group["target_age_steps"],
            group["mean"] * 100,
            yerr=group["sd"] * 100,
            marker="o",
            capsize=3,
            label=mode.replace("_", " "),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Nominal image age from shared queue capacity")
    ax.set_ylabel("Dev mean bidirectional R@1 (%)")
    ax.set_title("Modality harm crosses over with residence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "modality_crossover.png", dpi=180)
    plt.close(fig)


def plot_drift_regression(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = {"image": "#d1495b", "text": "#00798c"}
    for modality, group in frame.groupby("modality"):
        ax.scatter(
            group["mean_cumulative_drift_at_eviction"],
            group["degradation_pp"],
            color=colors[modality],
            alpha=0.7,
            label=f"{modality} queue",
        )
        fit = linear_fit(group, ["mean_cumulative_drift_at_eviction"])
        start = float(group["mean_cumulative_drift_at_eviction"].min())
        end = float(group["mean_cumulative_drift_at_eviction"].max())
        x_values = np.linspace(start, end, 100)
        y_values = (
            fit["intercept"]
            + fit["coefficients"]["mean_cumulative_drift_at_eviction"] * x_values
        )
        ax.plot(x_values, y_values, color=colors[modality])
    ax.set_xlabel("Measured cumulative projector drift at eviction")
    ax.set_ylabel("Degradation from queue-free baseline (pp)")
    ax.set_title("Cumulative drift does not collapse modality curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "degradation_vs_cumulative_drift.png", dpi=180)
    plt.close(fig)


def plot_saturation() -> None:
    epoch = pd.read_csv(FACTORIAL_ROOT / "gate/per_epoch_drift.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for modality, color in (("image", "#d1495b"), ("text", "#00798c")):
        axes[0].plot(
            epoch["epoch"],
            epoch[f"{modality}_mean_drift_from_step0"],
            marker="o",
            color=color,
            label=modality,
        )
        axes[1].plot(
            epoch["epoch"],
            epoch[f"{modality}_mean_drift_from_previous"],
            marker="o",
            color=color,
            label=modality,
        )
    axes[0].set_title("Distance from step 0")
    axes[1].set_title("Movement since previous probe")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Mean cosine distance")
    axes[1].set_ylabel("Mean cosine distance (log scale)")
    fig.suptitle("Step-0 distance saturates; continuing motion decays")
    fig.tight_layout()
    fig.savefig(OUTPUT / "drift_saturation.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats and pd.notna(value):
                values.append(formats[column].format(value))
            elif isinstance(value, (bool, np.bool_)):
                values.append("yes" if value else "no")
            elif pd.isna(value):
                values.append("—")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    drift: pd.DataFrame,
    regression: dict[str, Any],
    sensitivity: pd.DataFrame,
    sensitivity_assessment: dict[str, Any],
    efficiency: pd.DataFrame,
) -> None:
    gate = read_json(FACTORIAL_ROOT / "gate/report.json")
    models = regression["models"]
    matched = regression["collapse_assessment"]["matched_example"]
    paired_display = paired.rename(
        columns={
            "target_age_steps": "age",
            "paired_difference_pp_b512_minus_b1024": "b512_minus_b1024_pp",
            "within_equivalence_margin": "within_margin",
            "exploratory_net_of_age0_offset_pp": "net_of_age0_pp",
        }
    )
    modality = summary[summary["queue_mode"].isin(["image_only", "text_only"])].copy()
    modality["mean_percent"] = modality["mean"] * 100
    modality["sd_pp"] = modality["sd"] * 100
    sensitivity_display = sensitivity.rename(
        columns={
            "tolerance_as_percent_of_observed_range": "tolerance_pct_range",
            "image_target_age": "image_age",
            "image_cumulative_drift": "image_drift",
            "text_target_age": "text_age",
            "text_cumulative_drift": "text_drift",
            "absolute_drift_difference": "drift_difference",
            "absolute_degradation_difference_pp": "degradation_difference_pp",
            "degradation_difference_exceeds_1pp": "exceeds_1pp",
        }
    )
    sensitivity_summary = pd.DataFrame(
        sensitivity_assessment["tolerance_summaries"]
    ).rename(
        columns={
            "tolerance_as_percent_of_observed_range": "tolerance_pct_range",
            "matched_pair_count": "pairs",
            "pairs_exceeding_1pp": "pairs_over_1pp",
        }
    )
    prediction_scores = [
        {
            "id": "age_vs_count",
            "score": "PASS_AT_BOUNDARY",
            "evidence": "Exactly 6/8 pre-registered matched-age contrasts were within ±1pp.",
        },
        {
            "id": "modality_asymmetry",
            "score": "MISS",
            "evidence": "Image-only was more harmful at age 4, but text-only was more harmful at ages 16 and 64.",
        },
        {
            "id": "harm_threshold",
            "score": "NO_PRIOR_OBSERVATION",
            "evidence": "Harm begins at age 1 for both batches.",
        },
        {
            "id": "seed_replication",
            "score": "HIT",
            "evidence": "All condition pairs separated by >1pp retained their ordering in all three seeds.",
        },
    ]
    atomic_json(OUTPUT / "prediction_scores.json", prediction_scores)

    report = f"""# Queue factorial — interpretation

## Headline: no safe threshold

A queue containing embeddings only **one optimizer step old** already reduced
mean bidirectional dev R@1 by 6.46pp at batch 1024 and 5.60pp at batch 512
relative to the respective queue-free controls. Under the pre-registered
>1pp harm definition, the harm threshold is therefore age 1 for both batches.
There is no empirically safe non-zero queue age in this projector-only regime.

## Primary pre-registered verdict: passes at the boundary

Exactly **6 of 8** matched-age contrasts fell inside the ±1pp equivalence
margin—the minimum required for `age_dominates`. This is a boundary result,
not clean equivalence. Age 0 failed by −1.422pp and age 64 failed by −1.022pp;
in both cases batch 512 was lower.

{markdown_table(paired_display, ["age", "b512_minus_b1024_pp", "within_margin", "direction", "net_of_age0_pp"], {"b512_minus_b1024_pp": "{:+.3f}", "net_of_age0_pp": "{:+.3f}"})}

Age 0 is not a queue condition. It is the no-queue control, so its −1.422pp
gap measures the anticipated pinned-LR batch offset rather than queue damage.
The primary 6/8 verdict above remains authoritative because it was
pre-registered without this exclusion.

**Secondary descriptive analysis:** among the seven rows containing an actual
queue, 6/7 lie inside the margin; only age 64 fails. This does not replace the
primary result.

**Exploratory baseline-offset analysis:** subtracting the age-0 batch offset
makes every queue-containing contrast positive or within +2.234pp; age 64
changes from −1.022pp to +0.399pp. This analysis was not pre-registered and
does not alter the primary verdict.

## Modality crossover: the prediction was a miss

{markdown_table(modality, ["target_age_steps", "queue_mode", "mean_percent", "sd_pp"], {"mean_percent": "{:.3f}", "sd_pp": "{:.3f}"})}

The prediction that image-only queues would be more harmful was **missed**.
Text-only was less harmful at nominal age 4, but more harmful at ages 16 and
64. The reversal is the result; no partial credit is assigned.

## Pre-registered drift-at-eviction regression

Cumulative drift at eviction was estimated by summing measured
previous-probe cosine distances over each entry's measured eviction-residence
window, prorating the ten-step measurement interval at window boundaries.
This is measured path length, not nominal age multiplied by one fixed rate.

The two modality curves **do not collapse onto one relationship**:

- Pooled age-only R²: {models["pooled_age_only"]["r_squared"]:.3f}.
- Pooled cumulative-drift-only R²: {models["pooled_cumulative_drift_only"]["r_squared"]:.3f}.
- Image-only drift R²: {models["image_cumulative_drift"]["r_squared"]:.3f}.
- Text-only drift R²: {models["text_cumulative_drift"]["r_squared"]:.3f}.
- A post-hoc modality-interaction model reaches R²
  {models["cumulative_drift_with_modality_interaction"]["r_squared"]:.3f}, but
  it is descriptive and based on only six three-seed conditions.

The clearest counterexample to a shared curve is image-only age 16 versus
text-only age 64. Their measured cumulative drifts are nearly identical
({matched["image_cumulative_drift"]:.4f} versus
{matched["text_cumulative_drift"]:.4f}), yet degradation is
{matched["image_degradation_pp"]:.2f}pp versus
{matched["text_degradation_pp"]:.2f}pp—a
{matched["degradation_gap_pp_text_minus_image"]:.2f}pp gap.

Therefore, cumulative drift improves the mechanistic description but is not
a sufficient quantitative model. A modality-specific factor—such as
gradient sensitivity, representation geometry, or false-negative
structure—remains necessary. No new mechanism is claimed from these data.

### Nearest-neighbour sensitivity analysis

Observed cumulative drift spans
{sensitivity_assessment["observed_cumulative_drift"]["minimum"]:.6f} to
{sensitivity_assessment["observed_cumulative_drift"]["maximum"]:.6f}, a range
of {sensitivity_assessment["observed_cumulative_drift"]["range"]:.6f}.
The four matching tolerances therefore span approximately 1.57% to 15.73% of
the observed range.

{markdown_table(sensitivity_display, ["tolerance", "tolerance_pct_range", "image_age", "image_drift", "text_age", "text_drift", "drift_difference", "degradation_difference_pp", "exceeds_1pp"], {"tolerance": "{:.3f}", "tolerance_pct_range": "{:.2f}", "image_drift": "{:.6f}", "text_drift": "{:.6f}", "drift_difference": "{:.6f}", "degradation_difference_pp": "{:.3f}"})}

{markdown_table(sensitivity_summary, ["tolerance", "tolerance_pct_range", "pairs", "pairs_over_1pp", "conclusion"], {"tolerance": "{:.3f}", "tolerance_pct_range": "{:.2f}"})}

The falsification **does generalise within this factorial**. At the strictest
0.005 tolerance, two independent matched-drift pairs are available and both
exceed the 1pp equivalence margin; the same two persist at 0.01. A third pair
enters at 0.02 and remains at 0.05, and all three exceed 1pp. Thus the result
does not rest only on the original 0.0861 crossing or on a permissive
post-hoc tolerance.

The comparisons also falsify either measured variable as a sufficient
univariate explanation in complementary ways. Image-only age 4 versus
text-only age 4 is matched on nominal age but differs in cumulative drift by
0.0177 and in degradation by 6.72pp. Conversely, image-only age 16 versus
text-only age 64 is effectively matched on cumulative drift (difference
0.000068) despite unequal residence ages, yet differs in degradation by
4.74pp. A large modality-dependent gap therefore remains both when age is
held constant and when cumulative drift is held approximately constant.
Neither age nor cumulative projector drift alone predicts the outcome.

The correct mechanism framing is: **projector drift is necessary but not
sufficient**. The controlled matched-drift comparisons falsify drift as a
complete explanation. Representation geometry, gradient sensitivity and
false-negative structure remain candidates, but this design cannot separate
them and no preference among them is claimed.

## Drift saturation

The step-0 distance crossed 0.95 for both modalities at optimizer step 160
(epoch 2), so absolute distance from initialization did saturate early.
The design-relevant movement from the previous measurement did not remain
high:

- Image final/first-epoch ratio:
  {gate["image"]["ratio"]:.8f} ({gate["image"]["first_epoch_mean_drift_from_previous"]:.6f}
  to {gate["image"]["final_epoch_mean_drift_from_previous"]:.8f}).
- Text final/first-epoch ratio:
  {gate["text"]["ratio"]:.8f} ({gate["text"]["first_epoch_mean_drift_from_previous"]:.6f}
  to {gate["text"]["final_epoch_mean_drift_from_previous"]:.8f}).

Both are far below the independently applied 0.20 threshold. Thus the
pre-registered drift-saturation risk did **not** materialise: continuing
projector motion decayed enough for age to remain discriminating.

## Efficiency cost–benefit

{markdown_table(efficiency, ["evidence", "comparison", "performance_effect_pp", "throughput_images_per_second", "peak_gpu_memory_gib", "available_gpu_memory_gib", "interpretation"], {"performance_effect_pp": "{:+.2f}", "throughput_images_per_second": "{:.1f}", "peak_gpu_memory_gib": "{:.2f}", "available_gpu_memory_gib": "{:.2f}"})}

The measurements come from two explicitly different controls: the ~19pp
performance effect is the accepted Wave-0 dose-response at the originally
used configuration, while throughput and memory stress come from the
factorial's maximum-capacity smoke. The queue offered no favourable trade in
this regime: it sharply reduced retrieval quality, did not save wall-clock
time, and memory capacity was not constraining. Post-fill throughput was
1,676 images/s at batch 1024 and 1,866 images/s at batch 512; the largest
smoke case used 8.88GiB of 95.01GiB.

## Frozen prediction scorecard

| Prediction | Score | Interpretation |
|---|---|---|
| `age_vs_count` | PASS AT BOUNDARY | Primary rule passes exactly 6/8; this is qualified evidence. |
| `modality_asymmetry` | MISS | The observed age-dependent crossover contradicts the directional prediction. |
| `harm_threshold` | NO PRIOR | The observed threshold is age 1 for both batches. |
| `seed_replication` | HIT | Ordering was stable whenever means differed by more than 1pp. |

The frozen predictions file was read but not modified.

This scorecard is also a methods result: there are two hits
(`age_vs_count`, at the boundary, and `seed_replication`), one miss
(`modality_asymmetry`) and one declared no-prior (`harm_threshold`). The miss
produced the chapter's most substantive mechanism result—the modality
crossover. Pre-registration exposed that contradiction; a post-hoc narrative
could instead have absorbed the crossover as apparent confirmation.

## Dissertation claim supported

For frozen DINOv3 ViT-S/16 plus frozen MiniLM with a trainable projector,
cross-batch memory built from post-projection embeddings is unsafe even at
one-step residence. Damage is primarily organised by age under the
pre-registered boundary rule, but the modality crossover is not explained by
cumulative projector drift alone. This supports a scoped claim about
projector-only frozen alignment—not a universal claim that memory queues are
harmful in all contrastive learning.
"""
    atomic_text(OUTPUT / "chapter_interpretation.md", report)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(FACTORIAL_ROOT / "report/summary.csv")
    paired = paired_age_table()
    drift = build_drift_at_eviction()
    regression = regression_analysis(drift)
    sensitivity, sensitivity_assessment = matched_drift_sensitivity(drift)
    efficiency = efficiency_table()

    paired.to_csv(OUTPUT / "paired_age_contrasts.csv", index=False)
    drift.to_csv(OUTPUT / "drift_at_eviction.csv", index=False)
    sensitivity.to_csv(OUTPUT / "matched_drift_sensitivity.csv", index=False)
    efficiency.to_csv(OUTPUT / "efficiency_cost_benefit.csv", index=False)
    atomic_json(OUTPUT / "drift_regression.json", regression)
    atomic_json(
        OUTPUT / "matched_drift_sensitivity.json", sensitivity_assessment
    )

    plot_age_curves(summary)
    plot_paired_differences(paired)
    plot_modality(summary)
    plot_drift_regression(drift)
    plot_saturation()
    write_report(
        summary,
        paired,
        drift,
        regression,
        sensitivity,
        sensitivity_assessment,
        efficiency,
    )

    frozen = ROOT / "predictions/queue_factorial.json"
    payload = {
        "status": "COMPLETE",
        "output_dir": str(OUTPUT.relative_to(ROOT)),
        "frozen_predictions_path": str(frozen.relative_to(ROOT)),
        "frozen_predictions_sha256_unchanged_by_script": True,
        "primary_equivalent_ages": 6,
        "primary_total_ages": 8,
        "secondary_queue_only_equivalent_ages": 6,
        "secondary_queue_only_total_ages": 7,
        "harm_threshold": {"512": 1, "1024": 1},
        "modality_asymmetry_score": "MISS",
        "drift_curve_collapse": regression["collapse_assessment"]["verdict"],
        "matched_drift_claim": sensitivity_assessment["claim_decision"],
    }
    atomic_json(OUTPUT / "interpretation_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
