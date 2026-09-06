"""Cross-task ranks, correlations, specialisation, and cost-adjusted scores."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import atomic_csv, atomic_json
from .evidence import resolve_evaluation_root


TASK_METRICS = {
    ("coco_retrieval", "coco5_i2t_R@1"): "coco_i2t",
    ("coco_retrieval", "coco5_t2i_R@1"): "coco_t2i",
    ("cifar100_zeroshot", "top1_accuracy"): "cifar100",
    ("pets_zeroshot", "top1_accuracy"): "pets",
    ("eurosat_zeroshot", "top1_accuracy"): "eurosat",
}
NORMALISATION_METHODS = ("relative_to_task_best", "min_max_within_task", "z_score_within_task")


def normalise_task_scores(values: pd.Series, method: str = "relative_to_task_best") -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if method == "relative_to_task_best":
        best = numeric.max()
        return numeric / best if pd.notna(best) and best != 0 else pd.Series(0.0, index=values.index)
    if method == "min_max_within_task":
        low, high = numeric.min(), numeric.max()
        return (numeric - low) / (high - low) if pd.notna(high) and high > low else pd.Series(1.0, index=values.index)
    if method == "z_score_within_task":
        std = numeric.std(ddof=0)
        return (numeric - numeric.mean()) / std if pd.notna(std) and std > 0 else pd.Series(0.0, index=values.index)
    raise ValueError(f"Unknown task normalisation method: {method}")


def task_specialisation_score(normalised_scores: Iterable[float]) -> float:
    values = np.asarray(list(normalised_scores), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Specialisation requires finite task-normalised scores")
    return float(values.std(ddof=0))


def cost_adjusted_score(normalised_score: float, latency_ms: float, reference_latency_ms: float) -> float:
    if latency_ms <= 0 or reference_latency_ms <= 0:
        raise ValueError("latencies must be positive")
    return float(normalised_score * np.sqrt(reference_latency_ms / latency_ms))


def _kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for i, j in itertools.combinations(range(len(left)), 2):
        dx = np.sign(left[i] - left[j])
        dy = np.sign(right[i] - right[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_left += 1
        elif dy == 0:
            ties_right += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denominator = np.sqrt((concordant + discordant + ties_left) * (concordant + discordant + ties_right))
    return float((concordant - discordant) / denominator) if denominator else 0.0


def task_rank_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_a, task_b in itertools.combinations(matrix.columns, 2):
        paired = matrix[[task_a, task_b]].dropna()
        spearman = paired[task_a].rank(method="average").corr(paired[task_b].rank(method="average")) if len(paired) >= 2 else np.nan
        rows.append(
            {
                "task_a": task_a,
                "task_b": task_b,
                "configuration_count": len(paired),
                "spearman": float(spearman) if pd.notna(spearman) else 0.0,
                "kendall": _kendall_tau(paired[task_a].to_numpy(), paired[task_b].to_numpy()) if len(paired) >= 2 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _matrix_plot(matrix: pd.DataFrame, title: str, path: Path, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(max(7, matrix.shape[1] * 1.3), max(6, matrix.shape[0] * 0.35)))
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(matrix.shape[0]), [str(index + 1) for index in range(matrix.shape[0])])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _anchor_rankings(long: pd.DataFrame, kind: str, anchor: str) -> pd.DataFrame:
    if kind == "vision":
        selected = long[long["text_encoder"].eq(anchor) & long["variant"].eq("baseline")].copy()
        identity = "vision_encoder"
    else:
        selected = long[long["vision_encoder"].eq(anchor) & long["variant"].eq("baseline")].copy()
        identity = "text_encoder"
    selected["rank"] = selected.groupby("task_id")["raw_value"].rank(method="min", ascending=False)
    selected["anchor"] = anchor
    return selected[[identity, "task_id", "raw_value", "rank", "anchor"]].sort_values(["task_id", "rank", identity], kind="stable")


def run_cross_task_analysis(
    project_root: str | Path,
    normalisation: str = "relative_to_task_best",
    near_best_threshold: float = 0.02,
    vision_anchor_text: str = "all_minilm_l6_v2",
    text_anchor_vision: str = "dinov2_vits14",
    evaluation_root: str | Path | None = None,
) -> dict[str, Any]:
    if normalisation not in NORMALISATION_METHODS:
        raise ValueError(f"normalisation must be one of {NORMALISATION_METHODS}")
    if not 0 <= near_best_threshold < 1:
        raise ValueError("near_best_threshold must be in [0, 1)")
    root = Path(project_root).resolve()
    results = pd.read_csv(resolve_evaluation_root(root, evaluation_root) / "task_results_long.csv")
    if "mode" in results:
        results = results[results["mode"].eq("full")]
    selected_parts = []
    for (task, metric), task_id in TASK_METRICS.items():
        part = results[results["task"].eq(task) & results["metric"].eq(metric)].copy()
        part["task_id"] = task_id
        part["raw_value"] = pd.to_numeric(part["value"], errors="raise")
        selected_parts.append(part)
    long = pd.concat(selected_parts, ignore_index=True)
    if long.empty:
        raise ValueError("No cross-task headline metrics are available")
    if long.duplicated(["config_id", "task_id"]).any():
        raise ValueError("Cross-task input contains duplicate configuration/task metrics")
    for method in NORMALISATION_METHODS:
        long[method] = long.groupby("task_id")["raw_value"].transform(lambda value: normalise_task_scores(value, method))
    long["normalised_score"] = long[normalisation]
    long["rank"] = long.groupby("task_id")["raw_value"].rank(method="min", ascending=False)
    task_best = long.groupby("task_id")["raw_value"].transform("max")
    long["absolute_gap_to_best"] = task_best - long["raw_value"]
    long["relative_gap_to_best"] = 1.0 - long["relative_to_task_best"]
    identity = ["config_id", "vision_encoder", "text_encoder", "variant"]
    raw = long.pivot(index=identity, columns="task_id", values="raw_value").reset_index()
    normalised = long.pivot(index=identity, columns="task_id", values="normalised_score").reset_index()
    rankings = long[identity + ["task_id", "raw_value", "normalised_score", "rank", "absolute_gap_to_best", "relative_gap_to_best"]].copy()
    vision_rankings = _anchor_rankings(long, "vision", vision_anchor_text)
    text_rankings = _anchor_rankings(long, "text", text_anchor_vision)
    raw_task_matrix = raw.set_index("config_id")[[value for value in TASK_METRICS.values() if value in raw]].copy()
    correlations = task_rank_correlations(raw_task_matrix)
    efficiency_path = root / "results/phase15/efficiency/full_pairs.csv"
    efficiency = pd.read_csv(efficiency_path) if efficiency_path.exists() else pd.DataFrame(columns=["config_id", "latency_ms"])
    summary_rows: list[dict[str, Any]] = []
    minimum_latency = float(efficiency["latency_ms"].min()) if not efficiency.empty else 1.0
    for config_id, group in long.groupby("config_id", sort=True):
        ordered = group.sort_values("normalised_score", ascending=False)
        latency_row = efficiency[efficiency["config_id"].eq(config_id)]
        latency = float(latency_row["latency_ms"].iloc[0]) if not latency_row.empty else np.nan
        mean_score = float(group["normalised_score"].mean())
        summary_rows.append(
            {
                "config_id": config_id,
                "vision_encoder": group["vision_encoder"].iloc[0],
                "text_encoder": group["text_encoder"].iloc[0],
                "variant": group["variant"].iloc[0],
                "tasks_evaluated": int(group["task_id"].nunique()),
                "tasks_won": int(group["rank"].eq(1).sum()),
                "tasks_within_threshold": int(group["relative_gap_to_best"].le(near_best_threshold).sum()),
                "mean_normalised_multitask_score": mean_score,
                "std_normalised_task_performance": float(group["normalised_score"].std(ddof=0)),
                "task_specialisation_index": task_specialisation_score(group["normalised_score"]),
                "latency_ms": latency,
                "cost_adjusted_task_score": cost_adjusted_score(mean_score, latency, minimum_latency) if pd.notna(latency) else np.nan,
                "strongest_task": ordered["task_id"].iloc[0],
                "strongest_task_score": float(ordered["raw_value"].iloc[0]),
                "strongest_task_relative_gap": float(ordered["relative_gap_to_best"].iloc[0]),
                "weakest_task": ordered["task_id"].iloc[-1],
                "weakest_task_score": float(ordered["raw_value"].iloc[-1]),
                "weakest_task_relative_gap": float(ordered["relative_gap_to_best"].iloc[-1]),
                "task_specific_strengths": json.dumps(ordered.head(2)["task_id"].tolist()),
                "task_specific_weaknesses": json.dumps(ordered.tail(2)["task_id"].tolist()),
                "normalisation": normalisation,
            }
        )
    specialisation = pd.DataFrame(summary_rows).sort_values(["mean_normalised_multitask_score", "config_id"], ascending=[False, True], kind="stable")
    output = root / "results/phase15/cross_task"
    paths = {
        "raw": output / "task_performance_matrix_raw.csv",
        "normalised": output / "task_performance_matrix_normalised.csv",
        "configuration_rankings": output / "configuration_rankings.csv",
        "vision_rankings": output / "vision_encoder_rankings.csv",
        "text_rankings": output / "text_encoder_rankings.csv",
        "correlations": output / "task_rank_correlations.csv",
        "specialisation": output / "expert_specialisation.csv",
    }
    for key, frame in (
        ("raw", raw),
        ("normalised", normalised),
        ("configuration_rankings", rankings),
        ("vision_rankings", vision_rankings),
        ("text_rankings", text_rankings),
        ("correlations", correlations),
        ("specialisation", specialisation),
    ):
        atomic_csv(frame, paths[key])
    task_columns = [value for value in TASK_METRICS.values() if value in raw]
    _matrix_plot(raw.set_index("config_id")[task_columns], "Raw task performance (expert indices)", output / "task_performance_raw_heatmap.png")
    _matrix_plot(normalised.set_index("config_id")[task_columns], f"Task performance: {normalisation}", output / "task_performance_normalised_heatmap.png", vmin=0 if normalisation != "z_score_within_task" else None, vmax=1 if normalisation != "z_score_within_task" else None)
    rank_matrix = rankings.pivot(index="config_id", columns="task_id", values="rank")
    _matrix_plot(rank_matrix, "Rank by task (lower is better)", output / "rank_by_task_heatmap.png", cmap="viridis_r")
    correlation_matrix = raw_task_matrix.corr(method="spearman")
    _matrix_plot(correlation_matrix, "Task Spearman correlation", output / "task_correlation_heatmap.png", cmap="coolwarm", vmin=-1, vmax=1)
    if specialisation["latency_ms"].notna().any():
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(specialisation["latency_ms"], specialisation["mean_normalised_multitask_score"], alpha=0.8)
        axis.set_xlabel("Measured pair latency (ms)")
        axis.set_ylabel(f"Mean {normalisation}")
        axis.set_title("Cross-task performance versus latency")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "performance_vs_latency.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    summary = {
        "normalisation": normalisation,
        "raw_metrics_are_not_averaged": True,
        "tasks": task_columns,
        "configurations": int(long["config_id"].nunique()),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    atomic_json(summary, output / "cross_task_summary.json")
    return summary
