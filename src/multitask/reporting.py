from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


HEADLINE_METRICS = (
    ("coco_retrieval", "coco5_i2t_R@1", "COCO i2t R@1"),
    ("coco_retrieval", "coco5_t2i_R@1", "COCO t2i R@1"),
    ("cifar100_zeroshot", "top1_accuracy", "CIFAR-100 top-1 accuracy"),
    ("pets_zeroshot", "top1_accuracy", "Pets top-1 accuracy"),
    ("eurosat_zeroshot", "top1_accuracy", "EuroSAT top-1 accuracy"),
)

MATCHED_BLF_COMPARISONS = (
    {
        "comparison": "dinov2_minilm_local",
        "baseline_config_id": "dinov2_vits14__all_minilm_l6_v2__baseline",
        "blf_config_id": "dinov2_vits14__all_minilm_l6_v2__local",
    },
    {
        "comparison": "convnextv2_minilm_local_global",
        "baseline_config_id": "convnextv2_tiny__all_minilm_l6_v2__baseline",
        "blf_config_id": "convnextv2_tiny__all_minilm_l6_v2__local_global",
    },
    {
        "comparison": "dinov2_bge_local_exploratory",
        "baseline_config_id": "dinov2_vits14__bge_small_en__baseline",
        "blf_config_id": "dinov2_vits14__bge_small_en__local",
    },
)


def _full_results(results: pd.DataFrame) -> pd.DataFrame:
    """Prefer full-mode rows and validate the columns used by reporting."""
    required = {"config_id", "task", "metric", "value"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Multi-task results are missing columns: {sorted(missing)}")
    selected = results.copy()
    if "mode" in selected.columns and selected["mode"].eq("full").any():
        selected = selected[selected["mode"].eq("full")].copy()
    return selected


def _matched_blf_rows(
    results: pd.DataFrame,
    comparisons: Sequence[dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    selected = _full_results(results)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    def metric_value(config_id: str, task: str, metric: str) -> float | None:
        matches = selected[
            selected["config_id"].eq(config_id)
            & selected["task"].eq(task)
            & selected["metric"].eq(metric)
        ]
        if matches.empty:
            return None
        if len(matches) != 1:
            raise ValueError(
                "Expected one full result for "
                f"{config_id}/{task}/{metric}, found {len(matches)}"
            )
        return float(matches.iloc[0]["value"])

    for comparison in comparisons:
        baseline_id = comparison["baseline_config_id"]
        blf_id = comparison["blf_config_id"]
        for task, metric, metric_label in HEADLINE_METRICS:
            baseline_value = metric_value(baseline_id, task, metric)
            blf_value = metric_value(blf_id, task, metric)
            if baseline_value is None or blf_value is None:
                if baseline_value is None:
                    missing.append(
                        {
                            "comparison": comparison["comparison"],
                            "config_id": baseline_id,
                            "task": task,
                            "metric": metric,
                        }
                    )
                if blf_value is None:
                    missing.append(
                        {
                            "comparison": comparison["comparison"],
                            "config_id": blf_id,
                            "task": task,
                            "metric": metric,
                        }
                    )
                continue
            delta = blf_value - baseline_value
            outcome = "tie"
            if delta > 1e-12:
                outcome = "blf_win"
            elif delta < -1e-12:
                outcome = "blf_loss"
            rows.append(
                {
                    "comparison": comparison["comparison"],
                    "baseline_config_id": baseline_id,
                    "blf_config_id": blf_id,
                    "task": task,
                    "metric": metric,
                    "metric_label": metric_label,
                    "baseline_value": baseline_value,
                    "blf_value": blf_value,
                    "delta": delta,
                    "delta_percentage_points": 100.0 * delta,
                    "outcome": outcome,
                }
            )
    columns = [
        "comparison",
        "baseline_config_id",
        "blf_config_id",
        "task",
        "metric",
        "metric_label",
        "baseline_value",
        "blf_value",
        "delta",
        "delta_percentage_points",
        "outcome",
    ]
    return pd.DataFrame(rows, columns=columns), missing


def build_matched_blf_comparison(
    results: pd.DataFrame,
    comparisons: Sequence[dict[str, str]] = MATCHED_BLF_COMPARISONS,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Build the five-metric matched comparison used after Steps 11 and 11.5.

    The function intentionally reports task-specific deltas and win/tie/loss
    counts rather than averaging retrieval recall with classification accuracy.
    """
    comparison, missing = _matched_blf_rows(results, comparisons)
    comparison.attrs["missing"] = missing
    if require_complete and missing:
        preview = ", ".join(
            f"{row['config_id']}/{row['task']}/{row['metric']}" for row in missing[:5]
        )
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(
            "Matched BLF comparison is incomplete; finish Step 11.5 first. "
            f"Missing: {preview}{suffix}"
        )
    return comparison


def summarize_matched_blf_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Count per-candidate wins, ties, and losses without raw metric averaging."""
    columns = ["comparison", "baseline_config_id", "blf_config_id", "metrics_compared", "wins", "ties", "losses"]
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in comparison.groupby(
        ["comparison", "baseline_config_id", "blf_config_id"], sort=False
    ):
        comparison_id, baseline_id, blf_id = keys
        rows.append(
            {
                "comparison": comparison_id,
                "baseline_config_id": baseline_id,
                "blf_config_id": blf_id,
                "metrics_compared": int(len(group)),
                "wins": int(group["outcome"].eq("blf_win").sum()),
                "ties": int(group["outcome"].eq("tie").sum()),
                "losses": int(group["outcome"].eq("blf_loss").sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_multitask_summary(output_root: str | Path) -> tuple[Path, Path]:
    root = Path(output_root)
    results_path = root / "task_results_long.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Run Step 11 first: {results_path}")
    results = _full_results(pd.read_csv(results_path))
    status = pd.read_csv(root / "task_status.csv")
    headline_names = {metric for _, metric, _ in HEADLINE_METRICS}
    headline = results[results["metric"].isin(headline_names)]
    best = headline.loc[headline.groupby(["task", "metric"])["value"].idxmax()] if not headline.empty else headline
    comparison = build_matched_blf_comparison(results, require_complete=False)
    comparison_missing = comparison.attrs.get("missing", [])
    comparison_summary = summarize_matched_blf_comparison(comparison)
    comparison_complete = not comparison_missing and len(comparison) == len(MATCHED_BLF_COMPARISONS) * len(HEADLINE_METRICS)
    comparison_path = root / "blf_vs_baseline_multitask.csv"
    if comparison_complete:
        comparison.to_csv(comparison_path, index=False)
    payload = {
        "best_per_task": best.to_dict("records"),
        "matched_blf_comparison_complete": comparison_complete,
        "matched_blf_comparison": comparison.to_dict("records") if comparison_complete else [],
        "matched_blf_summary": comparison_summary.to_dict("records") if comparison_complete else [],
        "matched_blf_missing": comparison_missing,
        "matched_blf_output": str(comparison_path) if comparison_complete else None,
        "missing_tasks": status.loc[status.status != "ready", "task"].tolist(),
        "limitations": [
            "Tasks use different official metrics and are not raw-averaged.",
            "CIFAR-100 and Oxford-IIIT Pet use test splits; torchvision exposes EuroSAT only as the complete dataset.",
            "Zero-shot classes use a fixed three-template prompt ensemble with no task-specific fitting.",
            "Matched BLF changes are point estimates unless paired uncertainty is computed from sample-level predictions.",
            "Oracle and specialist claims require Phase 1.5 sample-level analysis.",
        ],
    }
    json_path = root / "multitask_summary.json"
    md_path = root / "multitask_summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Phase 1 multi-task findings",
        "",
        "Results use best checkpoints and frozen pretrained backbones.",
        "",
        "## Best configuration per task",
        "",
        best.to_markdown(index=False) if not best.empty else "No completed results.",
        "",
        "## Matched BLF versus baseline",
        "",
    ]
    if comparison_complete:
        display_columns = [
            "comparison",
            "metric_label",
            "baseline_value",
            "blf_value",
            "delta_percentage_points",
            "outcome",
        ]
        lines.extend(
            [
                comparison[display_columns].to_markdown(index=False),
                "",
                "### Win/tie/loss summary",
                "",
                comparison_summary.to_markdown(index=False),
                "",
                f"Detailed output: `{comparison_path}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Pending Step 11.5: {len(comparison_missing)} required full-result values are missing.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            *[f"- {value}" for value in payload["limitations"]],
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path
