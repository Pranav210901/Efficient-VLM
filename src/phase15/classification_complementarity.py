"""Pairwise and per-class classification error complementarity analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import atomic_csv, atomic_json
from .evidence import resolve_evaluation_root


REQUIRED = (
    "sample_id",
    "target_index",
    "target",
    "prediction_index",
    "correct",
    "confidence",
    "classification_margin",
)


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return 0.0
    value = left.corr(right, method=method)
    return 0.0 if pd.isna(value) else float(value)


def classification_pair_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    missing = set(REQUIRED) - set(left.columns) | (set(REQUIRED) - set(right.columns))
    if missing:
        raise ValueError(f"Classification predictions are missing columns: {sorted(missing)}")
    if left["sample_id"].duplicated().any() or right["sample_id"].duplicated().any():
        raise ValueError("Classification sample IDs must be unique")
    merged = left[list(REQUIRED)].merge(right[list(REQUIRED)], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Classification prediction files do not cover identical sample IDs")
    if not merged["target_index_a"].eq(merged["target_index_b"]).all():
        raise ValueError("Classification targets differ between configurations")
    a = merged["correct_a"].astype(bool)
    b = merged["correct_b"].astype(bool)
    both = a & b
    only_a = a & ~b
    only_b = ~a & b
    neither = ~a & ~b
    agreement = merged["prediction_index_a"].eq(merged["prediction_index_b"])
    union = a | b
    count = len(merged)
    return {
        "sample_count": count,
        "agreement_count": int(agreement.sum()),
        "agreement_rate": float(agreement.mean()),
        "disagreement_count": int((~agreement).sum()),
        "disagreement_rate": float((~agreement).mean()),
        "both_correct": int(both.sum()),
        "only_a_correct": int(only_a.sum()),
        "only_b_correct": int(only_b.sum()),
        "both_incorrect": int(neither.sum()),
        "both_correct_fraction": float(both.mean()),
        "only_a_fraction": float(only_a.mean()),
        "only_b_fraction": float(only_b.mean()),
        "both_incorrect_fraction": float(neither.mean()),
        "correct_set_jaccard": float(both.sum() / max(1, union.sum())),
        "prediction_label_agreement": float(agreement.mean()),
        "confidence_pearson": _safe_correlation(merged["confidence_a"], merged["confidence_b"], "pearson"),
        "confidence_spearman": _safe_correlation(merged["confidence_a"], merged["confidence_b"], "spearman"),
        "margin_pearson": _safe_correlation(merged["classification_margin_a"], merged["classification_margin_b"], "pearson"),
        "margin_spearman": _safe_correlation(merged["classification_margin_a"], merged["classification_margin_b"], "spearman"),
        "unique_correct_a": int(only_a.sum()),
        "unique_correct_b": int(only_b.sum()),
    }


def classification_per_class_metrics(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left[list(REQUIRED)].merge(right[list(REQUIRED)], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Classification prediction files do not cover identical sample IDs")
    rows: list[dict[str, Any]] = []
    for class_id, group in merged.groupby("target_index_a", sort=True):
        a = group["correct_a"].astype(bool)
        b = group["correct_b"].astype(bool)
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(group["target_a"].iloc[0]),
                "sample_count": len(group),
                "a_accuracy": float(a.mean()),
                "b_accuracy": float(b.mean()),
                "both_correct": int((a & b).sum()),
                "only_a_correct": int((a & ~b).sum()),
                "only_b_correct": int((~a & b).sum()),
                "both_incorrect": int((~a & ~b).sum()),
            }
        )
    return pd.DataFrame(rows)


def load_classification_predictions(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append({column: value.get(column) for column in REQUIRED})
    frame = pd.DataFrame(rows)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    return frame


def _plot_matrix(frame: pd.DataFrame, names: list[str], metric: str, dataset: str, output: Path, directional: bool = False) -> None:
    matrix = pd.DataFrame(np.eye(len(names)) if not directional else np.zeros((len(names), len(names)), dtype=float), index=names, columns=names)
    for row in frame.itertuples():
        matrix.loc[row.configuration_a, row.configuration_b] = float(getattr(row, metric))
        reverse = "only_b_fraction" if metric == "only_a_fraction" else metric
        matrix.loc[row.configuration_b, row.configuration_a] = float(getattr(row, reverse))
    figure, axis = plt.subplots(figsize=(max(7, len(names) * 0.42), max(6, len(names) * 0.40)))
    image = axis.imshow(matrix.values, vmin=-1 if "correlation" in metric else 0, vmax=1, cmap="coolwarm" if "correlation" in metric else "viridis")
    axis.set_xticks(range(len(names)), [str(index + 1) for index in range(len(names))], rotation=90)
    axis.set_yticks(range(len(names)), [str(index + 1) for index in range(len(names))])
    axis.set_title(f"{dataset}: {metric} (expert indices)")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_classification_complementarity(project_root: str | Path, mode: str = "full", evaluation_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = resolve_evaluation_root(root, evaluation_root) / "evaluation_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if "mode" not in manifest:
        manifest["mode"] = manifest["cache"].map(lambda value: "full" if str(value).endswith("__full.pt") else "smoke")
    selected = manifest[
        manifest["status"].eq("complete")
        & manifest["mode"].eq(mode)
        & manifest["task"].isin(["cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"])
    ].copy()
    if selected.empty:
        raise FileNotFoundError("No full classification predictions are listed in the evaluation manifest")
    output = root / "results/phase15/complementarity"
    pair_rows: list[dict[str, Any]] = []
    class_parts: list[pd.DataFrame] = []
    unique_rows: list[dict[str, Any]] = []
    heatmaps: list[str] = []
    for dataset, group in selected.groupby("task", sort=True):
        tables = {}
        for row in group.sort_values("config_id").itertuples():
            path = Path(str(row.predictions))
            if not path.is_absolute():
                path = root / path
            tables[str(row.config_id)] = load_classification_predictions(path)
        names = sorted(tables)
        dataset_rows: list[dict[str, Any]] = []
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                metrics = classification_pair_metrics(tables[left], tables[right])
                row = {"dataset": dataset, "configuration_a": left, "configuration_b": right, **metrics}
                dataset_rows.append(row)
                pair_rows.append(row)
                per_class = classification_per_class_metrics(tables[left], tables[right])
                per_class.insert(0, "dataset", dataset)
                per_class.insert(3, "configuration_a", left)
                per_class.insert(4, "configuration_b", right)
                class_parts.append(per_class)
                unique_rows.extend(
                    [
                        {"dataset": dataset, "configuration": left, "opponent": right, "unique_correct": metrics["unique_correct_a"], "unique_fraction": metrics["only_a_fraction"]},
                        {"dataset": dataset, "configuration": right, "opponent": left, "unique_correct": metrics["unique_correct_b"], "unique_fraction": metrics["only_b_fraction"]},
                    ]
                )
        dataset_frame = pd.DataFrame(dataset_rows)
        for metric, directional in (
            ("correct_set_jaccard", False),
            ("only_a_fraction", True),
            ("prediction_label_agreement", False),
            ("margin_spearman", False),
        ):
            path = output / "heatmaps" / f"{dataset}__{metric}.png"
            _plot_matrix(dataset_frame, names, metric, dataset, path, directional)
            heatmaps.append(str(path))
    pairwise = pd.DataFrame(pair_rows)
    per_class = pd.concat(class_parts, ignore_index=True) if class_parts else pd.DataFrame()
    unique = pd.DataFrame(unique_rows)
    paths = {
        "pairwise": output / "classification_pairwise.csv",
        "per_class": output / "classification_per_class.csv",
        "unique_wins": output / "classification_unique_wins.csv",
    }
    atomic_csv(pairwise, paths["pairwise"])
    atomic_csv(per_class, paths["per_class"])
    atomic_csv(unique, paths["unique_wins"])
    summary = {
        "datasets": sorted(selected["task"].unique()),
        "configurations": int(selected["config_id"].nunique()),
        "pair_rows": len(pairwise),
        "per_class_rows": len(per_class),
        "diagnostic_only": True,
        "outputs": {key: str(value) for key, value in paths.items()},
        "heatmaps": heatmaps,
    }
    atomic_json(summary, output / "classification_complementarity_summary.json")
    return summary
