"""Evidence-weighted, diversity-constrained expert selection for Phase 2 inputs.

This module selects frozen pretrained encoders and aligned pair checkpoints. It
does not implement a router, cross-attention, adaptive fusion, or Phase 2.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yaml

from src.common.configuration_ids import parse_configuration_id

from .evaluation_protocol import load_evaluation_protocol
from .evidence import resolve_evaluation_root
from .io_utils import atomic_csv, atomic_json, atomic_text, dataframe_records, stat_fingerprint
from .seed_reliability import RELIABILITY_SCORES, run_seed_reliability_mapping


DEFAULT_SELECTION_WEIGHTS = {
    "multitask_performance": 0.25,
    "unique_wins": 0.17,
    "oracle_contribution": 0.13,
    "cross_task_specialisation": 0.12,
    "latency_efficiency": 0.10,
    "memory_efficiency": 0.06,
    "seed_reliability": 0.07,
    "architectural_diversity": 0.05,
    "token_interface_readiness": 0.05,
}
DEFAULT_DIVERSITY_CONSTRAINTS = {
    "primary_vision_experts": 3,
    "primary_text_experts": 3,
    "cheap_fallback_vision_experts": 1,
    "cheap_fallback_text_experts": 1,
    "primary_pair_paths": 4,
    "maximum_pair_paths": 6,
    "maximum_paths_per_vision_encoder": 2,
    "maximum_paths_per_text_encoder": 2,
    "minimum_vision_architecture_families": 2,
    "minimum_distinct_text_encoders": 2,
    "require_low_cost_pair": True,
    "require_unique_error_pair": True,
    "prohibit_inconclusive_blf_duplicate": True,
}

VISION_FAMILIES = {
    "efficientnet_b0": "efficient_cnn",
    "convnext_tiny": "convnext_cnn",
    "convnextv2_tiny": "convnext_cnn",
    "dinov2_vits14": "plain_vit",
    "swin_tiny": "hierarchical_vit",
}
TEXT_FAMILIES = {
    "all_minilm_l6_v2": "minilm_sentence_transformer",
    "bge_small_en": "bge_embedding_model",
    "e5_small_v2": "e5_embedding_model",
    "distilbert": "masked_language_model",
}
TOKEN_READY_VISION = set(VISION_FAMILIES)
TOKEN_READY_TEXT = set(TEXT_FAMILIES)
HEADLINE_METRICS = {
    "coco_retrieval": ("coco5_i2t_R@1", "coco5_t2i_R@1"),
    "cifar100_zeroshot": ("top1_accuracy",),
    "pets_zeroshot": ("top1_accuracy",),
    "eurosat_zeroshot": ("top1_accuracy",),
}


class SelectedExpert(TypedDict):
    canonical_name: str
    role: str
    architecture_family: str
    checkpoint_requirement: str
    token_interface_expected: bool
    latency_ms: float
    peak_memory_mb: float
    parameter_count: int
    reliability_status: str
    reasons: list[str]


class SelectedPair(TypedDict):
    pair_id: str
    vision_encoder: str
    text_encoder: str
    variant: str
    checkpoint_path: str
    checkpoint_fingerprint: str
    selection_score: float
    unique_win_contribution: float
    oracle_contribution: float
    measured_pair_latency_ms: float
    reliability_status: str
    intended_role: str
    reasons: list[str]


def _score_higher(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.5, index=values.index)
    fill = numeric.fillna(numeric.min())
    return pd.Series(1.0, index=values.index) if fill.nunique() <= 1 else fill.rank(method="average", pct=True)


def _score_lower(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.5, index=values.index)
    fill = numeric.fillna(numeric.max())
    return pd.Series(1.0, index=values.index) if fill.nunique() <= 1 else fill.rank(method="average", pct=True, ascending=False)


def rank_experts(frame: pd.DataFrame, id_column: str, limit: int = 3) -> pd.DataFrame:
    """Backwards-compatible deterministic ranking helper used by unit tests."""
    required = {id_column, "performance_rank", "unique_wins", "latency_ms", "memory_mb"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing expert-selection columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["performance_score"] = _score_lower(work["performance_rank"])
    work["unique_score"] = _score_higher(work["unique_wins"])
    work["latency_score"] = _score_lower(work["latency_ms"])
    work["memory_score"] = _score_lower(work["memory_mb"])
    work["selection_score"] = work[["performance_score", "unique_score", "latency_score", "memory_score"]].mean(axis=1)
    return work.sort_values(["selection_score", id_column], ascending=[False, True], kind="stable").head(limit).reset_index(drop=True)


def _performance_features(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[
        results.apply(lambda row: row["metric"] in HEADLINE_METRICS.get(row["task"], ()), axis=1)
    ].copy()
    if selected.empty:
        raise ValueError("No headline multi-task metrics are available")
    selected["metric_key"] = selected["task"].astype(str) + "__" + selected["metric"].astype(str)
    selected["within_task_score"] = selected.groupby("metric_key")["value"].transform(_score_higher)
    identity = selected.groupby("config_id", as_index=False).agg(
        vision_encoder=("vision_encoder", "first"),
        text_encoder=("text_encoder", "first"),
        variant=("variant", "first"),
        multitask_performance_score=("within_task_score", "mean"),
        headline_metrics=("metric_key", "nunique"),
    )
    raw = selected.pivot_table(index="config_id", columns="metric_key", values="value", aggfunc="first")
    raw.columns = [f"metric__{column}" for column in raw.columns]
    return identity.merge(raw.reset_index(), on="config_id", how="left")


def _retrieval_features(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pairwise.itertuples():
        total = int(row.both_successful + row.only_a_successful + row.only_b_successful + row.both_unsuccessful)
        rows.extend(
            [
                {"config_id": row.expert_a, "retrieval_unique_wins": row.only_a_successful / max(1, total), "retrieval_error_diversity": 1.0 - float(row.jaccard)},
                {"config_id": row.expert_b, "retrieval_unique_wins": row.only_b_successful / max(1, total), "retrieval_error_diversity": 1.0 - float(row.jaccard)},
            ]
        )
    return pd.DataFrame(rows).groupby("config_id", as_index=False).mean(numeric_only=True) if rows else pd.DataFrame(columns=["config_id", "retrieval_unique_wins", "retrieval_error_diversity"])


def _classification_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["config_id", "classification_unique_wins", "classification_error_diversity"])
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples():
        rows.extend(
            [
                {"config_id": row.configuration_a, "classification_unique_wins": row.only_a_fraction, "classification_error_diversity": 1.0 - row.correct_set_jaccard},
                {"config_id": row.configuration_b, "classification_unique_wins": row.only_b_fraction, "classification_error_diversity": 1.0 - row.correct_set_jaccard},
            ]
        )
    return pd.DataFrame(rows).groupby("config_id", as_index=False).mean(numeric_only=True)


def _oracle_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["config_id", "oracle_contribution"])
    frame = pd.read_csv(path)
    if "subset_scope" in frame:
        full = frame[frame["subset_scope"].eq("full_pool")]
        if not full.empty:
            frame = full
    return frame.groupby("configuration", as_index=False)["marginal_oracle_loss"].mean().rename(
        columns={"configuration": "config_id", "marginal_oracle_loss": "oracle_contribution"}
    )


def _specialisation_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["config_id", "cross_task_specialisation", "cost_adjusted_task_score"])
    frame = pd.read_csv(path)
    return frame[["config_id", "task_specialisation_index", "cost_adjusted_task_score"]].rename(
        columns={"task_specialisation_index": "cross_task_specialisation"}
    )


def _seed_features(seed_results_path: Path) -> pd.DataFrame:
    """Compatibility wrapper that now maps every canonical baseline and BLF row."""
    if not seed_results_path.exists():
        return pd.DataFrame(columns=["config_id", "seed_runs", "standard_deviation", "seed_reliability_score", "reliability_status"])
    root = seed_results_path.parent
    paired = root / "results/seed_sweep_paired_deltas.csv" if (root / "results").exists() else seed_results_path.with_name("seed_sweep_paired_deltas.csv")
    from .seed_reliability import build_seed_reliability

    return build_seed_reliability(
        pd.read_csv(seed_results_path),
        pd.read_csv(paired) if paired.exists() else None,
    )


def build_pair_scores(
    results: pd.DataFrame,
    pairwise: pd.DataFrame,
    efficiency: pd.DataFrame,
    seed_results_path: Path,
    *,
    classification_pairwise_path: Path | None = None,
    oracle_marginal_path: Path | None = None,
    specialisation_path: Path | None = None,
    selection_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = selection_weights or DEFAULT_SELECTION_WEIGHTS
    required_efficiency = {"config_id", "latency_ms", "peak_allocated_bytes"}
    if not required_efficiency.issubset(efficiency.columns):
        raise ValueError(f"Efficiency results are missing: {sorted(required_efficiency - set(efficiency.columns))}")
    work = _performance_features(results)
    work = work.merge(_retrieval_features(pairwise), on="config_id", how="left")
    work = work.merge(_classification_features(classification_pairwise_path or Path("/nonexistent")), on="config_id", how="left")
    work = work.merge(_oracle_features(oracle_marginal_path or Path("/nonexistent")), on="config_id", how="left")
    work = work.merge(_specialisation_features(specialisation_path or Path("/nonexistent")), on="config_id", how="left")
    seed_features = _seed_features(seed_results_path)
    if "variant" in seed_features:
        seed_features = seed_features.rename(columns={"variant": "seed_variant"})
    work = work.merge(seed_features, on="config_id", how="left")
    efficiency_columns = [
        column
        for column in ("config_id", "latency_ms", "peak_allocated_bytes", "parameters", "flops_per_batch", "device_name", "checkpoint")
        if column in efficiency
    ]
    work = work.merge(efficiency[efficiency_columns], on="config_id", how="left")
    if work[["latency_ms", "peak_allocated_bytes"]].isna().any().any():
        missing = work.loc[work["latency_ms"].isna(), "config_id"].tolist()
        raise ValueError(f"Step 15 efficiency rows are missing for: {missing}")
    for column in (
        "retrieval_unique_wins",
        "retrieval_error_diversity",
        "classification_unique_wins",
        "classification_error_diversity",
        "oracle_contribution",
        "cross_task_specialisation",
        "cost_adjusted_task_score",
    ):
        work[column] = pd.to_numeric(work.get(column, 0.0), errors="coerce").fillna(0.0)
    work["unique_win_contribution"] = 0.5 * work["retrieval_unique_wins"] + 0.5 * work["classification_unique_wins"]
    work["error_diversity"] = 0.5 * work["retrieval_error_diversity"] + 0.5 * work["classification_error_diversity"]
    work["reliability_status"] = work.get("reliability_status", pd.Series(index=work.index, dtype=object)).fillna("not_tested")
    work["seed_runs"] = pd.to_numeric(work.get("seed_runs", 0), errors="coerce").fillna(0).astype(int)
    work["seed_reliability_score"] = work["reliability_status"].map(RELIABILITY_SCORES).fillna(RELIABILITY_SCORES["not_tested"])
    work["vision_architecture_family"] = work["vision_encoder"].map(VISION_FAMILIES).fillna("other")
    work["text_architecture_family"] = work["text_encoder"].map(TEXT_FAMILIES).fillna("other")
    work["token_interface_readiness"] = work["vision_encoder"].isin(TOKEN_READY_VISION) & work["text_encoder"].isin(TOKEN_READY_TEXT)
    work["multitask_performance_component"] = _score_higher(work["multitask_performance_score"])
    work["unique_wins_component"] = _score_higher(work["unique_win_contribution"])
    work["oracle_contribution_component"] = _score_higher(work["oracle_contribution"])
    work["cross_task_specialisation_component"] = _score_higher(work["cross_task_specialisation"])
    latency = _score_lower(work["latency_ms"])
    parameters = _score_lower(work["parameters"]) if "parameters" in work else pd.Series(0.5, index=work.index)
    flops = _score_lower(work["flops_per_batch"]) if "flops_per_batch" in work else pd.Series(0.5, index=work.index)
    work["latency_efficiency_component"] = 0.5 * latency + 0.25 * parameters + 0.25 * flops
    work["memory_efficiency_component"] = _score_lower(work["peak_allocated_bytes"])
    family_counts = work["vision_architecture_family"].value_counts()
    work["architectural_diversity_component"] = _score_lower(work["vision_architecture_family"].map(family_counts))
    work["token_interface_readiness_component"] = work["token_interface_readiness"].astype(float)
    component_columns = {
        "multitask_performance": "multitask_performance_component",
        "unique_wins": "unique_wins_component",
        "oracle_contribution": "oracle_contribution_component",
        "cross_task_specialisation": "cross_task_specialisation_component",
        "latency_efficiency": "latency_efficiency_component",
        "memory_efficiency": "memory_efficiency_component",
        "seed_reliability": "seed_reliability_score",
        "architectural_diversity": "architectural_diversity_component",
        "token_interface_readiness": "token_interface_readiness_component",
    }
    work["selection_score"] = sum(work[component_columns[key]] * float(value) for key, value in weights.items())
    work["performance_score"] = work["multitask_performance_score"]
    work["efficiency_score"] = 0.625 * work["latency_efficiency_component"] + 0.375 * work["memory_efficiency_component"]
    return work.sort_values(["selection_score", "multitask_performance_score", "config_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def _aggregate_experts(pair_scores: pd.DataFrame, efficiency: pd.DataFrame, kind: str) -> pd.DataFrame:
    identifier = f"{kind}_encoder"
    family = f"{kind}_architecture_family"
    grouped = pair_scores.groupby(identifier, as_index=False).agg(
        multitask_performance_score=("multitask_performance_score", "mean"),
        best_pair_performance=("multitask_performance_score", "max"),
        unique_win_contribution=("unique_win_contribution", "mean"),
        oracle_contribution=("oracle_contribution", "mean"),
        cross_task_specialisation=("cross_task_specialisation", "mean"),
        seed_reliability_score=("seed_reliability_score", "mean"),
        seed_evidence_pairs=("seed_runs", lambda values: int((values > 0).sum())),
        selection_score=("selection_score", "mean"),
    )
    efficiency_id = identifier
    columns = [efficiency_id, "architecture_family", "latency_ms", "peak_allocated_bytes", "parameters", "output_dim"]
    missing = set(columns) - set(efficiency.columns)
    if missing:
        raise ValueError(f"{kind.title()} efficiency results are missing: {sorted(missing)}")
    grouped = grouped.merge(efficiency[columns], on=identifier, how="left")
    grouped["token_interface_expected"] = grouped[identifier].isin(TOKEN_READY_VISION if kind == "vision" else TOKEN_READY_TEXT)
    grouped["latency_score"] = _score_lower(grouped["latency_ms"])
    grouped["memory_score"] = _score_lower(grouped["peak_allocated_bytes"])
    grouped["efficiency_score"] = 0.6 * grouped["latency_score"] + 0.4 * grouped["memory_score"]
    grouped["selection_score"] = 0.75 * grouped["selection_score"] + 0.25 * grouped["efficiency_score"]
    best = pair_scores.sort_values("selection_score", ascending=False).drop_duplicates(identifier)[[identifier, "config_id", "reliability_status"]]
    return grouped.merge(best.rename(columns={"config_id": "best_pair"}), on=identifier, how="left").sort_values(["selection_score", identifier], ascending=[False, True], kind="stable").reset_index(drop=True)


def _select_primary_experts(frame: pd.DataFrame, identifier: str, limit: int, require_family_diversity: bool) -> tuple[pd.DataFrame, pd.Series]:
    limit = min(limit, len(frame))
    best_combo: tuple[int, ...] | None = None
    best_score = -float("inf")
    for combo in itertools.combinations(frame.index, limit):
        selected = frame.loc[list(combo)]
        if require_family_diversity and limit >= 2 and selected["architecture_family"].nunique() < 2:
            continue
        score = float(selected["selection_score"].sum())
        key = tuple(selected[identifier].sort_values())
        if score > best_score or (np.isclose(score, best_score) and (best_combo is None or key < tuple(frame.loc[list(best_combo), identifier].sort_values()))):
            best_combo, best_score = combo, score
    if best_combo is None:
        best_combo = tuple(frame.head(limit).index)
    primary = frame.loc[list(best_combo)].sort_values(["selection_score", identifier], ascending=[False, True], kind="stable").reset_index(drop=True)
    primary["selection_order"] = range(1, len(primary) + 1)
    primary["role"] = "primary"
    remaining = frame[~frame[identifier].isin(primary[identifier])].copy()
    fallback_source = remaining if not remaining.empty else frame
    fallback = fallback_source.sort_values(["efficiency_score", "selection_score", identifier], ascending=[False, False, True], kind="stable").iloc[0]
    return primary, fallback


def _remove_inconclusive_duplicates(pair_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept: list[pd.DataFrame] = []
    excluded: list[dict[str, Any]] = []
    for _, group in pair_scores.groupby(["vision_encoder", "text_encoder"], sort=False):
        baseline = group[group["variant"].eq("baseline")]
        if baseline.empty:
            kept.append(group)
            continue
        kept.append(baseline)
        for row in group[~group["variant"].eq("baseline")].itertuples():
            if row.reliability_status == "supported":
                kept.append(group[group["config_id"].eq(row.config_id)])
            else:
                excluded.append({"config_id": row.config_id, "selection_score": row.selection_score, "exclusion_reason": f"{row.variant} BLF is {row.reliability_status}; prefer baseline and prohibit a redundant primary path"})
    return pd.concat(kept, ignore_index=True).drop_duplicates("config_id"), pd.DataFrame(excluded)


def _valid_pair_combo(selected: pd.DataFrame, constraints: dict[str, Any], low_cost_threshold: float, unique_threshold: float) -> bool:
    if selected["vision_encoder"].value_counts().max() > int(constraints["maximum_paths_per_vision_encoder"]):
        return False
    if selected["text_encoder"].value_counts().max() > int(constraints["maximum_paths_per_text_encoder"]):
        return False
    if selected["vision_architecture_family"].nunique() < int(constraints["minimum_vision_architecture_families"]):
        return False
    if selected["text_encoder"].nunique() < int(constraints["minimum_distinct_text_encoders"]):
        return False
    if bool(constraints.get("require_low_cost_pair")) and not selected["latency_ms"].le(low_cost_threshold).any():
        return False
    if bool(constraints.get("require_unique_error_pair")) and not selected["unique_win_contribution"].ge(unique_threshold).any():
        return False
    return True


def _select_pair_shortlist(
    pair_scores: pd.DataFrame,
    visions: pd.DataFrame,
    texts: pd.DataFrame,
    vision_fallback: pd.Series,
    text_fallback: pd.Series,
    constraints: dict[str, Any],
    pair_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible, excluded = _remove_inconclusive_duplicates(pair_scores)
    allowed_visions = set(visions["vision_encoder"]) | {str(vision_fallback["vision_encoder"])}
    allowed_texts = set(texts["text_encoder"]) | {str(text_fallback["text_encoder"])}
    eligible = eligible[eligible["vision_encoder"].isin(allowed_visions) & eligible["text_encoder"].isin(allowed_texts)].copy()
    primary_count = min(int(constraints.get("primary_pair_paths", 4)), pair_limit, len(eligible))
    low_cost_threshold = float(eligible["latency_ms"].quantile(0.25))
    unique_threshold = float(eligible["unique_win_contribution"].quantile(0.75))
    best_combo: tuple[int, ...] | None = None
    best_score = -float("inf")
    for combo in itertools.combinations(eligible.index, primary_count):
        selected = eligible.loc[list(combo)]
        if not _valid_pair_combo(selected, constraints, low_cost_threshold, unique_threshold):
            continue
        score = float(selected["selection_score"].sum())
        ids = tuple(sorted(selected["config_id"]))
        if score > best_score or (np.isclose(score, best_score) and (best_combo is None or ids < tuple(sorted(eligible.loc[list(best_combo), "config_id"])))):
            best_combo, best_score = combo, score
    warnings: list[str] = []
    if best_combo is None:
        warnings.append("No pair combination satisfied every hard constraint; deterministic score order was used and readiness will fail diagnostics.")
        best_combo = tuple(eligible.head(primary_count).index)
    primary = eligible.loc[list(best_combo)].sort_values(["selection_score", "config_id"], ascending=[False, True], kind="stable").copy()
    primary["shortlist_role"] = "primary"
    selected_indices = list(primary.index)
    remaining = eligible.drop(selected_indices).sort_values(["selection_score", "config_id"], ascending=[False, True], kind="stable")
    for index, row in remaining.iterrows():
        if len(selected_indices) >= min(pair_limit, int(constraints.get("maximum_pair_paths", pair_limit))):
            break
        trial = eligible.loc[selected_indices + [index]]
        if trial["vision_encoder"].value_counts().max() <= int(constraints["maximum_paths_per_vision_encoder"]) and trial["text_encoder"].value_counts().max() <= int(constraints["maximum_paths_per_text_encoder"]):
            selected_indices.append(index)
    shortlist = eligible.loc[selected_indices].copy()
    shortlist["shortlist_role"] = ["primary" if index in primary.index else "optional_ablation" for index in shortlist.index]
    shortlist["_role_order"] = shortlist["shortlist_role"].map({"primary": 0, "optional_ablation": 1})
    shortlist = shortlist.sort_values(["_role_order", "selection_score", "config_id"], ascending=[True, False, True], kind="stable").drop(columns="_role_order").reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", range(1, len(shortlist) + 1))
    selected_ids = set(shortlist["config_id"])
    redundant = pair_scores[~pair_scores["config_id"].isin(selected_ids)].head(20)
    additional = [
        {"config_id": row.config_id, "selection_score": row.selection_score, "exclusion_reason": "lower score or diversity/cost constraint"}
        for row in redundant.itertuples()
        if row.config_id not in set(excluded.get("config_id", []))
    ]
    excluded = pd.concat([excluded, pd.DataFrame(additional)], ignore_index=True)
    diagnostics = {
        "low_cost_latency_threshold_ms": low_cost_threshold,
        "unique_error_threshold": unique_threshold,
        "warnings": warnings,
        "primary_constraints_satisfied": _valid_pair_combo(primary, constraints, low_cost_threshold, unique_threshold),
    }
    return shortlist, excluded, diagnostics


def _expert_role(kind: str, row: pd.Series, best_score: float, fallback: bool = False) -> str:
    if fallback:
        return f"cheap {kind} expert"
    if np.isclose(float(row["selection_score"]), best_score):
        return "strong semantic vision expert" if kind == "vision" else "strong general text expert"
    if kind == "vision" and row["architecture_family"] in {"hierarchical_vit", "convnext_cnn"}:
        return "local/hierarchical vision expert"
    return f"complementary {kind} expert"


def _validate_selected_yaml(payload: dict[str, Any]) -> None:
    required = {"schema_version", "research_scope", "selection_protocol", "vision_experts", "text_experts", "cheap_fallback", "phase2_pair_shortlist", "excluded_high_scoring_pairs"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"selected_experts.yaml is missing keys: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("selected_experts.yaml schema_version must be 1")
    for key in ("vision_experts", "text_experts"):
        if not isinstance(payload[key], list) or not payload[key]:
            raise ValueError(f"{key} must be a non-empty list")
        for item in payload[key]:
            for field in SelectedExpert.__required_keys__:
                if field not in item:
                    raise ValueError(f"{key} item is missing {field}")
    if not isinstance(payload["phase2_pair_shortlist"], list) or not payload["phase2_pair_shortlist"]:
        raise ValueError("phase2_pair_shortlist must be non-empty")
    for item in payload["phase2_pair_shortlist"]:
        for field in SelectedPair.__required_keys__:
            if field not in item:
                raise ValueError(f"phase2_pair_shortlist item is missing {field}")


def _yaml_expert(row: pd.Series, kind: str, role: str) -> SelectedExpert:
    identifier = str(row[f"{kind}_encoder"])
    reliability = str(row.get("reliability_status", "not_tested"))
    return {
        "canonical_name": identifier,
        "role": role,
        "architecture_family": str(row["architecture_family"]),
        "checkpoint_requirement": "alignment checkpoint from a selected VE–TE path",
        "token_interface_expected": bool(row["token_interface_expected"]),
        "latency_ms": float(row["latency_ms"]),
        "peak_memory_mb": float(row["peak_allocated_bytes"]) / 1024**2,
        "parameter_count": int(row["parameters"]),
        "reliability_status": reliability,
        "reasons": [
            f"selection score {float(row['selection_score']):.4f}",
            f"best associated pair {row['best_pair']}",
            f"measured latency {float(row['latency_ms']):.3f} ms",
        ],
    }


def _yaml_pair(row: pd.Series) -> SelectedPair:
    checkpoint = Path(str(row.get("checkpoint", "")))
    return {
        "pair_id": str(row["config_id"]),
        "vision_encoder": str(row["vision_encoder"]),
        "text_encoder": str(row["text_encoder"]),
        "variant": str(row["variant"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_fingerprint": stat_fingerprint(checkpoint) if checkpoint.exists() else "missing",
        "selection_score": float(row["selection_score"]),
        "unique_win_contribution": float(row["unique_win_contribution"]),
        "oracle_contribution": float(row["oracle_contribution"]),
        "measured_pair_latency_ms": float(row["latency_ms"]),
        "reliability_status": str(row["reliability_status"]),
        "intended_role": str(row["shortlist_role"]),
        "reasons": [
            f"multi-task score {float(row['multitask_performance_score']):.4f}",
            f"efficiency score {float(row['efficiency_score']):.4f}",
            f"unique-win contribution {float(row['unique_win_contribution']):.6f}",
        ],
    }


def run_expert_selection(
    project_root: str | Path,
    vision_limit: int = 3,
    text_limit: int = 3,
    pair_limit: int = 6,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / "results/phase15/expert_selection"
    output.mkdir(parents=True, exist_ok=True)
    evidence_root = resolve_evaluation_root(root)
    try:
        protocol = load_evaluation_protocol(root)
    except FileNotFoundError:
        protocol = {
            "selection_weights": DEFAULT_SELECTION_WEIGHTS,
            "diversity_constraints": DEFAULT_DIVERSITY_CONSTRAINTS,
        }
    weights = {key: float(value) for key, value in protocol.get("selection_weights", DEFAULT_SELECTION_WEIGHTS).items()}
    constraints = dict(protocol["diversity_constraints"])
    constraints["primary_vision_experts"] = vision_limit
    constraints["primary_text_experts"] = text_limit
    constraints["maximum_pair_paths"] = pair_limit
    paths = {
        "results": evidence_root / "task_results_long.csv",
        "retrieval_pairwise": root / "results/phase15/complementarity/retrieval_pairwise.csv",
        "classification_pairwise": root / "results/phase15/complementarity/classification_pairwise.csv",
        "oracle_marginal": root / "results/phase15/oracle/classification_marginal_contributions.csv",
        "specialisation": root / "results/phase15/cross_task/expert_specialisation.csv",
        "pair_efficiency": root / "results/phase15/efficiency/full_pairs.csv",
        "vision_efficiency": root / "results/phase15/efficiency/vision_encoders.csv",
        "text_efficiency": root / "results/phase15/efficiency/text_encoders.csv",
        "seed_results": root / "results/seed_sweep_results.csv",
    }
    required = [paths[key] for key in ("results", "retrieval_pairwise", "pair_efficiency", "vision_efficiency", "text_efficiency")]
    missing = [str(value) for value in required if not value.exists()]
    if missing:
        raise FileNotFoundError(f"Expert selection prerequisites are missing: {missing}")
    results = pd.read_csv(paths["results"])
    if "mode" in results:
        results = results[results["mode"].eq("full")]
    if paths["seed_results"].exists():
        run_seed_reliability_mapping(root)
    pair_scores = build_pair_scores(
        results,
        pd.read_csv(paths["retrieval_pairwise"]),
        pd.read_csv(paths["pair_efficiency"]),
        paths["seed_results"],
        classification_pairwise_path=paths["classification_pairwise"],
        oracle_marginal_path=paths["oracle_marginal"],
        specialisation_path=paths["specialisation"],
        selection_weights=weights,
    )
    vision_ranked = _aggregate_experts(pair_scores, pd.read_csv(paths["vision_efficiency"]), "vision")
    text_ranked = _aggregate_experts(pair_scores, pd.read_csv(paths["text_efficiency"]), "text")
    visions, vision_fallback = _select_primary_experts(vision_ranked, "vision_encoder", vision_limit, True)
    texts, text_fallback = _select_primary_experts(text_ranked, "text_encoder", text_limit, False)
    shortlist, excluded, diagnostics = _select_pair_shortlist(
        pair_scores, visions, texts, vision_fallback, text_fallback, constraints, pair_limit
    )
    vision_best = float(visions["selection_score"].max())
    text_best = float(texts["selection_score"].max())
    visions["evidence_role"] = [_expert_role("vision", row, vision_best) for _, row in visions.iterrows()]
    texts["evidence_role"] = [_expert_role("text", row, text_best) for _, row in texts.iterrows()]
    vision_fallback_role = _expert_role("vision", vision_fallback, vision_best, True)
    text_fallback_role = _expert_role("text", text_fallback, text_best, True)
    outputs = {
        "vision_ranked": output / "vision_experts_ranked.csv",
        "text_ranked": output / "text_experts_ranked.csv",
        "pairs_ranked": output / "pairs_ranked.csv",
        "vision_experts": output / "vision_experts.csv",
        "text_experts": output / "text_experts.csv",
        "pair_scores": output / "pair_scores.csv",
        "pair_shortlist": output / "pair_shortlist.csv",
        "explanation": output / "selection_explanation.md",
        "diagnostics": output / "selection_diagnostics.json",
        "yaml": output / "selected_experts.yaml",
        "json": output / "expert_selection.json",
        "markdown": output / "expert_selection.md",
    }
    for key, frame in (
        ("vision_ranked", vision_ranked),
        ("text_ranked", text_ranked),
        ("pairs_ranked", pair_scores),
        ("vision_experts", visions),
        ("text_experts", texts),
        ("pair_scores", pair_scores),
        ("pair_shortlist", shortlist),
    ):
        atomic_csv(frame, outputs[key])
    primary = shortlist[shortlist["shortlist_role"].eq("primary")]
    constraints_result = {
        "at_most_two_per_vision": bool(primary["vision_encoder"].value_counts().max() <= int(constraints["maximum_paths_per_vision_encoder"])),
        "at_most_two_per_text": bool(primary["text_encoder"].value_counts().max() <= int(constraints["maximum_paths_per_text_encoder"])),
        "vision_architecture_families": int(primary["vision_architecture_family"].nunique()),
        "distinct_text_encoders": int(primary["text_encoder"].nunique()),
        "contains_low_cost_pair": bool(primary["latency_ms"].le(diagnostics["low_cost_latency_threshold_ms"]).any()),
        "contains_unique_error_pair": bool(primary["unique_win_contribution"].ge(diagnostics["unique_error_threshold"]).any()),
        "contains_inconclusive_blf_duplicate": bool(primary.duplicated(["vision_encoder", "text_encoder"], keep=False).any()),
    }
    diagnostics.update(
        {
            "selection_weights": weights,
            "diversity_constraints": constraints,
            "constraint_results": constraints_result,
            "selection_data_status": "frozen development protocol" if evidence_root.name.endswith("_development") else "exploratory/development; historical classification test outputs were used",
            "excluded_high_scoring_pairs": dataframe_records(excluded.head(20)),
        }
    )
    atomic_json(diagnostics, outputs["diagnostics"])
    source_fingerprints = {
        key: stat_fingerprint(value)
        for key, value in paths.items()
        if value.exists() and value.is_file()
    }
    yaml_payload = {
        "schema_version": 1,
        "research_scope": {
            "tasks": ["retrieval", "classification", "compositional_if_available"],
            "pretrained_backbones_frozen": True,
        },
        "selection_protocol": {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "development_protocol_path": str(root / "configs/evaluation_protocol.yaml"),
            "selection_status": "development_protocol_compliant" if evidence_root.name.endswith("_development") else "exploratory_until_development_split_predictions_replace_historical_test_outputs",
            "selection_weights": weights,
            "diversity_constraints": constraints,
            "source_result_fingerprints": source_fingerprints,
        },
        "vision_experts": [
            _yaml_expert(row, "vision", str(row["evidence_role"])) for _, row in visions.iterrows()
        ],
        "text_experts": [
            _yaml_expert(row, "text", str(row["evidence_role"])) for _, row in texts.iterrows()
        ],
        "cheap_fallback": {
            "vision_encoder": str(vision_fallback["vision_encoder"]),
            "text_encoder": str(text_fallback["text_encoder"]),
            "reasons": [
                f"{vision_fallback_role}: {float(vision_fallback['latency_ms']):.3f} ms",
                f"{text_fallback_role}: {float(text_fallback['latency_ms']):.3f} ms",
            ],
        },
        "phase2_pair_shortlist": [_yaml_pair(row) for _, row in shortlist.iterrows()],
        "excluded_high_scoring_pairs": [
            {"pair_id": str(row["config_id"]), "exclusion_reason": str(row["exclusion_reason"])}
            for _, row in excluded.head(20).iterrows()
        ],
    }
    _validate_selected_yaml(yaml_payload)
    atomic_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), outputs["yaml"])
    explanation = [
        "# Diversity-constrained expert selection",
        "",
        "This shortlist uses the frozen development protocol and is not a final-test claim." if evidence_root.name.endswith("_development") else "This is an exploratory development shortlist, not a final-test claim. Historical classification test outputs remain labelled exploratory and must be replaced by the frozen development protocol before Phase 2 readiness can pass.",
        "",
        "## Primary vision experts",
        "",
        visions.to_markdown(index=False),
        "",
        "## Primary text experts",
        "",
        texts.to_markdown(index=False),
        "",
        "## Constrained pair paths",
        "",
        shortlist.to_markdown(index=False),
        "",
        "## Cheap fallback",
        "",
        f"- Vision: {vision_fallback['vision_encoder']} ({float(vision_fallback['latency_ms']):.3f} ms)",
        f"- Text: {text_fallback['text_encoder']} ({float(text_fallback['latency_ms']):.3f} ms)",
        "",
        "Inconclusive BLF variants are not allowed to duplicate their matched baseline in the primary shortlist. Oracle evidence is a non-deployable diagnostic upper bound.",
    ]
    atomic_text("\n".join(explanation) + "\n", outputs["explanation"])
    atomic_text("\n".join(explanation) + "\n", outputs["markdown"])
    payload: dict[str, Any] = {
        "weights": weights,
        "diversity_constraints": constraints,
        "vision_experts": dataframe_records(visions),
        "text_experts": dataframe_records(texts),
        "cheap_fallback": {
            "vision_encoder": str(vision_fallback["vision_encoder"]),
            "text_encoder": str(text_fallback["text_encoder"]),
        },
        "pair_shortlist": dataframe_records(shortlist),
        "excluded_high_scoring_pairs": dataframe_records(excluded.head(20)),
        "selection_status": ("frozen development shortlist" if evidence_root.name.endswith("_development") else "exploratory development shortlist") + "; no Phase 2 method implemented",
        "canonical_yaml": str(outputs["yaml"]),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    atomic_json(payload, outputs["json"])
    return payload
