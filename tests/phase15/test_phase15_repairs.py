from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.configuration_ids import canonical_configuration_id, canonical_variant
from src.phase15.classification_complementarity import (
    classification_pair_metrics,
    classification_per_class_metrics,
)
from src.phase15.cross_task_analysis import (
    cost_adjusted_score,
    normalise_task_scores,
    task_rank_correlations,
    task_specialisation_score,
)
from src.phase15.evaluation_protocol import stratified_split_manifest
from src.phase15.expert_selection import _remove_inconclusive_duplicates, _valid_pair_combo
from src.phase15.hard_negative_mining import USAGE as DIAGNOSTIC_USAGE
from src.phase15.oracle_analysis import classification_oracle_subset
from src.phase15.phase2_readiness import SAFE_DEFAULTS, _training_negative_check, run_phase2_readiness
from src.phase15.prediction_validation import (
    _checkpoint_status,
    validate_classification_rows,
    validate_retrieval_rows,
)
from src.phase15.seed_reliability import build_seed_reliability
from src.phase15 import training_hard_negative_mining as training_mining
from src.multitask.checkpoint_selection import CheckpointSpec
from src.phase15.training_hard_negative_mining import (
    annotate_training_rows,
    validate_no_protected_ids,
    validate_training_negative_metadata,
)


def _retrieval_row(index: int, direction: str) -> dict[str, object]:
    return {
        "direction": direction,
        "query_id": f"q{index}",
        "correct_target_rank": 1,
        "recall_at_1": True,
        "recall_at_5": True,
        "recall_at_10": True,
        "top_candidate_ids": ["candidate"],
        "top_candidate_scores": [0.5],
    }


def _classification_frame(correct: list[bool], predictions: list[int] | None = None) -> pd.DataFrame:
    targets = [0, 0, 1, 1][: len(correct)]
    predicted = predictions or [target if value else 1 - target for target, value in zip(targets, correct)]
    return pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(len(correct))],
            "target_index": targets,
            "target": ["zero" if value == 0 else "one" for value in targets],
            "prediction_index": predicted,
            "correct": correct,
            "confidence": np.linspace(0.55, 0.85, len(correct)),
            "classification_margin": np.linspace(-0.2, 0.4, len(correct)),
        }
    )


def test_notebook_defaults_integrity_and_phase_order():
    path = Path("notebooks/01_experiment_workflow.ipynb")
    notebook = json.loads(path.read_text())
    by_id = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    setup = by_id["step-0-controls"]
    for key, value in SAFE_DEFAULTS.items():
        rendered = repr(value) if isinstance(value, str) else str(value)
        assert f"{key} = {rendered}" in setup or f'{key} = "{value}"' in setup
    headings = [by_id[f"step-{number}-md"] for number in range(13, 18)]
    indices = [next(index for index, cell in enumerate(notebook["cells"]) if cell.get("id") == f"step-{number}-md") for number in range(13, 18)]
    assert indices == sorted(indices)
    assert all(f"Step {number}" in heading for number, heading in zip(range(13, 18), headings))
    all_source = "\n".join(by_id.values())
    assert "src.phase2" in all_source and "# Phase 2" in all_source
    assert "RUN_PHASE2_SMOKE_TESTS = False" in by_id["phase2-controls"]
    assert "RUN_PHASE2_DDP_SMOKE_TEST = False" in by_id["phase2-controls"]
    assert "subprocess.run" not in "\n".join(by_id[cell_id] for cell_id in by_id if cell_id.startswith("phase2-"))
    assert "RESULTS_ONLY" in by_id["step-14-complementarity"]
    assert "RUN_EFFICIENCY_PROFILING" in by_id["step-15-efficiency"] and "RESULTS_ONLY" in by_id["step-15-efficiency"]
    assert "RUN_TRAINING_HARD_NEGATIVE_MINING" in by_id["step-16-hard-negatives"] and "RESULTS_ONLY" in by_id["step-16-hard-negatives"]
    for cell_id in ("step-13-export", "step-14-complementarity", "step-15-efficiency", "step-16-hard-negatives", "step-17-selection"):
        compile(by_id[cell_id], f"{path}:{cell_id}", "exec")


def test_retrieval_validator_handles_standard_coco_query_counts():
    image_rows = [_retrieval_row(index, "i2t") for index in range(5000)]
    caption_rows = [_retrieval_row(index, "t2i") for index in range(25014)]
    assert validate_retrieval_rows(image_rows, direction="i2t", expected_ids=[f"q{i}" for i in range(5000)])["observed_samples"] == 5000
    assert validate_retrieval_rows(caption_rows, direction="t2i", expected_ids=[f"q{i}" for i in range(25014)])["observed_samples"] == 25014


def test_prediction_validator_detects_duplicates_missing_nonfinite_and_bad_recall():
    rows = [_retrieval_row(0, "i2t"), _retrieval_row(0, "i2t")]
    rows[0]["top_candidate_scores"] = [float("nan")]
    rows[1]["correct_target_rank"] = 20
    result = validate_retrieval_rows(rows, direction="i2t", expected_ids=["q0", "q1"])
    assert result["duplicate_count"] == 1
    assert result["missing_count"] == 1
    assert result["nonfinite_score_count"] == 1
    assert result["invalid_prediction_count"] == 1
    assert result["valid"] is False


def test_classification_validator_and_checkpoint_fingerprint_mismatch(tmp_path):
    row = {
        "task": "tiny",
        "sample_id": "s0",
        "target_index": 0,
        "target": "zero",
        "prediction_index": 0,
        "prediction": "zero",
        "correct": True,
        "confidence": 0.8,
        "correct_class_score": 0.8,
        "highest_incorrect_index": 1,
        "highest_incorrect_score": 0.2,
        "classification_margin": 0.6,
        "top1_margin": 0.6,
        "top5": [{"score": 0.8}],
    }
    assert validate_classification_rows([row], task="tiny", expected_ids=["s0"], expected_classes=2)["valid"]
    checkpoint = tmp_path / "dinov2_vits14_all_minilm_l6_v2_baseline" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    status, _ = _checkpoint_status(
        "dinov2_vits14__all_minilm_l6_v2__baseline",
        checkpoint,
        {"checkpoint": str(checkpoint), "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns - 1},
    )
    assert status == "mismatch"


def test_classification_complementarity_and_per_class_values():
    left = _classification_frame([True, True, False, False], [0, 0, 0, 0])
    right = _classification_frame([True, False, True, False], [0, 1, 1, 0])
    metrics = classification_pair_metrics(left, right)
    assert (metrics["both_correct"], metrics["only_a_correct"], metrics["only_b_correct"], metrics["both_incorrect"]) == (1, 1, 1, 1)
    assert metrics["correct_set_jaccard"] == pytest.approx(1 / 3)
    assert np.isfinite(metrics["confidence_pearson"])
    assert np.isfinite(metrics["margin_spearman"])
    per_class = classification_per_class_metrics(left, right)
    assert set(per_class["class_id"]) == {0, 1}
    assert int(per_class["both_incorrect"].sum()) == 1


def test_classification_oracle_pairs_triples_full_and_marginal():
    correct = {
        "a": [True, False, False, False],
        "b": [False, True, False, False],
        "c": [False, False, True, False],
    }
    pair = classification_oracle_subset(correct, ["a", "b"])
    triple = classification_oracle_subset(correct, ["a", "b", "c"])
    full = classification_oracle_subset(correct)
    assert pair["oracle_top1_accuracy"] == 0.5
    assert triple["oracle_top1_accuracy"] == full["oracle_top1_accuracy"] == 0.75
    assert triple["unique_contribution_per_configuration"] == {"a": 1, "b": 1, "c": 1}
    assert triple["oracle_top1_accuracy"] - np.delete(triple["matrix"], 0, axis=0).any(axis=0).mean() == pytest.approx(0.25)


def test_cross_task_normalisation_ranks_correlations_specialisation_and_cost():
    values = pd.Series([0.4, 0.8])
    assert normalise_task_scores(values).tolist() == [0.5, 1.0]
    assert normalise_task_scores(values, "min_max_within_task").tolist() == [0.0, 1.0]
    matrix = pd.DataFrame({"a": [3, 2, 1], "b": [30, 20, 10], "c": [1, 2, 3]})
    correlations = task_rank_correlations(matrix)
    assert correlations.query("task_a == 'a' and task_b == 'b'").iloc[0].spearman == pytest.approx(1.0)
    assert correlations.query("task_a == 'a' and task_b == 'c'").iloc[0].kendall == pytest.approx(-1.0)
    assert task_specialisation_score([1.0, 1.0]) == 0.0
    assert task_specialisation_score([0.5, 1.0]) > 0
    assert cost_adjusted_score(0.8, latency_ms=4.0, reference_latency_ms=1.0) == pytest.approx(0.4)


def test_seed_mapping_normalises_known_display_and_variant_aliases():
    rows = []
    for comparison, vision, variant in (
        ("DINOv2 MiniLM", "DINOv2 ViT-S/14", "baseline"),
        ("DINOv2 MiniLM", "DINOv2 ViT-S/14", "local only"),
        ("ConvNeXtV2 MiniLM", "ConvNeXt V2 Tiny", "baseline"),
        ("ConvNeXtV2 MiniLM", "ConvNeXt V2 Tiny", "local+global"),
    ):
        for seed in range(5):
            rows.append({"comparison": comparison, "vision_encoder": vision, "text_encoder": "MiniLM-L6", "variant": variant, "seed": seed, "coco5_i2t_R@1": 0.5 + 0.01 * seed})
    paired = pd.DataFrame(
        [
            {"comparison": "DINOv2 MiniLM", "variant": "local", "mean_delta": 0.01, "delta_ci95_half_width": 0.02, "wins": 3, "ties": 0, "losses": 2},
            {"comparison": "ConvNeXtV2 MiniLM", "variant": "local global", "mean_delta": 0.01, "delta_ci95_half_width": 0.02, "wins": 3, "ties": 0, "losses": 2},
        ]
    )
    mapped = build_seed_reliability(pd.DataFrame(rows), paired)
    assert set(mapped["config_id"]) == {
        "dinov2_vits14__all_minilm_l6_v2__baseline",
        "dinov2_vits14__all_minilm_l6_v2__local",
        "convnextv2_tiny__all_minilm_l6_v2__baseline",
        "convnextv2_tiny__all_minilm_l6_v2__local_global",
    }
    assert set(mapped.query("variant != 'baseline'")["reliability_status"]) == {"inconclusive"}
    assert canonical_variant("global only") == "global"
    assert canonical_configuration_id("job_001_dinov2_vits14_all_minilm_l6_v2_local.log".removesuffix(".log")) == "dinov2_vits14__all_minilm_l6_v2__local"


def test_pair_constraints_and_inconclusive_blf_duplicate_removal():
    selected = pd.DataFrame(
        [
            {"vision_encoder": "v1", "text_encoder": "t1", "vision_architecture_family": "cnn", "latency_ms": 1.0, "unique_win_contribution": 0.1},
            {"vision_encoder": "v1", "text_encoder": "t2", "vision_architecture_family": "cnn", "latency_ms": 2.0, "unique_win_contribution": 0.8},
            {"vision_encoder": "v2", "text_encoder": "t1", "vision_architecture_family": "vit", "latency_ms": 3.0, "unique_win_contribution": 0.2},
            {"vision_encoder": "v2", "text_encoder": "t2", "vision_architecture_family": "vit", "latency_ms": 4.0, "unique_win_contribution": 0.3},
        ]
    )
    constraints = {
        "maximum_paths_per_vision_encoder": 2,
        "maximum_paths_per_text_encoder": 2,
        "minimum_vision_architecture_families": 2,
        "minimum_distinct_text_encoders": 2,
        "require_low_cost_pair": True,
        "require_unique_error_pair": True,
    }
    assert _valid_pair_combo(selected, constraints, low_cost_threshold=1.5, unique_threshold=0.7)
    assert not _valid_pair_combo(selected.assign(text_encoder="t1"), constraints, 1.5, 0.7)
    candidates = pd.DataFrame(
        [
            {"config_id": "v__t__baseline", "vision_encoder": "v", "text_encoder": "t", "variant": "baseline", "reliability_status": "not_tested", "selection_score": 0.8},
            {"config_id": "v__t__local", "vision_encoder": "v", "text_encoder": "t", "variant": "local", "reliability_status": "inconclusive", "selection_score": 0.9},
            {"config_id": "v__t__global", "vision_encoder": "v", "text_encoder": "t", "variant": "global", "reliability_status": "supported", "selection_score": 0.85},
        ]
    )
    kept, excluded = _remove_inconclusive_duplicates(candidates)
    assert set(kept["variant"]) == {"baseline", "global"}
    assert excluded.iloc[0].config_id == "v__t__local"


def test_training_metadata_leakage_and_diagnostic_restriction():
    frame = annotate_training_rows(
        pd.DataFrame({"sample_id": ["train-a"], "task": ["cifar100_zeroshot"]}),
        source_split="development_partition",
        dataset="tiny",
        task="classification",
        configuration_id="config",
        checkpoint_fingerprint="fingerprint",
    )
    assert frame.loc[0, "task"] == "classification"
    assert frame.loc[0, "source_task"] == "cifar100_zeroshot"
    validate_training_negative_metadata(frame)
    validate_no_protected_ids(frame, {"test-a"})
    with pytest.raises(ValueError, match="Protected"):
        validate_no_protected_ids(frame.assign(sample_id="test-a"), {"test-a"})
    with pytest.raises(ValueError, match="Protected/evaluation"):
        validate_training_negative_metadata(frame.assign(source_split="official_test"))
    assert DIAGNOSTIC_USAGE == "diagnostic_only"


def test_training_negative_readiness_requires_completed_commit_marker(tmp_path):
    output = tmp_path / "results/phase15/training_hard_negatives"
    output.mkdir(parents=True)
    (output / "retrieval_train_hard_negatives.parquet").write_bytes(b"uncommitted")
    (output / "classification_train_hard_negatives.parquet").write_bytes(b"uncommitted")
    ready, details, leakage_ready, _ = _training_negative_check(tmp_path, set())
    assert not ready and not leakage_ready
    assert "commit marker" in details


def _training_retrieval_frame(configuration_id: str, fingerprint: str, count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "usage": ["phase2_training"] * count,
            "allowed_for_training": [True] * count,
            "source_split": ["coco_train2017"] * count,
            "dataset": ["coco"] * count,
            "task": ["retrieval"] * count,
            "configuration_id": [configuration_id] * count,
            "checkpoint_fingerprint": [fingerprint] * count,
            "candidate_rank": list(range(1, count + 1)),
        }
    )


def test_training_shard_resume_validates_identity_and_negative_count(tmp_path):
    path = tmp_path / "config__retrieval__full.parquet"
    identity = training_mining._shard_identity(
        configuration_id="config",
        checkpoint_fingerprint="fingerprint",
        task="retrieval",
        mode="full",
        count=2,
        source_fingerprint="source",
    )
    training_mining._write_parquet_chunks([_training_retrieval_frame("config", "fingerprint")], path)

    assert training_mining._valid_training_shard(path, identity) == 2
    assert training_mining._shard_sidecar(path).exists()
    assert training_mining._valid_training_shard(path, {**identity, "count": 3}) is None
    assert training_mining._valid_training_shard(
        path,
        {**identity, "checkpoint_fingerprint": "stale"},
    ) is None


def test_partial_resume_preserves_retrieval_and_only_mines_classification(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/vision_text_baseline/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    spec = CheckpointSpec("vision", "text", "baseline", checkpoint, 0)
    sources = {
        "coco_captions": str(tmp_path / "captions.csv"),
        "coco_fingerprint": "coco-source",
        "classification_fingerprint": "classification-source",
        "eurosat_manifest": str(tmp_path / "eurosat.csv"),
    }
    retrieval_identity, _ = training_mining._training_identities(spec, "full", 2, sources)
    retrieval_path, _, _ = training_mining._training_paths(tmp_path, spec.config_id, "full")
    training_mining._write_parquet_chunks(
        [_training_retrieval_frame(spec.config_id, retrieval_identity["checkpoint_fingerprint"])],
        retrieval_path,
    )
    training_mining._commit_shard_metadata(retrieval_path, retrieval_identity, 2)
    original = retrieval_path.read_bytes()
    calls = {"retrieval": 0, "classification": 0}

    monkeypatch.setattr(training_mining, "prepare_training_hard_negative_sources", lambda _root: sources)
    monkeypatch.setattr(training_mining, "_resolve_training_specs", lambda _root, _ids: [spec])
    monkeypatch.setattr(training_mining, "load_frozen_model", lambda *_args: (object(), {"data": {}, "training": {}}, 0))
    monkeypatch.setattr(training_mining.torch.cuda, "empty_cache", lambda: None)

    def fail_retrieval(*_args, **_kwargs):
        calls["retrieval"] += 1
        raise AssertionError("valid retrieval shard must not be recomputed")

    def finish_classification(*_args, **_kwargs):
        calls["classification"] += 1
        return 11

    monkeypatch.setattr(training_mining, "_mine_retrieval_for_configuration", fail_retrieval)
    monkeypatch.setattr(training_mining, "_mine_classification_for_configuration", finish_classification)

    result = training_mining.mine_training_hard_negatives_for_config(
        tmp_path,
        spec.config_id,
        mode="full",
        count=2,
        device="cuda:0",
        resume=True,
    )

    assert calls == {"retrieval": 0, "classification": 1}
    assert result["retrieval_status"] == "resumed"
    assert result["classification_status"] == "complete"
    assert retrieval_path.read_bytes() == original


def test_finalizer_refuses_to_publish_when_any_selected_shard_is_missing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/vision_text_baseline/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    spec = CheckpointSpec("vision", "text", "baseline", checkpoint, 0)
    sources = {
        "coco_captions": str(tmp_path / "captions.csv"),
        "coco_fingerprint": "coco-source",
        "classification_fingerprint": "classification-source",
        "eurosat_manifest": str(tmp_path / "eurosat.csv"),
    }
    monkeypatch.setattr(training_mining, "prepare_training_hard_negative_sources", lambda _root: sources)
    monkeypatch.setattr(training_mining, "_resolve_training_specs", lambda _root, _ids: [spec])

    with pytest.raises(RuntimeError, match="missing or stale shards"):
        training_mining.finalize_training_hard_negative_mining(
            tmp_path,
            mode="full",
            count=2,
            config_ids=[spec.config_id],
        )
    assert not (tmp_path / "results/phase15/training_hard_negatives/schema.json").exists()


def test_finalizer_keeps_canonical_outputs_unchanged_when_staging_fails(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/vision_text_baseline/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    spec = CheckpointSpec("vision", "text", "baseline", checkpoint, 0)
    sources = {
        "coco_captions": str(tmp_path / "captions.csv"),
        "coco_fingerprint": "coco-source",
        "classification_fingerprint": "classification-source",
        "eurosat_manifest": str(tmp_path / "eurosat.csv"),
    }
    output = tmp_path / "results/phase15/training_hard_negatives"
    output.mkdir(parents=True)
    canonical_retrieval = output / "retrieval_train_hard_negatives.parquet"
    canonical_classification = output / "classification_train_hard_negatives.parquet"
    canonical_retrieval.write_bytes(b"old-retrieval")
    canonical_classification.write_bytes(b"old-classification")
    canonical_schema = output / "schema.json"
    canonical_schema.write_bytes(b"old-schema")

    monkeypatch.setattr(training_mining, "prepare_training_hard_negative_sources", lambda _root: sources)
    monkeypatch.setattr(training_mining, "_resolve_training_specs", lambda _root, _ids: [spec])
    monkeypatch.setattr(training_mining, "_valid_training_shard", lambda *_args: 1)
    calls = 0

    def fail_second_combine(_shards, destination):
        nonlocal calls
        calls += 1
        destination.write_bytes(b"staged")
        if calls == 2:
            raise RuntimeError("classification combine failed")
        return 1

    monkeypatch.setattr(training_mining, "_combine_parquet", fail_second_combine)
    with pytest.raises(RuntimeError, match="classification combine failed"):
        training_mining.finalize_training_hard_negative_mining(
            tmp_path,
            mode="full",
            count=2,
            config_ids=[spec.config_id],
        )

    assert canonical_retrieval.read_bytes() == b"old-retrieval"
    assert canonical_classification.read_bytes() == b"old-classification"
    assert canonical_schema.read_bytes() == b"old-schema"


def test_finalizer_publishes_hash_verified_commit_marker(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/vision_text_baseline/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    spec = CheckpointSpec("vision", "text", "baseline", checkpoint, 0)
    sources = {
        "coco_captions": str(tmp_path / "captions.csv"),
        "coco_fingerprint": "coco-source",
        "classification_fingerprint": "classification-source",
        "eurosat_manifest": str(tmp_path / "eurosat.csv"),
    }
    retrieval_identity, classification_identity = training_mining._training_identities(spec, "full", 2, sources)
    retrieval_path, classification_path, _ = training_mining._training_paths(tmp_path, spec.config_id, "full")
    retrieval = _training_retrieval_frame(spec.config_id, retrieval_identity["checkpoint_fingerprint"])
    retrieval["query_id"] = ["train-image-1", "train-image-1"]
    retrieval["positive_id"] = ["train-image-1", "train-image-1"]
    classification = pd.DataFrame(
        {
            "usage": ["phase2_training"],
            "allowed_for_training": [True],
            "source_split": ["cifar100_official_train_train_partition"],
            "dataset": ["cifar100"],
            "task": ["classification"],
            "configuration_id": [spec.config_id],
            "checkpoint_fingerprint": [classification_identity["checkpoint_fingerprint"]],
            "sample_id": ["train-sample-1"],
            "hard_negative_rank": [1],
        }
    )
    training_mining._write_parquet_chunks([retrieval], retrieval_path)
    training_mining._commit_shard_metadata(retrieval_path, retrieval_identity, len(retrieval))
    training_mining._write_parquet_chunks([classification], classification_path)
    training_mining._commit_shard_metadata(classification_path, classification_identity, len(classification))
    monkeypatch.setattr(training_mining, "prepare_training_hard_negative_sources", lambda _root: sources)
    monkeypatch.setattr(training_mining, "_resolve_training_specs", lambda _root, _ids: [spec])
    monkeypatch.setattr(training_mining, "protected_test_ids", lambda _root: set())

    result = training_mining.finalize_training_hard_negative_mining(
        tmp_path,
        mode="full",
        count=2,
        config_ids=[spec.config_id],
    )
    schema = json.loads((tmp_path / "results/phase15/training_hard_negatives/schema.json").read_text())
    assert result["status"] == "complete"
    assert schema["status"] == "complete"
    assert set(schema["output_sha256"]) == {"retrieval", "classification"}
    ready, _, leakage_ready, _ = _training_negative_check(tmp_path, set())
    assert ready and leakage_ready


def test_eurosat_split_is_deterministic_stratified_and_disjoint():
    ids = [f"class-{label}-{index}" for label in range(3) for index in range(20)]
    labels = [label for label in range(3) for _ in range(20)]
    names = [f"class-{value}" for value in labels]
    first = stratified_split_manifest(ids, labels, names, seed=42)
    second = stratified_split_manifest(ids, labels, names, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert not first["sample_id"].duplicated().any()
    assert set(first["split"]) == {"train", "development", "test"}
    assert (first.groupby(["class_id", "split"]).size() > 0).all()


def _write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def _complete_readiness_fixture(root: Path) -> None:
    notebook = root / "notebooks/01_experiment_workflow.ipynb"
    notebook.parent.mkdir(parents=True)
    source = "\n".join(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}" for key, value in SAFE_DEFAULTS.items())
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source.splitlines(True)}]}))
    _write_csv(root / "results/alignment_matrix_best_clean.csv", {"vision_encoder": "dinov2_vits14", "text_encoder": "all_minilm_l6_v2", "variant": "baseline"})
    _write_csv(root / "results/seed_sweep_results.csv", {"seed": 42, "variant": "baseline"})
    development = root / "results/phase1_multitask_development"
    _write_csv(development / "task_results_long.csv", {"config_id": "x", "task": "coco_retrieval", "metric": "r1", "value": 1.0, "dataset_split": "coco_val2017_development"})
    (development / "development_evidence_manifest.json").write_text(json.dumps({"status": "complete"}))
    _write_csv(root / "results/phase15/prediction_validation/coverage_report.csv", {"valid": True, "direction": "i2t", "configuration_id": "x", "task": "coco_retrieval", "dataset_split": "coco_val2017_development"})
    required = {
        "complementarity/retrieval_pairwise.csv": {"expert_a": "a", "expert_b": "b"},
        "complementarity/classification_pairwise.csv": {"dataset": "cifar100", "configuration_a": "a", "configuration_b": "b"},
        "oracle/retrieval_oracle.csv": {"expert_a": "a", "expert_b": "b"},
        "oracle/classification_oracle.csv": {"dataset": "cifar100", "subset_id": "a-b", "oracle_top1_accuracy": 1.0},
        "cross_task/task_performance_matrix_normalised.csv": {"config_id": "x"},
        "cross_task/configuration_rankings.csv": {"config_id": "x"},
        "efficiency/full_pairs.csv": {"config_id": "x", "latency_ms": 1.0, "peak_allocated_bytes": 1},
        "expert_selection/seed_reliability.csv": {"config_id": "x", "seed_runs": 5, "reliability_status": "supported"},
        "compositional/task_status.csv": {"task": "winoground", "status": "missing"},
    }
    for relative, row in required.items():
        _write_csv(root / "results/phase15" / relative, row)
    checkpoint_a = root / "checkpoints/a.pt"
    checkpoint_b = root / "checkpoints/b.pt"
    checkpoint_a.parent.mkdir(parents=True)
    checkpoint_a.write_bytes(b"a")
    checkpoint_b.write_bytes(b"b")
    selected = {
        "schema_version": 1,
        "selection_protocol": {"diversity_constraints": {"maximum_paths_per_vision_encoder": 2, "maximum_paths_per_text_encoder": 2, "minimum_vision_architecture_families": 2, "minimum_distinct_text_encoders": 2}},
        "vision_experts": [
            {"canonical_name": "dinov2_vits14", "architecture_family": "vit"},
            {"canonical_name": "convnext_tiny", "architecture_family": "cnn"},
        ],
        "text_experts": [{"canonical_name": "all_minilm_l6_v2"}, {"canonical_name": "bge_small_en"}],
        "phase2_pair_shortlist": [
            {"pair_id": "p1", "vision_encoder": "dinov2_vits14", "text_encoder": "all_minilm_l6_v2", "variant": "baseline", "reliability_status": "not_tested", "intended_role": "primary", "checkpoint_path": str(checkpoint_a)},
            {"pair_id": "p2", "vision_encoder": "dinov2_vits14", "text_encoder": "bge_small_en", "variant": "baseline", "reliability_status": "not_tested", "intended_role": "primary", "checkpoint_path": str(checkpoint_a)},
            {"pair_id": "p3", "vision_encoder": "convnext_tiny", "text_encoder": "all_minilm_l6_v2", "variant": "baseline", "reliability_status": "not_tested", "intended_role": "primary", "checkpoint_path": str(checkpoint_b)},
            {"pair_id": "p4", "vision_encoder": "convnext_tiny", "text_encoder": "bge_small_en", "variant": "baseline", "reliability_status": "not_tested", "intended_role": "primary", "checkpoint_path": str(checkpoint_b)},
        ],
    }
    selection = root / "results/phase15/expert_selection/selected_experts.yaml"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(yaml.safe_dump(selected))
    methodology = root / "results/phase15/methodology/evaluation_protocol.json"
    methodology.parent.mkdir(parents=True)
    methodology.write_text("{}")
    metadata = root / "results/phase15/hard_negatives/diagnostic_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"usage": "diagnostic_only", "allowed_for_training": False, "source_split": "validation_or_test"}))


@pytest.mark.parametrize(
    "missing_path",
    [
        "results/phase15/complementarity/classification_pairwise.csv",
        "results/phase15/expert_selection/selected_experts.yaml",
        "results/phase15/prediction_validation/coverage_report.csv",
    ],
)
def test_readiness_gate_required_missing_artifacts_are_not_ready(tmp_path, monkeypatch, missing_path):
    _complete_readiness_fixture(tmp_path)
    (tmp_path / missing_path).unlink()
    monkeypatch.setattr("src.phase15.phase2_readiness._training_negative_check", lambda *_args: (True, "valid", True, "no leakage"))
    assert run_phase2_readiness(tmp_path, protected_ids=set())["status"] == "NOT_READY"


def test_readiness_gate_leakage_is_not_ready_and_complete_state_is_ready(tmp_path, monkeypatch):
    _complete_readiness_fixture(tmp_path)
    monkeypatch.setattr("src.phase15.phase2_readiness._training_negative_check", lambda *_args: (False, "invalid", False, "protected test overlap"))
    assert run_phase2_readiness(tmp_path, protected_ids=set())["status"] == "NOT_READY"
    monkeypatch.setattr("src.phase15.phase2_readiness._training_negative_check", lambda *_args: (True, "valid", True, "zero overlap"))
    assert run_phase2_readiness(tmp_path, protected_ids=set())["status"] == "READY"
