from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from src.phase15.complementarity import classification_complementarity, compositional_complementarity, retrieval_complementarity
from src.phase15 import efficiency_profile
from src.phase15.efficiency_profile import profile_callable
from src.phase15.expert_selection import rank_experts, run_expert_selection
from src.phase15.hard_negative_mining import (
    enriched_classification_hard_negatives,
    mine_bidirectional_retrieval,
    mine_retrieval_hard_negatives,
    select_reliable_configs,
)
from src.phase15.oracle_analysis import oracle_accuracy, oracle_recall
from src.phase15.runner import full_retrieval_prediction_files, run_phase15
from src.multitask.checkpoint_selection import CheckpointSpec


def test_complementarity_and_oracles():
    result = retrieval_complementarity([1, 20, 1], [20, 1, 1])
    assert result["only_a_successful"] == 1 and result["only_b_successful"] == 1
    assert classification_complementarity([True, False], [False, True])["oracle_accuracy"] == 1.0
    assert compositional_complementarity([True], [True])["both_successful"] == 1
    assert oracle_recall([[1, 20], [20, 5]])["oracle_R@5"] == 1.0
    assert oracle_accuracy([[True, False], [False, True]]) == 1.0


def test_hard_negatives_exclude_same_image_false_negatives():
    similarity = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
    rows = mine_retrieval_hard_negatives(similarity, ["a", "b"], ["a", "a", "b"], count=2)
    assert all(row["negative_owner_id"] != row["query_id"] for row in rows)


def test_efficiency_flop_fallback():
    def broken(): raise RuntimeError("unsupported")
    result = profile_callable(lambda: 1, repeats=1, warmup=0, flop_counter=broken)
    assert result["flops"] == "unavailable" and "unsupported" in result["flop_error"]


def test_expert_selection_is_deterministic():
    frame = pd.DataFrame({"expert": ["b", "a"], "performance_rank": [1, 1], "unique_wins": [2, 2], "latency_ms": [3, 3], "memory_mb": [4, 4]})
    first = rank_experts(frame, "expert"); second = rank_experts(frame.sample(frac=1, random_state=3), "expert")
    assert first.expert.tolist() == second.expert.tolist() == ["a", "b"]


def test_phase15_uses_manifest_full_predictions_across_filename_versions(tmp_path):
    result_root = tmp_path / "results/phase1_multitask"
    predictions = result_root / "predictions"
    predictions.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "direction": ["i2t", "i2t"],
            "query_id": ["image-a", "image-b"],
            "correct_target_rank": [1, 20],
        }
    )
    legacy_full = predictions / "legacy_config__coco_i2t.jsonl"
    named_full = predictions / "named_config__coco_i2t__full.jsonl"
    smoke = predictions / "smoke_config__coco_i2t.jsonl"
    for path in (legacy_full, named_full, smoke):
        rows.to_json(path, orient="records", lines=True)
    pd.DataFrame(
        [
            {
                "config_id": "legacy_config",
                "task": "coco_retrieval",
                "status": "complete",
                "cache": str(result_root / "cache/legacy_config__coco__full.pt"),
                "predictions": str(legacy_full),
            },
            {
                "config_id": "named_config",
                "task": "coco_retrieval",
                "mode": "full",
                "status": "complete",
                "cache": str(result_root / "cache/named_config__coco__full.pt"),
                "predictions": str(named_full),
            },
            {
                "config_id": "smoke_config",
                "task": "coco_retrieval",
                "mode": "smoke",
                "status": "complete",
                "cache": str(result_root / "cache/smoke_config__coco__smoke.pt"),
                "predictions": str(smoke),
            },
        ]
    ).to_csv(result_root / "evaluation_manifest.csv", index=False)

    selected = full_retrieval_prediction_files(tmp_path)
    assert [config_id for config_id, _ in selected] == ["legacy_config", "named_config"]
    summary = run_phase15(tmp_path)
    assert summary["prediction_files"] == 2
    assert summary["config_ids"] == ["legacy_config", "named_config"]
    pairwise = pd.read_csv(tmp_path / "results/phase15/complementarity/retrieval_pairwise.csv")
    assert pairwise[["expert_a", "expert_b"]].iloc[0].tolist() == ["legacy_config", "named_config"]


def test_bidirectional_hard_negatives_exclude_every_valid_owner():
    images = torch.eye(2)
    texts = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    image_ids = ["a.jpg", "b.jpg"]
    owners = ["a.jpg", "a.jpg", "b.jpg", "b.jpg"]
    captions = ["a one", "a two", "b one", "b two"]
    rows = mine_bidirectional_retrieval(images, texts, image_ids, owners, captions, count=1, chunk_size=1)
    assert set(rows.direction) == {"i2t", "t2i"}
    assert (rows.query_id != rows.negative_owner_id).all()
    assert len(rows) == len(image_ids) + len(captions)


def test_enriched_classification_negatives_include_prompt_metadata():
    payload = {
        "logits": torch.tensor([[4.0, 3.0, 2.0], [1.0, 5.0, 4.0]]),
        "targets": torch.tensor([0, 1]),
        "sample_ids": ["a", "b"],
        "class_names": ["cat", "dog", "bird"],
    }
    rows = enriched_classification_hard_negatives(payload, "tiny", count=1)
    assert rows.negative_class.tolist() == ["dog", "bird"]
    assert rows.hard_negative_rank.tolist() == [1, 1]
    assert rows.negative_prompts.str.contains("photo").all()


def test_reliable_config_selection_uses_all_headline_metrics():
    rows = []
    for config_id, value in (("strong", 0.9), ("weak", 0.1)):
        for task, metrics in {
            "coco_retrieval": ["coco5_i2t_R@1", "coco5_t2i_R@1"],
            "cifar100_zeroshot": ["top1_accuracy"],
            "pets_zeroshot": ["top1_accuracy"],
            "eurosat_zeroshot": ["top1_accuracy"],
        }.items():
            for metric in metrics:
                rows.append({"config_id": config_id, "task": task, "metric": metric, "value": value})
    selected = select_reliable_configs(pd.DataFrame(rows), limit=1)
    assert selected.iloc[0].config_id == "strong"
    assert selected.iloc[0].headline_metrics == 5


class _TinyVision(nn.Module):
    output_dim = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, images):
        return images.mean(dim=(2, 3))[:, :2] * self.scale


class _TinyTokenizer:
    def __call__(self, captions, **_kwargs):
        return {"attention_mask": torch.ones(len(captions), 3, dtype=torch.long)}


class _TinyText(nn.Module):
    output_dim = 2
    is_openclip = False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.tokenizer = _TinyTokenizer()

    def _prepare_captions(self, captions):
        return captions

    def forward(self, captions):
        return torch.ones(len(captions), 2) * self.scale


class _TinyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = _TinyVision()
        self.text_encoder = _TinyText()

    def forward(self, images, captions):
        return self.vision_encoder(images) @ self.text_encoder(captions).t()


def test_efficiency_pipeline_writes_and_resumes_all_tables(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/tiny/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    spec = CheckpointSpec("efficientnet_b0", "all_minilm_l6_v2", "baseline", checkpoint, 1)
    monkeypatch.setattr(efficiency_profile, "resolve_best_checkpoints", lambda *args, **kwargs: [spec])
    calls = {"count": 0}

    def load(*_args, **_kwargs):
        calls["count"] += 1
        return _TinyPair(), {"data": {"image_size": 4}, "model": {"shared_dim": 2}}, 1

    monkeypatch.setattr(efficiency_profile, "load_frozen_model", load)
    summary = efficiency_profile.profile_phase15_efficiency(
        tmp_path, device="cpu", batch_size=2, warmup=0, repeats=1, measure_flops=False
    )
    assert summary["vision_encoders_complete"] == 1
    assert summary["text_encoders_complete"] == 1
    assert summary["pairs_complete"] == 1
    output = tmp_path / "results/phase15/efficiency"
    assert len(pd.read_csv(output / "vision_encoders.csv")) == 1
    assert len(pd.read_csv(output / "text_encoders.csv")) == 1
    assert len(pd.read_csv(output / "full_pairs.csv")) == 1
    efficiency_profile.profile_phase15_efficiency(
        tmp_path, device="cpu", batch_size=2, warmup=0, repeats=1, measure_flops=False
    )
    assert calls["count"] == 1


def test_efficiency_pipeline_profiles_explicit_blf_variants(tmp_path, monkeypatch):
    baseline_checkpoint = tmp_path / "checkpoints/efficientnet_b0_all_minilm_l6_v2_baseline/best.pt"
    blf_checkpoint = tmp_path / "checkpoints/efficientnet_b0_all_minilm_l6_v2_local/best.pt"
    for checkpoint in (baseline_checkpoint, blf_checkpoint):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    specs = [
        CheckpointSpec("efficientnet_b0", "all_minilm_l6_v2", "baseline", baseline_checkpoint, 1),
        CheckpointSpec("efficientnet_b0", "all_minilm_l6_v2", "local", blf_checkpoint, 1),
    ]
    requested_ids = {spec.config_id for spec in specs}
    captured = {}

    def resolve(_root, _vision_encoders, variants, config_ids=None, **_kwargs):
        captured["variants"] = set(variants)
        captured["config_ids"] = set(config_ids or ())
        return [spec for spec in specs if config_ids is None or spec.config_id in config_ids]

    monkeypatch.setattr(efficiency_profile, "resolve_best_checkpoints", resolve)
    monkeypatch.setattr(
        efficiency_profile,
        "load_frozen_model",
        lambda *_args, **_kwargs: (
            _TinyPair(),
            {"data": {"image_size": 4}, "model": {"shared_dim": 2}},
            1,
        ),
    )
    summary = efficiency_profile.profile_phase15_efficiency(
        tmp_path,
        device="cpu",
        batch_size=2,
        warmup=0,
        repeats=1,
        measure_flops=False,
        config_ids=sorted(requested_ids),
    )
    pairs = pd.read_csv(tmp_path / "results/phase15/efficiency/full_pairs.csv")
    assert captured == {"variants": {"baseline", "local"}, "config_ids": requested_ids}
    assert set(pairs["config_id"]) == requested_ids
    assert set(pairs["variant"]) == {"baseline", "local"}
    assert summary["status"] == "complete"
    assert summary["pairs_complete"] == 2
    assert summary["pairs_expected"] == 2
    assert summary["pairs_total_in_file"] == 2


def test_expert_selection_writes_rankings_and_shortlist(tmp_path):
    phase1 = tmp_path / "results/phase1_multitask"
    phase15 = tmp_path / "results/phase15"
    efficiency = phase15 / "efficiency"
    complementarity = phase15 / "complementarity"
    efficiency.mkdir(parents=True)
    complementarity.mkdir(parents=True)
    configs = [
        ("v1__t1__baseline", "v1", "t1", 0.9),
        ("v1__t2__baseline", "v1", "t2", 0.7),
        ("v2__t1__baseline", "v2", "t1", 0.6),
        ("v2__t2__baseline", "v2", "t2", 0.4),
    ]
    result_rows = []
    for config_id, vision, text, value in configs:
        for task, metrics in {
            "coco_retrieval": ["coco5_i2t_R@1", "coco5_t2i_R@1"],
            "cifar100_zeroshot": ["top1_accuracy"],
            "pets_zeroshot": ["top1_accuracy"],
            "eurosat_zeroshot": ["top1_accuracy"],
        }.items():
            for metric in metrics:
                result_rows.append({"config_id": config_id, "vision_encoder": vision, "text_encoder": text, "variant": "baseline", "mode": "full", "task": task, "metric": metric, "value": value})
    phase1.mkdir(parents=True)
    pd.DataFrame(result_rows).to_csv(phase1 / "task_results_long.csv", index=False)
    pair_rows = []
    for index, left in enumerate(configs):
        for right in configs[index + 1:]:
            pair_rows.append({"expert_a": left[0], "expert_b": right[0], "both_successful": 2, "only_a_successful": 2, "only_b_successful": 1, "both_unsuccessful": 5, "jaccard": 0.4, "oracle_recall": 0.5})
    pd.DataFrame(pair_rows).to_csv(complementarity / "retrieval_pairwise.csv", index=False)
    pd.DataFrame([
        {"config_id": config_id, "vision_encoder": vision, "text_encoder": text, "latency_ms": 10 + index, "peak_allocated_bytes": 100 + index, "parameters": 1000, "device_name": "cpu"}
        for index, (config_id, vision, text, _value) in enumerate(configs)
    ]).to_csv(efficiency / "full_pairs.csv", index=False)
    pd.DataFrame([
        {"vision_encoder": "v1", "architecture_family": "cnn", "latency_ms": 5, "peak_allocated_bytes": 50, "parameters": 500, "output_dim": 2},
        {"vision_encoder": "v2", "architecture_family": "vit", "latency_ms": 6, "peak_allocated_bytes": 60, "parameters": 600, "output_dim": 2},
    ]).to_csv(efficiency / "vision_encoders.csv", index=False)
    pd.DataFrame([
        {"text_encoder": "t1", "architecture_family": "sentence", "latency_ms": 3, "peak_allocated_bytes": 30, "parameters": 300, "output_dim": 2},
        {"text_encoder": "t2", "architecture_family": "masked", "latency_ms": 4, "peak_allocated_bytes": 40, "parameters": 400, "output_dim": 2},
    ]).to_csv(efficiency / "text_encoders.csv", index=False)
    payload = run_expert_selection(tmp_path, vision_limit=2, text_limit=2, pair_limit=4)
    assert len(payload["vision_experts"]) == 2
    assert len(payload["text_experts"]) == 2
    assert len(payload["pair_shortlist"]) == 4
    assert (phase15 / "expert_selection/expert_selection.md").exists()
