from __future__ import annotations

import tempfile
from math import log
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.multitask import runner as multitask_runner
from src.multitask import qualitative_retrieval
from src.multitask.checkpoint_selection import (
    CheckpointSpec,
    infer_vision_backend,
    load_frozen_model,
    parse_checkpoint_dir,
    remap_legacy_state_dict,
)
from src.multitask.classification_evaluator import (
    classification_metrics,
    classification_predictions,
    evaluate_zeroshot_classification,
)
from src.multitask.compositional_evaluator import sugarcrepe_scores, winoground_scores
from src.multitask.datasets import load_zeroshot_dataset, validate_datasets, zeroshot_subset_indices
from src.multitask.embedding_cache import load_cache, save_cache
from src.multitask.prompt_templates import class_prompts, clean_class_name, ensemble_class_embeddings
from src.multitask.qualitative_retrieval import QualitativeModel
from src.multitask.retrieval_evaluator import text_to_image_predictions
from src.multitask.task_registry import get_task


def test_task_registry_and_checkpoint_parsing():
    assert get_task("coco_retrieval").group == "retrieval"
    assert parse_checkpoint_dir("dinov2_vits14_all_minilm_l6_v2_local_global", ["dinov2_vits14"]) == ("dinov2_vits14", "all_minilm_l6_v2", "local_global")


def test_legacy_timm_checkpoint_keys_are_remapped_only_when_expected():
    legacy = {
        "vision_encoder.encoder.stem.weight": torch.ones(1),
        "image_projection.linear.weight": torch.ones(1),
    }
    expected = {"vision_encoder.encoder.model.stem.weight", "image_projection.linear.weight"}
    remapped = remap_legacy_state_dict(legacy, expected)
    assert set(remapped) == expected


def test_checkpoint_vision_backend_is_inferred_from_saved_module_keys():
    hf = {"vision_encoder.encoder.model.embeddings.cls_token": torch.ones(1)}
    timm = {"vision_encoder.encoder.model.patch_embed.proj.weight": torch.ones(1)}
    assert infer_vision_backend(hf) == "hf"
    assert infer_vision_backend(timm) == "timm"


class TinyCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)


def test_checkpoint_loader_uses_saved_weights_without_pretrained_downloads(tmp_path, monkeypatch):
    original = TinyCheckpointModel()
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": {"model": {}}, "model_state": original.state_dict(), "epoch": 4}, checkpoint)
    captured = {}

    def build(config):
        captured.update(config["model"])
        model = TinyCheckpointModel()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    monkeypatch.setattr("src.training.train.build_model_from_config", build)
    spec = CheckpointSpec("efficientnet_b0", "all_minilm_l6_v2", "baseline", checkpoint, 0)
    model, _, epoch = load_frozen_model(spec, "cpu")
    assert epoch == 4
    assert captured["pretrained_vision"] is False
    assert captured["pretrained_text"] is False
    assert captured["vision_backend"] == "auto"
    assert not any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())


def test_dataset_validation_and_cache_roundtrip():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root / "data").mkdir(); (root / "data/val_all_captions.csv").write_text("image_path,caption\na.jpg,hello\n")
        assert validate_datasets(root, ["coco_retrieval"])[0]["status"] == "ready"
        path = root / "cache.pt"; save_cache(path, {"tensor": torch.eye(2)}, {"version": 1})
        assert torch.equal(load_cache(path, {"version": 1})["tensor"], torch.eye(2))
        assert load_cache(path, {"version": 2}) is None


class FakeTextModel:
    def encode_text(self, prompts):
        return torch.tensor([[1.0, float(index + 1)] for index, _ in enumerate(prompts)])


def test_prompt_ensembling():
    assert len(class_prompts("tabby_cat")) == 3
    assert clean_class_name("HerbaceousVegetation") == "herbaceous vegetation"
    assert clean_class_name("Sea-Lake") == "sea lake"
    values = ensemble_class_embeddings(FakeTextModel(), ["cat", "dog"])
    assert values.shape == (2, 2)
    assert torch.allclose(values.norm(dim=1), torch.ones(2))


def test_zero_shot_metrics_top1_top5_macro_f1():
    logits = torch.tensor([[9.0, 1.0, 0.0], [0.0, 2.0, 3.0], [0.0, 5.0, 1.0]])
    metrics = classification_metrics(logits, torch.tensor([0, 2, 1]))
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_classification_margin_and_prediction_export_are_signed():
    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0]])
    targets = torch.tensor([0, 0])
    metrics = classification_metrics(logits, targets)
    rows = classification_predictions(logits, targets, ["bad", "good"], ["cat", "dog"], "tiny")
    assert abs(metrics["classification_margin"]) < 1e-6
    assert rows[0]["classification_margin"] < 0
    assert rows[0]["highest_incorrect"] == "dog"
    assert rows[1]["correct"] is True
    assert len(rows[0]["top5"]) == 2


class TinyClassificationDataset(Dataset):
    def __init__(self):
        self.rows = [
            (torch.tensor([1.0, 0.0]), 0, "sample-0"),
            (torch.tensor([0.0, 1.0]), 1, "sample-1"),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TinyZeroShotModel:
    def __init__(self):
        self.logit_scale = torch.tensor(log(10.0))

    def eval(self):
        return self

    def encode_image(self, images):
        return images.float()

    def encode_text(self, prompts):
        return torch.tensor([[1.0, 0.0] if "cat" in value else [0.0, 1.0] for value in prompts])


def test_zero_shot_evaluator_end_to_end_with_frozen_style_model():
    result = evaluate_zeroshot_classification(
        TinyZeroShotModel(),
        TinyClassificationDataset(),
        ["cat", "dog"],
        device="cpu",
        task="tiny",
        batch_size=2,
    )
    assert result["metrics"]["top1_accuracy"] == 1.0
    assert result["sample_ids"] == ["sample-0", "sample-1"]
    assert result["logits"].shape == (2, 2)


def test_torchvision_loader_is_explicitly_offline():
    sentinel = object()
    with patch("torchvision.datasets.CIFAR100", return_value=sentinel) as constructor:
        assert load_zeroshot_dataset("/tmp/project", "cifar100_zeroshot") is sentinel
    assert constructor.call_args.kwargs["download"] is False
    assert constructor.call_args.kwargs["train"] is False


def test_development_partition_is_deterministic_stratified_and_uses_train_data(monkeypatch, tmp_path):
    class TinyDataset:
        targets = [label for label in range(4) for _ in range(10)]

    first = zeroshot_subset_indices(tmp_path, "cifar100_zeroshot", TinyDataset(), "development")
    second = zeroshot_subset_indices(tmp_path, "cifar100_zeroshot", TinyDataset(), "development")
    assert first == second
    assert len(first) == 8
    assert {TinyDataset.targets[index] for index in first} == {0, 1, 2, 3}
    sentinel = object()
    with patch("torchvision.datasets.CIFAR100", return_value=sentinel) as constructor:
        assert load_zeroshot_dataset(tmp_path, "cifar100_zeroshot", split_role="development") is sentinel
    assert constructor.call_args.kwargs["train"] is True


def test_compositional_registry_tasks_are_optional():
    assert get_task("winoground").mandatory is False
    assert get_task("sugarcrepe").mandatory is False


def test_runner_writes_and_reuses_zero_shot_cache(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/tiny/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-marker")
    spec = CheckpointSpec("efficientnet_b0", "all_minilm_l6_v2", "baseline", checkpoint, 0)
    status = {
        "task": "cifar100_zeroshot",
        "group": "classification",
        "status": "ready",
        "samples": 2,
        "classes_or_categories": 2,
        "optional": False,
        "path": str(tmp_path / "data/multitask/cifar100"),
        "detail": "test fixture",
    }
    monkeypatch.setattr(multitask_runner, "resolve_best_checkpoints", lambda *args, **kwargs: [spec])
    monkeypatch.setattr(multitask_runner, "validate_datasets", lambda *args, **kwargs: [status])
    monkeypatch.setattr(
        multitask_runner,
        "load_frozen_model",
        lambda *args, **kwargs: (TinyZeroShotModel(), {"data": {"image_size": 2, "num_workers": 0}, "training": {"batch_size": 2}}, 7),
    )
    monkeypatch.setattr(
        multitask_runner,
        "prepare_zeroshot_dataset",
        lambda *args, **kwargs: (TinyClassificationDataset(), ["cat", "dog"]),
    )

    output = tmp_path / "results/phase1_multitask"
    first = multitask_runner.run_multitask_evaluation(
        tmp_path,
        ["cifar100_zeroshot"],
        mode="smoke",
        device="cpu",
        config_ids=[spec.config_id],
        output_root_override=output,
    )
    assert first["manifest"].iloc[0].status == "complete"
    assert first["manifest"].iloc[0]["mode"] == "smoke"
    assert first["results"].iloc[0]["mode"] == "smoke"
    assert (output / "predictions" / f"{spec.config_id}__cifar100_zeroshot__smoke.jsonl").exists()

    monkeypatch.setattr(
        multitask_runner,
        "load_frozen_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = multitask_runner.run_multitask_evaluation(
        tmp_path,
        ["cifar100_zeroshot"],
        mode="smoke",
        device="cpu",
        config_ids=[spec.config_id],
        output_root_override=output,
    )
    assert second["results"].query("metric == 'top1_accuracy'").iloc[0].value == 1.0


def test_text_to_image_predictions_rank_the_correct_gallery_image():
    rows = text_to_image_predictions(
        torch.eye(3),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ["a.jpg", "b.jpg", "c.jpg"],
        ["b.jpg", "a.jpg"],
        topk=2,
    )
    assert [row["correct_target_rank"] for row in rows] == [1, 1]
    assert rows[0]["top_candidate_ids"][0] == "b.jpg"


def test_qualitative_retrieval_uses_shared_queries_and_cache(tmp_path, monkeypatch):
    data = tmp_path / "data"
    image_dir = data / "coco/val2017"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "one.jpg"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    (data / "val_all_captions.csv").write_text(
        "image_path,caption\n"
        "data/coco/val2017/one.jpg,a red square\n"
    )
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"a")
    checkpoint_b.write_bytes(b"b")
    models = [
        QualitativeModel("model a", "efficientnet_b0", "all_minilm_l6_v2", "local", checkpoint_a),
        QualitativeModel("model b", "convnextv2_tiny", "all_minilm_l6_v2", "local_global", checkpoint_b),
    ]
    monkeypatch.setattr(
        qualitative_retrieval,
        "load_frozen_model",
        lambda *args, **kwargs: (object(), {"data": {"image_size": 8, "num_workers": 0}, "training": {"batch_size": 1}}, 3),
    )
    monkeypatch.setattr(
        qualitative_retrieval,
        "_encode_gallery",
        lambda *args, **kwargs: (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])),
    )
    first = qualitative_retrieval.run_qualitative_retrieval(
        tmp_path,
        models,
        device="cpu",
        num_queries=1,
        topk=1,
    )
    assert len(first["rows"]) == 2
    assert first["summary"]["R@1"].tolist() == [1.0, 1.0]

    monkeypatch.setattr(
        qualitative_retrieval,
        "load_frozen_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = qualitative_retrieval.run_qualitative_retrieval(
        tmp_path,
        models,
        device="cpu",
        num_queries=1,
        topk=1,
    )
    assert len(second["rows"]) == 2


def test_official_compositional_scores():
    scores = winoground_scores(torch.tensor([[4.0, 1.0, 1.0, 4.0], [1.0, 2.0, 2.0, 1.0]]))
    assert scores == {"text_score": 0.5, "image_score": 0.5, "group_score": 0.5}
    sugar = sugarcrepe_scores(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0]), ["swap", "swap"])
    assert sugar["overall_accuracy"] == 0.5
