from __future__ import annotations

import math

from src.alignment_v3.mixed_data_training import build_config, jobs, pipeline
from src.data.mixed_coco_cc3m import pass_permutation


def test_frozen_two_arm_arithmetic_and_sealed_test() -> None:
    spec = pipeline("configs/mixed_data_training/pipeline.yaml")
    assert spec["reporting"]["evaluate_flickr_test"] is False
    assert spec["data"]["flickr_test_sealed"] is True
    assert len(jobs(spec)) == 6
    for arm in spec["training"]["arms"]:
        steps = math.ceil(
            int(arm["images_per_pass"]) / int(spec["training"]["batch_size"])
        )
        assert steps == int(arm["optimizer_steps_per_pass"])
        assert int(arm["total_optimizer_steps"]) == 24 * steps


def test_all_jobs_resolve_to_c4_and_locked_recipe() -> None:
    spec = pipeline("configs/mixed_data_training/pipeline.yaml")
    for job in jobs(spec):
        config = build_config(spec, job)
        assert config["recipe"]["image_token_aggregation"] == "transformer_128"
        assert config["recipe"]["token_pooler_dim"] == 256
        assert config["recipe"]["token_pooler_heads"] == 8
        assert config["recipe"]["token_pooler_blocks"] == 2
        assert config["recipe"]["token_pooler_ff_dim"] == 512
        assert config["recipe"]["distillation"] is True
        assert config["training"]["lr"] == 0.002545584412271571
        assert config["training"]["memory_queue_size"] == 0
        assert config["training"]["batch_size"] == 1024
        assert config["data"]["image_size"] == 224


def test_pass_order_is_reproducible_but_seed_and_pass_specific() -> None:
    first = pass_permutation(1000, seed=42, pass_index=0, arm_id="mixed")
    assert (first == pass_permutation(1000, seed=42, pass_index=0, arm_id="mixed")).all()
    assert not (first == pass_permutation(1000, seed=43, pass_index=0, arm_id="mixed")).all()
    assert not (first == pass_permutation(1000, seed=42, pass_index=1, arm_id="mixed")).all()


def test_caption_sampling_unit_is_image() -> None:
    spec = pipeline("configs/mixed_data_training/pipeline.yaml")
    assert spec["data"]["sampling_unit"] == "image"
    assert spec["data"]["coco_caption_policy"] == "all_available_positive_captions_per_image"
    assert spec["data"]["mixture_sampling"].startswith("uniform_without_replacement")


def test_nonround_cursor_resume_preserves_exact_sample_sequence() -> None:
    order = pass_permutation(
        1465, seed=42, pass_index=7, arm_id="mixed_coco_cc3m"
    ).tolist()
    interrupted_at = 737
    uninterrupted = order
    resumed = order[:interrupted_at] + order[interrupted_at:]
    assert resumed == uninterrupted
    assert resumed[interrupted_at] == uninterrupted[interrupted_at]
