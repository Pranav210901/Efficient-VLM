from __future__ import annotations

import torch

from src.alignment_v3.data_scale_pilot import (
    build_config,
    canonical_jobs,
    load_pipeline,
    prepare_flickr_validation,
)
from src.alignment_v3.efficiency_frontier import _token_shape_and_utilization


def test_pilot_has_seven_unique_cells_and_shared_full_data_control() -> None:
    pipeline = load_pipeline("configs/data_scale_pilot/pipeline.yaml")
    jobs = canonical_jobs(pipeline)

    assert len(jobs) == 7
    assert len({job["run_id"] for job in jobs}) == 7
    full = [job for job in jobs if job["percentage"] == 100]
    assert len(full) == 1
    assert full[0]["memberships"] == ["practical_scaling", "fixed_update"]


def test_sealed_flickr_manifest_is_reused_without_dataset_rebuild() -> None:
    pipeline = load_pipeline("configs/data_scale_pilot/pipeline.yaml")
    result = prepare_flickr_validation(pipeline)
    assert result["prepare_action"] == "REUSED_VERIFIED"
    assert result["test_overlap_images"] == 0
    assert result["test_sealed"] is True


def test_fixed_update_cells_share_exact_step_budget_and_locked_lr() -> None:
    pipeline = load_pipeline("configs/data_scale_pilot/pipeline.yaml")
    jobs = canonical_jobs(pipeline)
    configs = [(job, build_config(pipeline, job)) for job in jobs]

    practical_full = next(
        config
        for job, config in configs
        if job["percentage"] == 100 and job["schedule"] == "practical_12_epochs"
    )
    full_steps = (
        practical_full["provenance"]["data_scale_pilot"]["actual_schedule_steps"]
    )
    locked_lr = practical_full["training"]["lr"]

    for job, config in configs:
        assert config["training"]["lr"] == locked_lr
        assert config["training"]["select_on_dev"] is False
        assert config["data"]["val_csv"] != pipeline["split"]["flickr_test_csv"]
        if "fixed_update" in job["memberships"]:
            assert (
                config["provenance"]["data_scale_pilot"][
                    "fixed_update_target_steps"
                ]
                == full_steps
            )
            if job["percentage"] != 100:
                assert config["training"]["max_optimizer_steps"] == full_steps


def test_token_accounting_distinguishes_dynamic_and_fixed_padding() -> None:
    dynamic = {
        "input_ids": torch.ones((3, 8), dtype=torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
    }
    dynamic_stats = _token_shape_and_utilization(dynamic)
    assert dynamic_stats["padded_sequence_length"] == 8
    assert dynamic_stats["raw_token_length_mean"] == 16 / 3
    assert dynamic_stats["tokenizer_output_fixed_length"] is False

    fixed = torch.ones((3, 77), dtype=torch.long)
    fixed_stats = _token_shape_and_utilization(fixed)
    assert fixed_stats["padded_sequence_length"] == 77
    assert fixed_stats["raw_token_length_mean"] == 77
    assert fixed_stats["tokenizer_output_fixed_length"] is True
