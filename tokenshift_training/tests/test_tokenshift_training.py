from pathlib import Path

from freezeshift.model_adaptation import (
    build_model as build_parent_model,
    lora_parameter_count,
)
from src.alignment_v3 import runner as alignment_runner
from tokenshift_training.model import build_model


ROOT = Path(__file__).resolve().parents[2]
PIPELINES = (
    "tokenshift_training/configs/pipeline_frozen_block8.yaml",
    "tokenshift_training/configs/pipeline_frozen_block6.yaml",
    "tokenshift_training/configs/pipeline_dual_block8.yaml",
    "tokenshift_training/configs/pipeline_dual_block6.yaml",
)


def _config(path: str):
    _, pipeline = alignment_runner.load_pipeline(path)
    jobs = alignment_runner.sensitivity_jobs(pipeline)
    assert [job["seed"] for job in jobs] == [42, 43, 44]
    return pipeline, alignment_runner.build_job_config(
        pipeline, jobs[0], "sensitivity"
    )


def test_four_unique_three_seed_pipelines_and_sealed_test():
    roots = set()
    for path in PIPELINES:
        pipeline, config = _config(path)
        roots.add(pipeline["checkpoint_root"])
        assert "test" not in pipeline["optional_transfer"]["flickr30k_csv"].lower()
        assert config["training"]["epochs"] == 24
        assert config["training"]["save_every_epoch"] is True
        assert config["training"]["disable_early_stopping"] is True
        assert config["recipe"]["tokenshift_merge_after_block"] in {6, 8}
    assert len(roots) == 4


def test_tokenshift_is_state_dict_and_parameter_neutral_for_both_parents():
    for path, expected_lora, expected_trainable in (
        (PIPELINES[0], 0, 2_896_389),
        (PIPELINES[2], 1_966_080, 4_862_469),
    ):
        _, config = _config(path)
        parent = build_parent_model(config, pretrained=False)
        model = build_model(config, pretrained=False)
        assert tuple(parent.state_dict()) == tuple(model.state_dict())
        assert sum(p.numel() for p in parent.parameters()) == sum(
            p.numel() for p in model.parameters()
        )
        assert lora_parameter_count(model) == expected_lora
        assert model.parameter_summary()["params_trainable_inference"] == expected_trainable


def test_runtime_controller_matches_resolved_config():
    for path in PIPELINES:
        pipeline, config = _config(path)
        model = build_model(config, pretrained=False)
        controller = model.vision_encoder.model._tokenshift_controller
        assert controller.merge_after_block == pipeline["tokenshift_training"][
            "merge_after_block"
        ]

