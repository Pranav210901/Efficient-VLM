from __future__ import annotations

from src.alignment_v3.queue_factorial import load_pipeline
from src.alignment_v3.queue_mechanism import build_config, mechanism_jobs


def test_queue_mechanism_job_partition_and_seed_replication() -> None:
    pipeline = load_pipeline("configs/queue_mechanism/pipeline.yaml")
    jobs = mechanism_jobs(pipeline)
    assert len(jobs) == 15
    assert [job["index"] for job in jobs] == list(range(15))
    assert sum(job["stage"] == "capacity" for job in jobs) == 6
    assert sum(job["stage"] == "mechanism" for job in jobs) == 9
    assert {job["seed"] for job in jobs} == {42, 43, 44}


def test_queue_mechanism_changes_only_preregistered_queue_fields() -> None:
    pipeline = load_pipeline("configs/queue_mechanism/pipeline.yaml")
    jobs = mechanism_jobs(pipeline)
    semantic = next(job for job in jobs if job["arm"] == "semantic_filter")
    random_control = next(job for job in jobs if job["arm"] == "matched_random_filter")
    mass = next(job for job in jobs if job["arm"] == "mass_normalized")
    semantic_config = build_config(pipeline, semantic)
    random_config = build_config(pipeline, random_control)
    mass_config = build_config(pipeline, mass)
    assert semantic_config["training"]["queue_filter_mode"] == "semantic"
    assert random_config["training"]["queue_filter_mode"] == "matched_random"
    assert mass_config["training"]["queue_weight_mode"] == "match_inbatch"
    for config in (semantic_config, random_config, mass_config):
        assert config["training"]["queue_mode"] == "both_fresh"
        assert config["training"]["memory_queue_size"] == 16384
        assert config["training"]["batch_size"] == 1024
        assert config["training"]["lr"] == pipeline["factorial"]["pinned_resolved_lr"]
