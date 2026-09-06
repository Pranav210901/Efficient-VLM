from __future__ import annotations

import torch
from torch import nn

from freezeshift.model_adaptation import LoRALinear, lora_parameter_count, merge_lora_
from freezeshift.runner import PIPELINES, _config
from src.alignment_v3.resolution_distillation_trajectory import _unique_peak

import pandas as pd
import pytest


def test_merge_is_exact_in_eval_without_dropout() -> None:
    torch.manual_seed(7)
    base = nn.Linear(11, 13)
    wrapped = LoRALinear(base, rank=4, alpha=8, dropout=0.0).eval()
    nn.init.normal_(wrapped.lora_b.weight)
    model = nn.Sequential(wrapped)
    values = torch.randn(5, 11)
    expected = model(values)
    names = merge_lora_(model)
    observed = model(values)
    assert names == ["0"]
    assert torch.allclose(expected, observed, atol=1e-6, rtol=1e-6)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_lora_counter_excludes_base() -> None:
    wrapped = LoRALinear(nn.Linear(8, 12), rank=3, alpha=6, dropout=0.0)
    assert lora_parameter_count(wrapped) == 3 * (8 + 12)


def test_runtime_parameter_contracts_match_declared_counts() -> None:
    for path in PIPELINES:
        pipeline, config = _config(path)
        contract = config["provenance"]["parameter_contract"]
        observed = contract["params_trainable_inference"]
        expected = pipeline["freezeshift"][
            "expected_inference_trainable_parameters"
        ]
        assert int(observed) == int(expected)
        assert set(contract) == {
            "params_trainable_inference",
            "params_trainable_training",
            "params_training_only",
        }


def test_peak_ties_remain_strict_without_an_explicit_policy() -> None:
    frame = pd.DataFrame({"epoch": [23, 24], "score": [0.631, 0.631]})
    with pytest.raises(RuntimeError, match="no tie-break was pre-registered"):
        _unique_peak(frame, "score", 43)


def test_earliest_epoch_tie_break_is_deterministic_and_audited() -> None:
    frame = pd.DataFrame({"epoch": [24, 23], "score": [0.631, 0.631]})
    events: list[dict[str, object]] = []
    selected = _unique_peak(
        frame,
        "score",
        43,
        tie_break="earliest_epoch",
        tie_events=events,
    )
    assert int(selected["epoch"]) == 23
    assert events == [
        {
            "seed": 43,
            "metric": "score",
            "maximum": 0.631,
            "tied_epochs": [24, 23],
            "selected_epoch": 23,
            "rule": "earliest_epoch",
        }
    ]


def test_freezeshift_reports_use_the_matched_m_t1_baseline() -> None:
    for path in PIPELINES:
        pipeline, _ = _config(path)
        reporting = pipeline["reporting"]
        assert reporting["peak_tie_break"] == "earliest_epoch"
        assert reporting["tie_break_status"] == "post_observation_reporting_amendment"
        assert reporting["baseline_to_beat"] == {
            "flickr_validation_mean_R1": 0.5448717921972275,
            "full_stack_latency_ms": 8.99804092478007,
        }
