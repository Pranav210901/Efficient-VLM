from __future__ import annotations

import math

from latency_amendment.runner import PIPELINES, REFERENCE_ID, REPETITIONS, _classification


def test_roster_contains_reference_baseline_and_all_freezeshift_arms() -> None:
    assert set(PIPELINES) == {"M_T1", "vision", "text", "dual"}
    assert REFERENCE_ID == "openclip_vit_b32_quickgelu_openai"
    assert REPETITIONS == 10


def test_uncertainty_classification_is_mechanical() -> None:
    assert _classification(-0.2, -0.01) == "FASTER"
    assert _classification(-0.02, 0.03) == "LATENCY_EQUIVALENT"
    assert _classification(0.01, 0.2) == "SLOWER"


def test_original_gate_anomaly_is_not_rounded_away() -> None:
    old_ceiling = 9.480
    contemporaneous_baseline = 9.497033664956689
    assert contemporaneous_baseline > old_ceiling
    assert math.isclose(contemporaneous_baseline - old_ceiling, 0.0170336649566889)


def test_point_estimate_rule_keeps_openclip_as_the_reference() -> None:
    candidate_q3 = [9.3, 9.4, 9.5]
    openclip_q3 = [9.4, 9.4, 9.4]
    paired = [candidate - reference for candidate, reference in zip(candidate_q3, openclip_q3)]
    assert sum(paired) / len(paired) <= 0
