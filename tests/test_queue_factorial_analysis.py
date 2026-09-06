import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_queue_factorial", ROOT / "scripts/analyze_queue_factorial.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def test_residence_window_path_length_prorates_boundary_interval():
    drift = pd.DataFrame(
        {
            "optimizer_step": [10, 20, 30],
            "image_from_previous": [0.1, 0.2, 0.3],
        }
    )
    value, windows = ANALYSIS.residence_window_path_length(
        drift, "image", residence_steps=15, first_full_step=30
    )
    # Window [15, 30]: half of the second interval plus all of the third.
    assert value == pytest.approx(0.4)
    assert windows == 1


def test_interpretation_outputs_are_complete_and_predictions_remain_separate():
    output = ROOT / "results/queue_factorial/interpretation"
    summary = ANALYSIS.read_json(output / "interpretation_summary.json")
    scores = ANALYSIS.read_json(output / "prediction_scores.json")
    drift = pd.read_csv(output / "drift_at_eviction.csv")

    assert summary["status"] == "COMPLETE"
    assert summary["primary_equivalent_ages"] == 6
    assert summary["primary_total_ages"] == 8
    assert summary["drift_curve_collapse"] == "DOES_NOT_COLLAPSE"
    assert summary["matched_drift_claim"] == "GENERAL"
    assert len(drift) == 18
    assert set(drift["seed"]) == {42, 43, 44}
    assert {row["id"]: row["score"] for row in scores}[
        "modality_asymmetry"
    ] == "MISS"
    assert (ROOT / "predictions/queue_factorial.json").is_file()


def test_matched_drift_sensitivity_supports_general_claim():
    drift = pd.read_csv(
        ROOT / "results/queue_factorial/interpretation/drift_at_eviction.csv"
    )
    pairs, assessment = ANALYSIS.matched_drift_sensitivity(drift)
    summaries = {row["tolerance"]: row for row in assessment["tolerance_summaries"]}

    assert assessment["claim_decision"] == "GENERAL"
    assert len(pairs[pairs["tolerance"] == 0.005]) == 2
    assert summaries[0.005]["pairs_exceeding_1pp"] == 2
    assert summaries[0.05]["matched_pair_count"] == 3
    assert summaries[0.05]["pairs_exceeding_1pp"] == 3
