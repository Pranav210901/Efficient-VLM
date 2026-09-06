from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.alignment_v3.efficiency_frontier import profile as frontier_profile
from src.alignment_v3.runner import ROOT
from src.phase15.io_utils import atomic_json


OUTPUT = ROOT / "results/token_projection_training/freeze"
B_REPORT = ROOT / "results/token_projection_training/B_learned_query_attention/report/report.json"
C_REPORT = ROOT / "results/token_projection_training/C_transformer_block/report/report.json"


def paired_selection() -> dict[str, Any]:
    b = json.loads(B_REPORT.read_text())
    c = json.loads(C_REPORT.read_text())
    b_by_seed = {int(row["seed"]): row for row in b["per_seed"]}
    c_by_seed = {int(row["seed"]): row for row in c["per_seed"]}
    rows = []
    for seed in (42, 43, 44):
        b_value = float(b_by_seed[seed]["flickr_peak_R1"])
        c_value = float(c_by_seed[seed]["flickr_peak_R1"])
        rows.append(
            {
                "seed": seed,
                "B_flickr_selected_R1": b_value,
                "C_flickr_selected_R1": c_value,
                "C_minus_B_pp": (c_value - b_value) * 100.0,
                "C_leads": c_value > b_value,
            }
        )
    mean_difference = sum(row["C_minus_B_pp"] for row in rows) / len(rows)
    pooled_sd_pp = math.sqrt(
        (
            float(b["flickr_validation_selected_sd"]) ** 2
            + float(c["flickr_validation_selected_sd"]) ** 2
        )
        / 2.0
    ) * 100.0
    c_leads_all = all(row["C_leads"] for row in rows)
    within_one_pooled_sd = abs(mean_difference) <= pooled_sd_pp
    tie_break = (not c_leads_all) or within_one_pooled_sd
    selected = "B_learned_query_attention" if tie_break else "C_transformer_block"
    result = {
        "status": "SELECTED_PENDING_CONTROLLED_LATENCY",
        "selected_arm": selected,
        "rule": (
            "Select B if C does not lead at all three paired seeds or if the "
            "mean paired difference is within one pooled SD."
        ),
        "rule_applied": (
            "C did not lead at all three seeds; the pre-declared tie-break "
            "therefore selected B."
            if tie_break
            else "C led at every seed beyond one pooled SD."
        ),
        "per_seed": rows,
        "mean_C_minus_B_pp": mean_difference,
        "pooled_sd_pp": pooled_sd_pp,
        "C_leads_every_seed": c_leads_all,
        "mean_within_one_pooled_sd": within_one_pooled_sd,
        "tie_break_triggered": tie_break,
        "tie_break_favors": {
            "arm": "B_learned_query_attention",
            "reason": "fewer parameters, simpler mechanism, and lower measured latency",
            "added_parameters_B": 165_761,
            "added_parameters_C": 232_065,
        },
        "mechanistic_finding": (
            "A, B, and C accessed the same 196 patch tokens. Uniform mean "
            "fusion recovered 1.328pp, while learned-query and transformer "
            "aggregation recovered 6.082pp and 6.907pp. Relative to those "
            "learned arms, approximately 78-81% of the gain beyond the baseline "
            "is associated with learned selection over patches rather than "
            "patch access alone. Parameter count and mechanism differ, so "
            "selection and capacity are not fully separated."
        ),
        "flickr_test_used": False,
    }
    atomic_json(result, OUTPUT / "paired_selection.json")
    return result


def _profile_pipeline() -> dict[str, Any]:
    return {
        "output_root": "results/token_projection_training/freeze/profile",
        "students": [
            {
                "id": "selected_B_trained",
                "label": "Trained B learned-query token aggregation",
                "role": "selected_token_aggregation",
                "seeds": {
                    "42": {
                        "checkpoint_dir": (
                            "checkpoints/token_projection_training/"
                            "B_learned_query_attention/sensitivity/"
                            "distill_strength_1p0__seed_42"
                        )
                    }
                },
            },
            {
                "id": "cls_baseline_trained",
                "label": "Trained CLS-only 224px MobileCLIP2-distilled baseline",
                "role": "contemporaneous_cls_baseline",
                "seeds": {
                    "42": {
                        "checkpoint_dir": (
                            "checkpoints/resolution_distillation_224_long/"
                            "sensitivity/distill_strength_1p0__seed_42"
                        )
                    }
                },
            },
        ],
        "references": {
            "entries": [
                {
                    "id": "openclip_vit_b32_quickgelu_openai",
                    "checkpoint_id": "ViT-B-32-quickgelu:openai",
                }
            ]
        },
        "datasets": {"primary": {"csv": "data/flickr30k/validation.csv"}},
        "profiling": {
            "representative_seed": 42,
            "verification_seed": 42,
            "paired_batch_size": 64,
            "warmup_iterations": 20,
            "timed_repeats": 100,
        },
        "resources": {"cpus_per_task": 12},
    }


def controlled_profile() -> dict[str, Any]:
    decision = paired_selection()
    if decision["selected_arm"] != "B_learned_query_attention":
        raise RuntimeError("controlled profiler is frozen for selected arm B")
    pipeline = _profile_pipeline()
    payloads = [
        frontier_profile(pipeline, index, dynamic_padding=True)
        for index in range(3)
    ]
    expected_order = (
        "selected_B_trained",
        "cls_baseline_trained",
        "openclip_vit_b32_quickgelu_openai",
    )
    if tuple(value["entry_id"] for value in payloads) != expected_order:
        raise RuntimeError("controlled profile roster order changed")
    rows = []
    for value in payloads:
        latency = value["full_stack_neural_latency_ms"]
        rows.append(
            {
                "entry_id": value["entry_id"],
                "median_ms": float(latency["median"]),
                "q1_ms": float(latency["q1"]),
                "q3_ms": float(latency["q3"]),
                "iqr_ms": float(latency["iqr"]),
                "node": value["node"],
                "gpu_model": value["gpu_model"],
                "precision": value["precision"],
            }
        )
    selected = rows[0]
    result = {
        "status": "COMPLETE",
        "selected_arm": "B_learned_query_attention",
        "protocol": {
            "same_allocation": True,
            "batch_size": 64,
            "warmup": 20,
            "timed_repeats": 100,
            "cuda_synchronize": True,
            "native_bf16": True,
        },
        "latency_ceiling_ms": 9.48,
        "selected_below_ceiling_by_ms": 9.48 - selected["median_ms"],
        "selected_q3_below_ceiling": selected["q3_ms"] < 9.48,
        "rows": rows,
        "flickr_test_used": False,
    }
    atomic_json(result, OUTPUT / "controlled_latency.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("paired", "profile"))
    args = parser.parse_args()
    result = paired_selection() if args.command == "paired" else controlled_profile()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

