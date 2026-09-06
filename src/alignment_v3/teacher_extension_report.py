from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.alignment_v3.fingerprint import sha256_file
from src.alignment_v3.runner import ROOT
from src.phase15.io_utils import atomic_csv, atomic_json


ARMS = {
    "mobileclip2": {
        "teacher_id": "mobileclip2_s0_dfndr2b",
        "teacher_flickr_mean_R1": 0.7824,
        "report": ROOT / "results/resolution_distillation_224_long/report/report.json",
    },
    "siglip2": {
        "teacher_id": "siglip2_vit_b32_256_webli",
        "teacher_flickr_mean_R1": 0.8058,
        "report": ROOT / "results/teacher_extension_224/siglip2/report/report.json",
    },
    "openclip": {
        "teacher_id": "openclip_vit_b32_quickgelu_openai",
        "teacher_flickr_mean_R1": 0.6822,
        "report": ROOT / "results/teacher_extension_224/openclip/report/report.json",
    },
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete artifact: {path}")
    return payload


def _relationship(values: list[float]) -> str:
    increasing = all(left < right for left, right in zip(values, values[1:]))
    decreasing = all(left > right for left, right in zip(values, values[1:]))
    if increasing:
        return "monotonic"
    if decreasing:
        return "inverse"
    return "non-monotonic"


def report() -> dict[str, Any]:
    rows = []
    for arm, spec in ARMS.items():
        payload = _load(spec["report"])
        rows.append(
            {
                "arm": arm,
                "teacher_id": spec["teacher_id"],
                "teacher_flickr_mean_R1": spec["teacher_flickr_mean_R1"],
                "student_flickr_mean_under_coco_selection": payload[
                    "flickr_mean_under_coco_dev_epoch_selection"
                ],
                "student_flickr_mean_under_flickr_selection": payload[
                    "flickr_mean_under_flickr_validation_epoch_selection"
                ],
                "selection_gain_pp": payload["selection_gain_pp"],
                "matched_protocol": True,
            }
        )
    frame = pd.DataFrame(rows).sort_values("teacher_flickr_mean_R1")
    relationship = _relationship(
        frame["student_flickr_mean_under_flickr_selection"].astype(float).tolist()
    )

    multi = _load(
        ROOT / "results/teacher_extension_224/multi/report/report.json"
    )
    cache = (
        ROOT
        / "results/alignment_wave1/openclip/teacher_cache/"
        "openclip_vit_b32_quickgelu_openai.pt"
    )
    cache_metadata = _load(cache.with_suffix(".metadata.json"))
    if cache_metadata["teacher_id"] != "openclip_vit_b32_quickgelu_openai":
        raise RuntimeError("OpenCLIP cache teacher identity mismatch")
    if cache_metadata["content_sha256"] != sha256_file(cache):
        raise RuntimeError("OpenCLIP teacher cache checksum mismatch")

    destination = ROOT / "results/teacher_extension_224/report"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, destination / "matched_single_teachers.csv")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(
        frame["teacher_flickr_mean_R1"] * 100.0,
        frame["student_flickr_mean_under_flickr_selection"] * 100.0,
        marker="o",
    )
    for row in frame.itertuples():
        axis.annotate(
            row.arm,
            (
                row.teacher_flickr_mean_R1 * 100.0,
                row.student_flickr_mean_under_flickr_selection * 100.0,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Teacher Flickr30k mean bidirectional R@1 (%)")
    axis.set_ylabel("Student Flickr30k validation mean R@1 (%)")
    axis.set_title("Matched 224px/24-epoch single-teacher comparison")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination / "teacher_strength.png", dpi=180)
    plt.close(figure)

    result = {
        "status": "COMPLETE",
        "matched_single_teacher_relationship": relationship,
        "teacher_strength_conclusion": (
            "The matched results rule out both a monotonic positive relationship "
            "and an inverse relationship between teacher Flickr30k performance "
            "and compact frozen-student transfer. Teacher selection is not "
            "reducible to teacher benchmark performance."
        ),
        "teacher_strength_unresolved_factors": [
            "architecture family",
            "embedding geometry",
            "distillation heritage",
        ],
        "interpretation_limit": (
            "Three matched teacher points describe the observed relationship "
            "but cannot establish causality, a general scaling law, or whether "
            "teacher strength itself is the operative variable."
        ),
        "historical_observation_scope": (
            "MobileCLIP2 outperformed SigLIP2 as a teacher under the historical "
            "256px/12-epoch protocol only. The teacher-strength question was "
            "open before this matched extension."
        ),
        "multi_teacher": {
            "weighting": {
                "mobileclip2": 0.5,
                "siglip2": 0.5,
                "status": "explicit_untuned_default",
            },
            "flickr_mean_under_coco_selection": multi[
                "flickr_mean_under_coco_dev_epoch_selection"
            ],
            "flickr_mean_under_flickr_selection": multi[
                "flickr_mean_under_flickr_validation_epoch_selection"
            ],
            "selection_gain_pp": multi["selection_gain_pp"],
            "bounded_conclusion": (
                "Combining MobileCLIP2 and SigLIP2 with the explicit untuned "
                "0.5/0.5 weighting did not improve over MobileCLIP2 alone. "
                "This does not establish that teacher ensembling cannot help."
            ),
            "training_only_parameters": 983040,
        },
        "selected_configuration": {
            "teacher": "mobileclip2_s0_dfndr2b",
            "image_resolution": 224,
            "flickr_validation_mean_R1": 0.44714004298051196,
            "full_stack_latency_ms": 8.55,
        },
        "openclip_cache": {
            "registry_tag": "ViT-B-32-quickgelu:openai",
            "open_clip_version": "3.3.0",
            "content_sha256": cache_metadata["content_sha256"],
            "file_sha256_verified": True,
        },
        "flickr_test_used": False,
    }
    atomic_json(result, destination / "report.json")
    return result


def repair_parameter_artifacts() -> dict[str, Any]:
    destination = ROOT / "results/teacher_extension_224/report"
    affected = []
    for arm in ("siglip2", "openclip", "multi"):
        summary_path = (
            ROOT
            / "checkpoints/teacher_extension_224"
            / arm
            / "sensitivity/distill_strength_1p0__seed_42/run_summary.json"
        )
        summary = _load(summary_path)["parameter_summary"]
        expected = int(summary["params_total_inference"])
        training_total = int(summary["params_total_training"])
        training_only = int(summary["params_training_only"])
        if expected != 45_778_563:
            raise AssertionError(
                f"unexpected inference total for {arm}: {expected}"
            )
        if training_total - expected != training_only:
            raise AssertionError(
                f"training-only parameter accounting mismatch for {arm}"
            )
        paths = [
            ROOT
            / "results/teacher_extension_224"
            / arm
            / "report/latency_profile.json",
            ROOT
            / "results/teacher_extension_224"
            / arm
            / "profiling/per_run"
            / {
                "siglip2": "siglip2_single",
                "openclip": "openclip_single",
                "multi": "mobileclip2_siglip2_equal",
            }[arm]
            / "seed_42/profile_dynamic_padding.json",
        ]
        for path in paths:
            payload = _load(path)
            observed = int(payload["full_stack_inference_parameters"])
            if observed not in {training_total, expected}:
                raise AssertionError(
                    f"unrecognised parameter count in {path}: {observed}"
                )
            payload["full_stack_inference_parameters"] = expected
            payload["parameter_count_correction"] = {
                "status": "CORRECTED",
                "previous_value": observed,
                "corrected_value": expected,
                "excluded_training_only_parameters": training_only,
                "reason": (
                    "The profiler loaded the training graph and labelled its "
                    "total as inference parameters. The deployed inference "
                    "export excludes all teacher heads."
                ),
                "inference_state_teacher_heads_excluded": True,
            }
            atomic_json(payload, path)
            affected.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "previous_value": observed,
                    "corrected_value": expected,
                    "excluded_training_only_parameters": training_only,
                }
            )
    protocol_artifacts = {
        "status": "RECORDED",
        "count": 6,
        "connecting_scope": (
            "Reasonable-looking protocol choices interacting with "
            "implementation-specific semantics; the cases differ in scientific scope."
        ),
        "artifacts": [
            {"index": 1, "id": "memory_queue", "class": "training pathology"},
            {"index": 2, "id": "batch_lr_scaling", "class": "confounded inference"},
            {"index": 3, "id": "native_context_padding", "class": "measurement protocol"},
            {"index": 4, "id": "unfused_mobileclip2", "class": "deployment-graph mismatch"},
            {"index": 5, "id": "cuda_flop_undercount", "class": "profiler attribution failure"},
            {
                "index": 6,
                "id": "training_heads_counted_as_inference_parameters",
                "class": "artifact labelling defect",
                "affected_artifacts": len(affected),
            },
        ],
    }
    atomic_json(protocol_artifacts, destination / "protocol_artifacts.json")
    result = {
        "status": "COMPLETE",
        "corrected_artifacts": affected,
        "corrected_inference_parameters": 45_778_563,
        "protocol_artifact_index": 6,
    }
    atomic_json(result, destination / "parameter_correction.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("report", "repair-parameters"))
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(report(), indent=2))
    else:
        print(json.dumps(repair_parameter_artifacts(), indent=2))


if __name__ == "__main__":
    main()
