from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from src.alignment_v3.runner import (
    _selected_pair,
    build_job_config,
    configured_teacher_id,
    load_pipeline,
    report,
    select_distillation,
    sensitivity_jobs,
)
from src.utils.config import load_config


PIPELINES = (
    "configs/alignment_v4/pipeline_mobileclip2.yaml",
    "configs/alignment_v4/pipeline_siglip2.yaml",
)


class AlignmentV4Tests(unittest.TestCase):
    def test_probe_is_locked_and_expands_to_exactly_four_jobs(self) -> None:
        for path in PIPELINES:
            _, pipeline = load_pipeline(path)
            self.assertEqual(
                _selected_pair(pipeline),
                {
                    "vision_encoder": "dinov3_vits16",
                    "text_encoder": "all_minilm_l6_v2",
                },
            )
            jobs = sensitivity_jobs(pipeline)
            self.assertEqual(len(jobs), 4)
            self.assertEqual({job["seed"] for job in jobs}, {42, 43})
            self.assertEqual(
                {job["experiment_id"] for job in jobs},
                {"matched_baseline", "distill_strength_1p0"},
            )

    def test_teacher_specific_dimensions_and_caches_are_isolated(self) -> None:
        _, mobile = load_pipeline(PIPELINES[0])
        _, siglip = load_pipeline(PIPELINES[1])
        mobile_base = load_config(mobile["base_config"])
        siglip_base = load_config(siglip["base_config"])
        self.assertEqual(configured_teacher_id(mobile), "mobileclip2_s0_dfndr2b")
        self.assertEqual(configured_teacher_id(siglip), "siglip2_vit_b32_256_webli")
        self.assertEqual(mobile_base["recipe"]["teacher_dim"], 512)
        self.assertEqual(siglip_base["recipe"]["teacher_dim"], 768)
        self.assertNotEqual(
            mobile_base["distillation"]["cache_path"],
            siglip_base["distillation"]["cache_path"],
        )

    def test_probe_uses_matched_batch_epoch_and_lr_policy(self) -> None:
        for path in PIPELINES:
            _, pipeline = load_pipeline(path)
            for job in sensitivity_jobs(pipeline):
                config = build_job_config(pipeline, job, "sensitivity")
                self.assertEqual(config["training"]["batch_size"], 512)
                self.assertEqual(config["training"]["epochs"], 12)
                self.assertAlmostEqual(
                    config["training"]["applied_lr_scale"], math.sqrt(512 / 128)
                )
                self.assertAlmostEqual(config["training"]["lr"], 0.0006)

    def test_capture_fraction_and_decision_are_written(self) -> None:
        _, pipeline = load_pipeline(PIPELINES[0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline["output_root"] = str(root / "output")
            pipeline["checkpoint_root"] = str(root / "checkpoints")
            pipeline["log_root"] = str(root / "logs")
            pipeline_path = root / "pipeline.yaml"
            pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False))
            values = {
                ("matched_baseline", 42): 0.20,
                ("matched_baseline", 43): 0.22,
                ("distill_strength_1p0", 42): 0.31,
                ("distill_strength_1p0", 43): 0.33,
            }
            for job in sensitivity_jobs(pipeline):
                destination = (
                    Path(pipeline["output_root"]) / "sensitivity" / job["run_id"]
                )
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_text(
                    json.dumps(
                        {
                            "kind": "sensitivity",
                            "experiment_id": job["experiment_id"],
                            "seed": job["seed"],
                            "mean_R@1": values[(job["experiment_id"], job["seed"])],
                        }
                    )
                )
            selected = select_distillation(pipeline_path)
            self.assertEqual(selected["decision"], "VIABLE_ABSOLUTE_PATH")
            self.assertAlmostEqual(selected["baseline_R@1"], 0.21)
            self.assertAlmostEqual(selected["distilled_R@1"], 0.32)
            expected = (0.32 - 0.21) / (pipeline["probe"]["teacher_r1"] - 0.21)
            self.assertAlmostEqual(selected["capture_fraction"], expected)
            self.assertTrue(
                (Path(pipeline["output_root"]) / "probe_report.json").is_file()
            )
            reported = report(pipeline_path)
            self.assertEqual(reported["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()

