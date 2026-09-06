from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.alignment_v3.runner import (
    build_job_config,
    load_pipeline,
    wave0_jobs,
    wave0_followup_jobs,
    wave0_lr_control_jobs,
    wave0_lr_sweep_jobs,
    wave0_oom_jobs,
    wave0_queue_ablation_jobs,
    wave0_rescreen_confirmation_jobs,
    wave0_rescreen_jobs,
    wave0_winner_confirmation_jobs,
)
from src.training.losses import positive_mask, sigmoid_contrastive_loss


PIPELINE = "configs/alignment_v4_wave0/pipeline.yaml"


class AlignmentV4Wave0Tests(unittest.TestCase):
    def test_sigmoid_loss_matches_hand_computation_and_gradient_routing(self) -> None:
        torch.manual_seed(7)
        images = F.normalize(torch.randn(4, 8), dim=-1)
        texts = F.normalize(torch.randn(8, 8), dim=-1)
        image_ids = [f"i{index}" for index in range(4)]
        text_ids = [f"i{index // 2}" for index in range(8)]
        sigmoid_scale_log = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        sigmoid_bias = torch.nn.Parameter(torch.tensor(-10.0))
        clip_scale_log = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        scale = sigmoid_scale_log.exp()
        actual = sigmoid_contrastive_loss(
            images, texts, scale, sigmoid_bias, image_ids, text_ids
        )
        logits = scale * images @ texts.t() + sigmoid_bias
        mask = positive_mask(image_ids, text_ids, logits.device)
        labels = mask.float() * 2 - 1
        expected = -F.logsigmoid(labels * logits).sum() / 4
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNotNone(sigmoid_scale_log.grad)
        self.assertIsNotNone(sigmoid_bias.grad)
        self.assertIsNone(clip_scale_log.grad)

    def test_wave0_manifests_span_the_preregistered_grid(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        screen = wave0_jobs(pipeline)
        sweep = wave0_lr_sweep_jobs(pipeline)
        smoke = wave0_oom_jobs(pipeline)
        self.assertEqual(len(screen), 18)
        self.assertEqual(len(sweep), 6)
        self.assertEqual(len(smoke), 6)
        combinations = {
            (
                job["loss_type"],
                "all" if job["captions_per_image"] is None else job["captions_per_image"],
                job["batch_size"],
            )
            for job in screen
        }
        self.assertEqual(
            combinations,
            {
                (loss, captions, batch)
                for loss in ("infonce_queue", "infonce_no_queue", "sigmoid")
                for captions in (2, "all")
                for batch in (512, 1024, 2048)
            },
        )

    def test_wave0_lr_audit_fields_and_caption_queue_overrides(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        for job in wave0_jobs(pipeline):
            config = build_job_config(pipeline, job, "wave0-screen")
            batch = int(job["batch_size"])
            raw = math.sqrt(batch / 128)
            expected_scale = min(raw, 3.0)
            self.assertAlmostEqual(
                config["training"]["sqrt_scale_factor"], expected_scale
            )
            self.assertEqual(
                config["training"]["lr_scale_cap_hit"], raw > 3.0
            )
            self.assertEqual(
                config["data"]["train_captions_per_image"],
                job["captions_per_image"],
            )
            expected_queue = 16384 if job["loss_type"] == "infonce_queue" else 0
            self.assertEqual(config["training"]["memory_queue_size"], expected_queue)

    def test_wave0_lr_controls_are_exact_and_resolve_expected_lrs(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        controls = wave0_lr_control_jobs(pipeline)
        self.assertEqual(len(controls), 4)
        self.assertEqual(
            {
                (
                    job["loss_type"],
                    job["captions_per_image"],
                    job["batch_size"],
                    job["sweep_multiplier"],
                    job["seed"],
                )
                for job in controls
            },
            {
                ("infonce_no_queue", None, 1024, 3.0, 42),
                ("infonce_no_queue", None, 1024, 6.0, 42),
                ("sigmoid", None, 1024, 6.0, 42),
                ("sigmoid", None, 1024, 10.0, 42),
            },
        )
        for job in controls:
            config = build_job_config(pipeline, job, "wave0-lr-control")
            self.assertEqual(config["training"]["memory_queue_size"], 0)
            self.assertAlmostEqual(
                config["training"]["lr"],
                0.0003 * math.sqrt(1024 / 128) * job["sweep_multiplier"],
            )
            self.assertEqual(
                config["training"]["sweep_multiplier"], job["sweep_multiplier"]
            )

    def test_followups_are_four_controls_three_queue_doses_and_six_pairs(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        queue_jobs = wave0_queue_ablation_jobs(pipeline)
        rescreen_jobs = wave0_rescreen_jobs(pipeline)
        followups = wave0_followup_jobs(pipeline)
        self.assertEqual(
            [job["memory_queue_size"] for job in queue_jobs],
            [1024, 4096, 8192],
        )
        self.assertEqual(len(rescreen_jobs), 6)
        self.assertEqual(len(followups), 13)
        self.assertEqual(
            {stage for stage, _, _ in followups},
            {"wave0-lr-control", "wave0-queue-ablation", "wave0-rescreen"},
        )
        for job in rescreen_jobs:
            self.assertEqual(job["hardware_profile"], "teaching")
            self.assertEqual(job["precision_mode"], "native_bf16")
            self.assertTrue(job["run_id"].endswith("__hw_teaching_native_bf16"))
            config = build_job_config(pipeline, job, "wave0-rescreen")
            self.assertEqual(config["recipe"]["loss_type"], "infonce_no_queue")
            self.assertIsNone(config["data"]["train_captions_per_image"])
            self.assertEqual(config["training"]["batch_size"], 1024)
            self.assertEqual(config["training"]["memory_queue_size"], 0)
            self.assertEqual(config["training"]["sweep_multiplier"], 3.0)
            self.assertEqual(
                config["model"]["vision_encoder"], job["vision_encoder"]
            )
            self.assertEqual(config["model"]["text_encoder"], job["text_encoder"])

    def test_winner_confirmation_is_exactly_seeds_43_and_44(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        jobs = wave0_winner_confirmation_jobs(pipeline)
        self.assertEqual([job["seed"] for job in jobs], [43, 44])
        for job in jobs:
            config = build_job_config(
                pipeline, job, "wave0-winner-confirmation"
            )
            self.assertEqual(config["recipe"]["loss_type"], "infonce_no_queue")
            self.assertIsNone(config["data"]["train_captions_per_image"])
            self.assertEqual(config["training"]["batch_size"], 1024)
            self.assertEqual(config["training"]["memory_queue_size"], 0)
            self.assertEqual(config["training"]["sweep_multiplier"], 3.0)

    def test_top_three_pair_confirmation_is_exact_approved_grid(self) -> None:
        _, pipeline = load_pipeline(PIPELINE)
        jobs = wave0_rescreen_confirmation_jobs(pipeline)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(
            {
                (job["vision_encoder"], job["text_encoder"], job["seed"])
                for job in jobs
            },
            {
                ("dinov3_vits16", text_encoder, seed)
                for text_encoder in (
                    "e5_small_v2",
                    "bge_small_en",
                    "all_minilm_l6_v2",
                )
                for seed in (43, 44)
            },
        )
        for job in jobs:
            config = build_job_config(
                pipeline, job, "wave0-rescreen-confirmation"
            )
            self.assertEqual(config["recipe"]["loss_type"], "infonce_no_queue")
            self.assertIsNone(config["data"]["train_captions_per_image"])
            self.assertEqual(config["training"]["batch_size"], 1024)
            self.assertEqual(config["training"]["memory_queue_size"], 0)
            self.assertEqual(config["training"]["sweep_multiplier"], 3.0)
            self.assertEqual(job["hardware_profile"], "teaching")
            self.assertEqual(job["precision_mode"], "native_bf16")

    def test_prediction_artifact_has_exact_required_ids_and_nonempty_statements(self) -> None:
        payload = json.loads(Path("predictions/wave0.json").read_text())
        self.assertEqual(payload["wave"], "wave0")
        self.assertTrue(payload["written_before_any_runs"])
        self.assertTrue(payload["written_at"].endswith("Z"))
        values = payload["predictions"]
        self.assertEqual(
            {value["id"] for value in values},
            {"replication_anchor", "wave0_winner", "batch_effect"},
        )
        self.assertTrue(all(value["statement"].strip() for value in values))


if __name__ == "__main__":
    unittest.main()
