from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.alignment_v3.queue_factorial import (
    build_config,
    canonical_jobs,
    first_cell_job,
    gate_verdict,
    load_pipeline,
    prediction_gate,
    remaining_jobs,
)
from src.training.losses import CrossBatchMemory, contrastive_loss_with_memory


PIPELINE = "configs/queue_factorial/pipeline.yaml"


class QueueFactorialTests(unittest.TestCase):
    def test_manifest_has_66_unique_cells_and_expected_overlap(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        jobs = canonical_jobs(pipeline)
        self.assertEqual(len(jobs), 66)
        self.assertEqual(len({job["run_id"] for job in jobs}), 66)
        age_zero = [
            job
            for job in jobs
            if job["batch_size"] == 1024 and job["target_age_steps"] == 0
        ]
        self.assertEqual(len(age_zero), 3)
        self.assertTrue(
            all(
                set(job["memberships"])
                == {"grid1_matched_age", "grid2_modality", "control_batch"}
                for job in age_zero
            )
        )

    def test_lr_is_pinned_for_both_batches_and_counterfactual_is_logged(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        jobs = canonical_jobs(pipeline)
        configs = {
            batch: build_config(
                pipeline,
                next(
                    job
                    for job in jobs
                    if job["batch_size"] == batch
                    and job["target_age_steps"] == 0
                    and job["seed"] == 42
                ),
            )
            for batch in (512, 1024)
        }
        self.assertEqual(configs[512]["training"]["lr"], configs[1024]["training"]["lr"])
        self.assertEqual(configs[512]["training"]["counterfactual_sqrt_lr"], 0.0018)
        self.assertEqual(
            configs[1024]["training"]["counterfactual_sqrt_lr"],
            configs[1024]["training"]["lr"],
        )

    def test_directional_components_reproduce_legacy_objective_and_gradients(self) -> None:
        torch.manual_seed(11)
        images = torch.nn.functional.normalize(torch.randn(3, 7), dim=-1).requires_grad_()
        texts = torch.nn.functional.normalize(torch.randn(6, 7), dim=-1).requires_grad_()
        scale = torch.tensor(9.0, requires_grad=True)
        image_ids = ["a", "b", "c"]
        text_ids = ["a", "a", "b", "b", "c", "c"]
        legacy = contrastive_loss_with_memory(
            images, texts, scale, image_ids, text_ids
        )
        detailed = contrastive_loss_with_memory(
            images,
            texts,
            scale,
            image_ids,
            text_ids,
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(legacy, detailed.total))
        self.assertTrue(torch.equal(detailed.total, 0.5 * (detailed.i2t + detailed.t2i)))
        legacy_grads = torch.autograd.grad(legacy, (images, texts, scale), retain_graph=True)
        detailed_grads = torch.autograd.grad(detailed.total, (images, texts, scale))
        for left, right in zip(legacy_grads, detailed_grads):
            self.assertTrue(torch.equal(left, right))

    def test_modality_flags_and_age_one_first_reuse(self) -> None:
        memory = CrossBatchMemory(4, enable_image=True, enable_text=False)
        memory.enqueue(
            torch.randn(2, 3),
            torch.randn(4, 3),
            ["a", "b"],
            ["a", "a", "b", "b"],
            optimizer_step=7,
        )
        self.assertIsNotNone(memory.image_embeddings)
        self.assertIsNone(memory.text_embeddings)
        diagnostics = memory.diagnostics(8)
        self.assertEqual(diagnostics["image_age"]["mean"], 1)
        self.assertIsNone(diagnostics["text_age"])

    def test_fifo_eviction_keeps_metadata_aligned(self) -> None:
        memory = CrossBatchMemory(2)
        for step, value in enumerate(("a", "b", "c"), start=1):
            embedding = torch.tensor([[float(step), 0.0]])
            memory.enqueue(embedding, embedding, [value], [value], optimizer_step=step)
        self.assertEqual(memory.image_ids, ["b", "c"])
        self.assertEqual(memory.image_insertion_steps, [2, 3])

    def test_predictions_are_frozen_and_nonempty(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        payload = prediction_gate(pipeline)
        self.assertEqual(payload["wave"], "queue_factorial")
        self.assertEqual(len(payload["unseen_predictions"]), 4)
        self.assertTrue(payload["written_at"].endswith("Z"))
        self.assertTrue(
            all(row["statement"].strip() for row in payload["unseen_predictions"])
        )

    def test_no_fp16_and_resources_are_approved(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        self.assertEqual(pipeline["resources"]["precision"], "bf16")
        self.assertEqual(pipeline["resources"]["cpus_per_task"], 12)
        self.assertEqual(pipeline["resources"]["host_memory"], "96G")
        self.assertEqual(pipeline["resources"]["wall_time"], "08:00:00")
        self.assertEqual(pipeline["resources"]["dataloader_workers"], 12)

    def test_first_cell_and_remaining_grid_are_exact(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        first = first_cell_job(pipeline)
        self.assertEqual(
            (
                first["batch_size"],
                first["memory_queue_size"],
                first["queue_mode"],
                first["seed"],
            ),
            (1024, 65536, "both", 42),
        )
        remaining = remaining_jobs(pipeline)
        self.assertEqual(len(remaining), 62)
        self.assertNotIn(first["run_id"], {job["run_id"] for job in remaining})

    def test_independent_three_way_gate_boundaries(self) -> None:
        self.assertEqual(gate_verdict(0.19, 0.19), "PASS")
        self.assertEqual(gate_verdict(0.19, 0.20), "PARTIAL")
        self.assertEqual(gate_verdict(0.20, 0.19), "PARTIAL")
        self.assertEqual(gate_verdict(0.20, 0.20), "FAIL")


if __name__ == "__main__":
    unittest.main()
