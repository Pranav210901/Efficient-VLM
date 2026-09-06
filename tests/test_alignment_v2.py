from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from scripts.run_alignment_v2_notebook_cell import selected_cell_indices
from src.alignment_v2.runner import build_unimodal_config, reference_jobs, unimodal_jobs
from src.data.collate import image_text_collate
from src.models.projection import ResidualProjectionHead
from src.phase2.evaluator import _fused_scores
from src.training.losses import (
    CrossBatchMemory,
    contrastive_loss_with_memory,
    multi_positive_contrastive_loss,
)
from src.utils.config import load_config


class AlignmentV2Tests(unittest.TestCase):
    def test_grouped_collate_reuses_each_image_for_multiple_captions(self) -> None:
        batch = [
            {"image": torch.zeros(3, 4, 4), "caption": ["one", "two"], "image_path": "a.jpg"},
            {"image": torch.ones(3, 4, 4), "caption": ["three"], "image_path": "b.jpg"},
        ]
        result = image_text_collate(batch)
        self.assertEqual(result["images"].shape[0], 2)
        self.assertEqual(result["captions"], ["one", "two", "three"])
        self.assertEqual(result["text_image_paths"], ["a.jpg", "a.jpg", "b.jpg"])

    def test_multi_positive_loss_accepts_rectangular_logits(self) -> None:
        logits = torch.tensor([[5.0, 4.0, -3.0], [-2.0, -1.0, 5.0]])
        loss = multi_positive_contrastive_loss(logits, ["a", "b"], ["a", "a", "b"])
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 0.2)

    def test_cross_batch_memory_adds_detached_negatives(self) -> None:
        memory = CrossBatchMemory(4)
        images = torch.nn.functional.normalize(torch.eye(2), dim=-1)
        texts = images.clone()
        memory.enqueue(images, texts, ["a", "b"], ["a", "b"])
        loss = contrastive_loss_with_memory(images, texts, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(memory.image_embeddings.shape, (2, 2))
        self.assertFalse(memory.image_embeddings.requires_grad)

    def test_residual_projector_preserves_batch_shape(self) -> None:
        projector = ResidualProjectionHead(8, output_dim=4, hidden_dim=16, dropout=0.0)
        self.assertEqual(projector(torch.randn(3, 8)).shape, (3, 4))

    def test_pipeline_expands_three_seeded_unimodal_experiments(self) -> None:
        pipeline = load_config("configs/alignment_v2/pipeline.yaml")
        references = reference_jobs(pipeline)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0]["model"]["name"], "ViT-B-32-quickgelu")
        jobs = unimodal_jobs(pipeline)
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({job["run_id"] for job in jobs}), 9)
        config = build_unimodal_config(jobs[0])
        self.assertTrue(config["model"]["freeze_vision"])
        self.assertTrue(config["model"]["freeze_text"])
        self.assertEqual(config["training"]["selection_metric"], "mean_R@1")
        self.assertEqual(config["training"]["epochs"], 12)
        self.assertEqual(config["data"]["train_captions_per_image"], 2)
        self.assertEqual(config["data"]["num_workers"], 10)

    def test_dual_cross_fusion_retains_both_signals(self) -> None:
        dual = torch.tensor([[3.0, 2.0, 1.0]])
        cross = torch.tensor([[1.0, 3.0, 2.0]])
        fused = _fused_scores(dual, cross, 0.5)
        self.assertEqual(fused.shape, dual.shape)
        self.assertFalse(torch.equal(fused, dual))
        self.assertFalse(torch.equal(fused, cross))

    def test_results_notebook_has_selective_cells_for_every_pipeline_stage(self) -> None:
        path = Path("notebooks/02_alignment_v2_results.ipynb")
        notebook = json.loads(path.read_text())
        expected_stages = ("validate", "prefetch", "reference", "train", "evaluate", "report", "status")
        for stage in expected_stages:
            indices = selected_cell_indices(notebook, stage)
            self.assertEqual(len(indices), 2, stage)
            self.assertEqual(indices[0], 1, stage)
        self.assertEqual(len(selected_cell_indices(notebook, "all")), 8)


if __name__ == "__main__":
    unittest.main()
