from __future__ import annotations

import unittest

import torch

from src.training.metrics import compute_retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_one_to_one_identity_is_perfect(self) -> None:
        embeddings = torch.eye(4)
        metrics = compute_retrieval_metrics(embeddings, embeddings)
        self.assertEqual(metrics["i2t_R@1"], 1.0)
        self.assertEqual(metrics["t2i_R@1"], 1.0)

    def test_multiple_captions_use_any_matching_caption_as_positive(self) -> None:
        image_embeds = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        text_embeds = image_embeds.clone()
        image_ids = ["image-a", "image-a", "image-b", "image-b"]
        metrics = compute_retrieval_metrics(
            image_embeds,
            text_embeds,
            image_ids=image_ids,
            text_image_ids=image_ids,
        )
        self.assertEqual(metrics["i2t_R@1"], 1.0)
        self.assertEqual(metrics["t2i_R@1"], 1.0)
        self.assertEqual(metrics["num_image_queries"], 2.0)
        self.assertEqual(metrics["num_text_queries"], 4.0)

    def test_unknown_caption_image_id_is_rejected(self) -> None:
        embeddings = torch.eye(2)
        with self.assertRaisesRegex(ValueError, "unknown image ID"):
            compute_retrieval_metrics(
                embeddings,
                embeddings,
                image_ids=["a", "b"],
                text_image_ids=["a", "missing"],
            )


if __name__ == "__main__":
    unittest.main()
