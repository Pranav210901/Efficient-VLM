from __future__ import annotations

import torch

from src.alignment_v3.guarded_hard_negative_study import (
    IMAGE_TEXT_THRESHOLD,
    TEXT_TEXT_THRESHOLD,
    _guard_block,
    _scatter_owner_max,
)


def test_owner_max_preserves_best_caption_per_image() -> None:
    scores = torch.tensor([[0.1, 0.7, 0.2, 0.4], [0.9, 0.3, 0.8, 0.1]])
    owners = torch.tensor([0, 0, 1, 1])
    observed = _scatter_owner_max(scores, owners, 2)
    expected = torch.tensor([[0.7, 0.4], [0.9, 0.8]])
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_either_cross_modal_direction_excludes_candidate() -> None:
    # Pair 0 crosses only query-image -> candidate-caption. Pair 1 crosses
    # only query-caption -> candidate-image. Both must be excluded.
    teacher_images = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    teacher_texts = torch.tensor([
        [0.0, 1.0],  # owner 0
        [1.0, 0.0],  # owner 1: high for image 0
        [0.0, 1.0],  # owner 2: high for image 1
    ])
    indices = torch.tensor([[0], [1], [2]])
    mask = torch.ones_like(indices, dtype=torch.bool)
    query = torch.tensor([0, 1])
    candidate = torch.tensor([1, 2])
    cross, _, excluded = _guard_block(query, candidate, teacher_images, teacher_texts, indices, mask)
    assert torch.all(cross >= IMAGE_TEXT_THRESHOLD)
    assert excluded.tolist() == [True, True]


def test_caption_guard_is_or_combined_with_cross_modal_guard() -> None:
    # Cross-modal similarities are zero, while caption-caption similarity is
    # one. Caption semantics alone must exclude the candidate.
    teacher_images = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    teacher_texts = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    indices = torch.tensor([[0], [1]])
    mask = torch.ones_like(indices, dtype=torch.bool)
    cross, text, excluded = _guard_block(
        torch.tensor([0]), torch.tensor([1]), teacher_images, teacher_texts, indices, mask
    )
    assert float(cross[0]) < IMAGE_TEXT_THRESHOLD
    assert float(text[0]) >= TEXT_TEXT_THRESHOLD
    assert excluded.tolist() == [True]
