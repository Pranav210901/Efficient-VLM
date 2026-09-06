"""Correctness of the fresh-queue identification control.

The control is only meaningful if it changes exactly one thing.  These tests
pin that down:

  * with the projector unmoved, a fresh queue is numerically identical to a
    stale one -- if this fails, the re-projection path is wrong;
  * once the projector moves, they differ -- if this fails, nothing is being
    controlled;
  * the negative set, its order and the positive masks are untouched.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.training.losses import CrossBatchMemory, contrastive_loss_with_memory


class _Head(nn.Module):
    """Deterministic stand-in for ResidualProjectionHead."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _fixture(seed: int = 0):
    torch.manual_seed(seed)
    image_head, text_head = _Head(), _Head()
    features_i, features_t = torch.randn(4, 8), torch.randn(4, 8)
    with torch.no_grad():
        z_i = torch.nn.functional.normalize(image_head(features_i), dim=-1)
        z_t = torch.nn.functional.normalize(text_head(features_t), dim=-1)
    memory = CrossBatchMemory(8, store_features=True)
    memory.enqueue(z_i, z_t, ["a", "b", "c", "d"], ["a", "b", "c", "d"],
                   optimizer_step=1, image_features=features_i, text_features=features_t)
    batch_i = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    batch_t = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    ids = ["x", "y", "z"]
    return image_head, text_head, memory, batch_i, batch_t, ids


def _loss(memory, batch_i, batch_t, ids, image_head=None, text_head=None):
    return contrastive_loss_with_memory(
        batch_i, batch_t, torch.tensor(10.0), ids, ids, memory,
        image_reprojector=image_head, text_reprojector=text_head,
    )


def test_fresh_equals_stale_before_the_projector_moves():
    """The identifying assertion: at zero staleness the arms coincide."""
    image_head, text_head, memory, bi, bt, ids = _fixture()
    stale = _loss(memory, bi, bt, ids)
    fresh = _loss(memory, bi, bt, ids, image_head, text_head)
    assert torch.allclose(stale, fresh, atol=1e-6), (
        f"fresh re-projection is not identity at zero staleness: {stale} vs {fresh}"
    )


def test_fresh_differs_once_the_projector_moves():
    image_head, text_head, memory, bi, bt, ids = _fixture()
    stale = _loss(memory, bi, bt, ids)
    with torch.no_grad():  # simulate an optimizer step
        for head in (image_head, text_head):
            head.linear.weight.add_(torch.randn_like(head.linear.weight) * 0.5)
    fresh = _loss(memory, bi, bt, ids, image_head, text_head)
    assert not torch.allclose(stale, fresh, atol=1e-4), "the control changed nothing"


def test_negative_set_is_unchanged():
    """Fresh must alter the residents' values, never their identity or count."""
    image_head, text_head, memory, bi, bt, ids = _fixture()
    before = (list(memory.image_ids), list(memory.text_image_ids),
              memory.image_embeddings.shape, memory.text_embeddings.shape)
    _loss(memory, bi, bt, ids, image_head, text_head)
    after = (list(memory.image_ids), list(memory.text_image_ids),
             memory.image_embeddings.shape, memory.text_embeddings.shape)
    assert before == after


def test_fresh_requires_stored_features():
    _, _, _, bi, bt, ids = _fixture()
    memory = CrossBatchMemory(8, store_features=False)
    memory.enqueue(bi, bt, ids, ids, optimizer_step=1)
    try:
        _loss(memory, bi, bt, ids, _Head(), _Head())
    except ValueError as error:
        assert "store_features" in str(error)
    else:
        raise AssertionError("expected a ValueError when features were never stored")


def test_store_features_requires_features_at_enqueue():
    memory = CrossBatchMemory(8, store_features=True)
    try:
        memory.enqueue(torch.randn(2, 8), torch.randn(2, 8), ["a", "b"], ["a", "b"])
    except ValueError as error:
        assert "store_features" in str(error)
    else:
        raise AssertionError("expected a ValueError when features were omitted")


def test_stale_path_is_untouched_by_the_change():
    """Existing arms must be bit-identical to their pre-change behaviour."""
    _, _, memory, bi, bt, ids = _fixture()
    plain = CrossBatchMemory(8)
    plain.enqueue(memory.image_embeddings, memory.text_embeddings,
                  memory.image_ids, memory.text_image_ids, optimizer_step=1)
    assert torch.equal(_loss(memory, bi, bt, ids), _loss(plain, bi, bt, ids))


def _semantic_fixture():
    batch_i = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch_t = batch_i.clone()
    queued_i = torch.tensor([[0.8, 0.6], [0.6, 0.8], [-1.0, 0.0]])
    queued_t = queued_i.clone()
    memory = CrossBatchMemory(3, store_semantic_embeddings=True)
    memory.enqueue(
        queued_i,
        queued_t,
        ["q0", "q1", "q2"],
        ["q0", "q1", "q2"],
        image_semantic_embeddings=queued_i,
        text_semantic_embeddings=queued_t,
    )
    return batch_i, batch_t, memory


def test_semantic_queue_filter_removes_only_suspicious_negatives():
    bi, bt, memory = _semantic_fixture()
    baseline = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory,
        return_diagnostics=True,
    )
    filtered = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory,
        return_diagnostics=True,
        semantic_image_embeddings=bi,
        semantic_text_embeddings=bt,
        queue_filter_mode="semantic",
        queue_filter_threshold=0.75,
    )
    assert filtered.total < baseline.total
    assert abs(filtered.image_queue_filtered_fraction - 2 / 6) < 1e-7
    assert abs(filtered.text_queue_filtered_fraction - 2 / 6) < 1e-7


def test_matched_random_filter_removes_the_same_count():
    bi, bt, memory = _semantic_fixture()
    semantic = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory,
        return_diagnostics=True,
        semantic_image_embeddings=bi,
        semantic_text_embeddings=bt,
        queue_filter_mode="semantic",
        queue_filter_threshold=0.75,
    )
    random_control = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory,
        return_diagnostics=True,
        semantic_image_embeddings=bi,
        semantic_text_embeddings=bt,
        queue_filter_mode="matched_random",
        queue_filter_threshold=0.75,
        queue_filter_seed=123,
    )
    assert random_control.image_queue_filtered_fraction == semantic.image_queue_filtered_fraction
    assert random_control.text_queue_filtered_fraction == semantic.text_queue_filtered_fraction


def test_match_inbatch_queue_weight_reduces_queue_denominator_mass():
    bi, bt, memory = _semantic_fixture()
    plain = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory
    )
    normalised = contrastive_loss_with_memory(
        bi, bt, torch.tensor(10.0), ["a", "b"], ["a", "b"], memory,
        queue_weight_mode="match_inbatch",
    )
    assert normalised < plain
