from __future__ import annotations

import torch

from src.alignment_v3.model import LearnedQueryTextPool
from src.alignment_v3.text_aggregation_study import select_efficiency_cell
from src.models.text_encoders import mean_pool


def test_masked_mean_legacy_path_is_bitwise_identical() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(3, 7, 384)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    legacy = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(1) / mask.sum(1).unsqueeze(-1).clamp_min(1.0)
    assert torch.equal(mean_pool(hidden, mask), legacy)


def test_learned_query_ignores_padding_and_preserves_shape() -> None:
    torch.manual_seed(17)
    pool = LearnedQueryTextPool(384, pool_dim=96, heads=3).eval()
    valid = torch.randn(2, 5, 384)
    padded_a = torch.cat((valid, torch.randn(2, 3, 384)), dim=1)
    padded_b = torch.cat((valid, torch.randn(2, 3, 384) * 1000), dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]] * 2)
    baseline = mean_pool(padded_a, mask)
    out_a = pool(padded_a, mask, baseline)
    out_b = pool(padded_b, mask, baseline)
    assert out_a.shape == (2, 384)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


def test_parameter_additions_are_exact_and_under_budget() -> None:
    expected = {(128, 4): 165_761, (96, 3): 112_321}
    for (dim, heads), count in expected.items():
        pool = LearnedQueryTextPool(384, pool_dim=dim, heads=heads)
        assert sum(parameter.numel() for parameter in pool.parameters()) == count
        assert 2_730_628 + count < 5_000_000


def test_frozen_input_does_not_receive_gradients() -> None:
    pool = LearnedQueryTextPool(384, pool_dim=128, heads=4)
    hidden = torch.randn(2, 6, 384)
    mask = torch.ones(2, 6, dtype=torch.long)
    loss = pool(hidden, mask, hidden.mean(1)).square().mean()
    loss.backward()
    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in pool.parameters())


def test_primary_fallback_and_null_hierarchy() -> None:
    def rows(primary: bool, small: bool):
        return {
            "M_T0": {"efficiency_pass": True},
            "M_T1": {"efficiency_pass": primary},
            "M_T1_small": {"efficiency_pass": small},
        }
    assert select_efficiency_cell("M", rows(True, True)) == "M_T1"
    assert select_efficiency_cell("M", rows(False, True)) == "M_T1_small"
    assert select_efficiency_cell("M", rows(False, False)) == "M_T0"
