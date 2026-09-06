from __future__ import annotations

import numpy as np

from src.alignment_v3.final_bootstrap import _paired_bootstrap, _paired_retrieval_bootstrap


def test_paired_bootstrap_preserves_known_effect() -> None:
    result = _paired_bootstrap(
        np.ones(20) * 0.25,
        iterations=1000,
        rng=np.random.default_rng(7),
    )
    assert result["effect"] == 0.25
    assert result["ci_low"] == 0.25
    assert result["ci_high"] == 0.25


def test_retrieval_bootstrap_averages_directions() -> None:
    result = _paired_retrieval_bootstrap(
        np.ones(10) * 0.2,
        np.ones(20) * 0.4,
        iterations=1000,
        rng=np.random.default_rng(11),
    )
    assert abs(result["effect"] - 0.3) < 1e-12
    assert abs(result["ci_low"] - 0.3) < 1e-12
    assert abs(result["ci_high"] - 0.3) < 1e-12
