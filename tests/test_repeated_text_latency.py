from __future__ import annotations

import pandas as pd

from src.alignment_v3.repeated_text_latency import REPETITIONS, SEEDS, summarize


def test_summary_preserves_pairing_and_emits_all_groups() -> None:
    rows = []
    for seed in SEEDS:
        for repetition in range(1, REPETITIONS + 1):
            rows.extend([
                {"arm": "M_T0", "seed": seed, "repetition": repetition, "q3_ms": 9.0 + seed / 10000 + repetition / 100000},
                {"arm": "M_T1", "seed": seed, "repetition": repetition, "q3_ms": 9.2 + seed / 10000 + repetition / 100000},
            ])
    summary, paired = summarize(pd.DataFrame(rows))
    assert len(paired) == 30
    assert len(summary) == 12
    pooled = summary.loc[
        (summary.comparison == "M_T1_minus_M_T0_paired")
        & (summary.seed == "pooled")
    ].iloc[0]
    assert abs(float(pooled.mean_q3_ms) - 0.2) < 1e-12
