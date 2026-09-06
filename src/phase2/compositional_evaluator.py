from __future__ import annotations

import pandas as pd


def compositional_status() -> dict[str, str]:
    return {
        "training_status": "evaluation_only",
        "reason": "No separate leakage-safe compositional training split is registered; Winoground and SugarCrepe remain evaluation-only.",
    }


def paired_accuracy(frame: pd.DataFrame, positive_column: str = "positive_score", negative_column: str = "negative_score") -> float:
    if frame.empty:
        return float("nan")
    return float(frame[positive_column].gt(frame[negative_column]).mean())
