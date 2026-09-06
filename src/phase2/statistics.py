from __future__ import annotations

import math
import pandas as pd


def seed_statistics(frame: pd.DataFrame, group_columns: list[str], metric_column: str = "value") -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[metric_column]
    result = grouped.agg(["count", "mean", "std"]).reset_index()
    result["standard_error"] = result["std"] / result["count"].pow(0.5)
    result["ci95_low"] = result["mean"] - 1.96 * result["standard_error"]
    result["ci95_high"] = result["mean"] + 1.96 * result["standard_error"]
    return result


def matched_seed_differences(frame: pd.DataFrame, method_a: str, method_b: str) -> dict[str, float | int]:
    pivot = frame.pivot_table(index=["seed", "task", "metric"], columns="method", values="value").dropna()
    difference = pivot[method_a] - pivot[method_b]
    return {"mean_difference": float(difference.mean()), "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()), "ties": int((difference == 0).sum())}


def pareto_front(frame: pd.DataFrame, performance: str = "performance", cost: str = "latency_ms") -> pd.DataFrame:
    finite = frame[frame[performance].notna() & frame[cost].notna()].copy()
    keep = []
    for index, row in finite.iterrows():
        dominated = ((finite[performance] >= row[performance]) & (finite[cost] <= row[cost]) & ((finite[performance] > row[performance]) | (finite[cost] < row[cost]))).any()
        keep.append(not dominated)
    finite["pareto_optimal"] = keep
    return finite[finite["pareto_optimal"]].sort_values(cost).reset_index(drop=True)
