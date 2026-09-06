from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch


def candidate_coverage(candidate_ids: list[list[str]], positive_ids: list[set[str]]) -> float:
    if len(candidate_ids) != len(positive_ids):
        raise ValueError("candidate and positive rows must align")
    if not candidate_ids:
        return float("nan")
    return float(np.mean([bool(set(candidates).intersection(positives)) for candidates, positives in zip(candidate_ids, positive_ids)]))


def rerank_topk(
    dual_scores: torch.Tensor,
    cross_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rerank only dual-encoder top-K candidates.

    ``cross_score`` receives flattened query and candidate index tensors and
    returns one score per pair. The full Cartesian product is never scored.
    """
    if dual_scores.ndim != 2:
        raise ValueError("dual_scores must be [queries,candidates]")
    depth = min(int(depth), dual_scores.shape[1])
    _, candidate_indices = dual_scores.topk(depth, dim=1)
    query_indices = torch.arange(dual_scores.shape[0], device=dual_scores.device).unsqueeze(1).expand_as(candidate_indices)
    scores = cross_score(query_indices.reshape(-1), candidate_indices.reshape(-1)).reshape_as(candidate_indices)
    order = scores.argsort(dim=1, descending=True)
    return candidate_indices.gather(1, order), scores.gather(1, order)


def retrieval_metrics(ranked_ids: list[list[str]], positives: list[set[str]]) -> dict[str, float]:
    ranks = []
    for ranking, valid in zip(ranked_ids, positives):
        rank = next((index + 1 for index, item in enumerate(ranking) if item in valid), len(ranking) + 1)
        ranks.append(rank)
    values = np.asarray(ranks, dtype=float)
    return {
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "mean_rank": float(values.mean()),
        "median_rank": float(np.median(values)),
    }


def per_sample_retrieval_rows(query_ids, ranked_ids, positives, depth: int) -> pd.DataFrame:
    rows = []
    for query, ranking, valid in zip(query_ids, ranked_ids, positives):
        rank = next((i + 1 for i, item in enumerate(ranking) if item in valid), len(ranking) + 1)
        rows.append({"query_id": query, "candidate_depth": depth, "positive_rank": rank, "covered": rank <= depth})
    return pd.DataFrame(rows)
