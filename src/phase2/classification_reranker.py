from __future__ import annotations

import torch


def rerank_class_candidates(dual_logits: torch.Tensor, cross_scores: torch.Tensor, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dual_logits.ndim != 2:
        raise ValueError("dual_logits must be [samples,classes]")
    depth = min(int(depth), dual_logits.shape[1])
    candidates = dual_logits.topk(depth, dim=1).indices
    if cross_scores.shape != candidates.shape:
        raise ValueError("cross_scores must align with the selected top-K candidates")
    order = cross_scores.argsort(dim=1, descending=True)
    return candidates.gather(1, order), cross_scores.gather(1, order)


def classification_metrics(ranked_classes: torch.Tensor, targets: torch.Tensor, candidate_count: int) -> dict[str, float]:
    targets = targets.reshape(-1, 1).to(ranked_classes.device)
    covered = ranked_classes.eq(targets).any(dim=1)
    return {
        "candidate_coverage": float(covered.float().mean()),
        "top1": float(ranked_classes[:, :1].eq(targets).any(dim=1).float().mean()),
        "top5": float(ranked_classes[:, : min(5, ranked_classes.shape[1])].eq(targets).any(dim=1).float().mean()),
        "candidate_count": int(candidate_count),
    }
