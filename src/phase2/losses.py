from __future__ import annotations

import torch
import torch.nn.functional as F


def image_text_matching_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    scores = torch.cat((positive_scores.reshape(-1), negative_scores.reshape(-1)))
    targets = torch.cat((torch.ones_like(positive_scores).reshape(-1), torch.zeros_like(negative_scores).reshape(-1)))
    return F.binary_cross_entropy_with_logits(scores, targets)


def pairwise_ranking_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    return F.relu(float(margin) - positive_scores.reshape(-1) + negative_scores.reshape(-1)).mean()


def projection_contrastive_loss(positive_features: torch.Tensor, negative_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    positive = F.normalize(positive_features, dim=-1)
    negative = F.normalize(negative_features, dim=-1)
    logits = torch.stack(((positive * positive.detach()).sum(-1), (positive * negative).sum(-1)), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))


def bridge_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    positive_features: torch.Tensor | None = None,
    negative_features: torch.Tensor | None = None,
    rank_weight: float = 1.0,
    contrastive_weight: float = 0.0,
    margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    itm = image_text_matching_loss(positive_scores, negative_scores)
    ranking = pairwise_ranking_loss(positive_scores, negative_scores, margin)
    contrastive = itm.new_zeros(())
    if contrastive_weight and positive_features is not None and negative_features is not None:
        contrastive = projection_contrastive_loss(positive_features, negative_features)
    total = itm + rank_weight * ranking + contrastive_weight * contrastive
    return {"loss": total, "itm_loss": itm, "ranking_loss": ranking, "contrastive_loss": contrastive}
