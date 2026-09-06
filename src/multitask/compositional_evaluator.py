from __future__ import annotations

import torch


def winoground_scores(scores: torch.Tensor) -> dict[str, float]:
    """Official ordering for [image0-text0, image0-text1, image1-text0, image1-text1]."""
    if scores.ndim != 2 or scores.size(1) != 4:
        raise ValueError("Winoground scores must have shape [groups, 4]")
    s00, s01, s10, s11 = scores.unbind(dim=1)
    text = (s00 > s01) & (s11 > s10)
    image = (s00 > s10) & (s11 > s01)
    return {"text_score": float(text.float().mean()), "image_score": float(image.float().mean()), "group_score": float((text & image).float().mean())}


def sugarcrepe_scores(positive: torch.Tensor, negative: torch.Tensor, categories: list[str]) -> dict[str, float]:
    if positive.shape != negative.shape or positive.numel() != len(categories):
        raise ValueError("SugarCrepe scores and categories must have matching lengths")
    correct = positive > negative
    result = {"overall_accuracy": float(correct.float().mean()), "positive_score": float(positive.mean()), "negative_score": float(negative.mean()), "score_margin": float((positive - negative).mean())}
    for category in sorted(set(categories)):
        mask = torch.tensor([value == category for value in categories])
        result[f"accuracy/{category}"] = float(correct[mask].float().mean())
    return result
