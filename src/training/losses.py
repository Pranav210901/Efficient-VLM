from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# the diagonal case: every caption in the batch is the single positive for its image
def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] != logits.shape[1]:
        raise ValueError("diagonal CLIP loss requires a square logit matrix")
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def positive_mask(
    query_ids: Sequence[Hashable],
    target_ids: Sequence[Hashable],
    device: torch.device,
) -> torch.Tensor:
    target_lookup: dict[Hashable, list[int]] = {}
    for index, target_id in enumerate(target_ids):
        target_lookup.setdefault(target_id, []).append(index)
    mask = torch.zeros((len(query_ids), len(target_ids)), dtype=torch.bool, device=device)
    for query_index, query_id in enumerate(query_ids):
        indices = target_lookup.get(query_id, [])
        if indices:
            mask[query_index, torch.tensor(indices, device=device)] = True
    if not mask.any(dim=1).all():
        raise ValueError("every contrastive query must have at least one positive")
    return mask


def _multi_positive_direction(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    if logits.shape != positives.shape:
        raise ValueError("logits and positive mask must have matching shapes")
    positive_logits = logits.masked_fill(~positives, torch.finfo(logits.dtype).min)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


# COCO gives five captions per image, so a batch has several correct answers per row;
# the diagonal loss above would score four of them as negatives
def multi_positive_contrastive_loss(
    logits: torch.Tensor,
    image_ids: Sequence[Hashable],
    text_image_ids: Sequence[Hashable],
) -> torch.Tensor:
    positives = positive_mask(image_ids, text_image_ids, logits.device)
    return 0.5 * (
        _multi_positive_direction(logits, positives)
        + _multi_positive_direction(logits.t(), positives.t())
    )


class CrossBatchMemory:
    # a short detached queue that adds negatives without changing inference cost

    def __init__(
        self,
        capacity: int,
        *,
        enable_image: bool = True,
        enable_text: bool = True,
    ) -> None:
        self.capacity = max(0, int(capacity))
        self.enable_image = bool(enable_image) and self.capacity > 0
        self.enable_text = bool(enable_text) and self.capacity > 0
        self.image_embeddings: torch.Tensor | None = None
        self.text_embeddings: torch.Tensor | None = None
        self.image_ids: list[Hashable] = []
        self.text_image_ids: list[Hashable] = []
        self.image_insertion_steps: list[int] = []
        self.text_insertion_steps: list[int] = []
        self.image_exposures: list[int] = []
        self.text_exposures: list[int] = []

    def _append(self, current: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        combined = values.detach() if current is None else torch.cat((current, values.detach()), dim=0)
        return combined[-self.capacity :] if self.capacity else combined[:0]

    @torch.no_grad()
    def enqueue(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_ids: Sequence[Hashable],
        text_image_ids: Sequence[Hashable],
        *,
        optimizer_step: int | None = None,
    ) -> None:
        if self.capacity == 0:
            return
        step = int(optimizer_step or 0)
        if self.enable_image:
            self.image_embeddings = self._append(self.image_embeddings, image_embeddings)
            self.image_ids = (self.image_ids + list(image_ids))[-self.capacity :]
            self.image_insertion_steps = (
                self.image_insertion_steps + [step] * len(image_ids)
            )[-self.capacity :]
            self.image_exposures = (
                self.image_exposures + [0] * len(image_ids)
            )[-self.capacity :]
        if self.enable_text:
            self.text_embeddings = self._append(self.text_embeddings, text_embeddings)
            self.text_image_ids = (self.text_image_ids + list(text_image_ids))[-self.capacity :]
            self.text_insertion_steps = (
                self.text_insertion_steps + [step] * len(text_image_ids)
            )[-self.capacity :]
            self.text_exposures = (
                self.text_exposures + [0] * len(text_image_ids)
            )[-self.capacity :]

    def record_exposure(self) -> None:
        if self.image_exposures:
            self.image_exposures = [value + 1 for value in self.image_exposures]
        if self.text_exposures:
            self.text_exposures = [value + 1 for value in self.text_exposures]

    def is_full(self, modality: str) -> bool:
        enabled = self.enable_image if modality == "image" else self.enable_text
        values = self.image_ids if modality == "image" else self.text_image_ids
        return (not enabled) or len(values) >= self.capacity

    @staticmethod
    def _summary(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        count = len(ordered)
        return {
            "mean": sum(ordered) / count,
            "p50": ordered[min(count - 1, int(0.50 * (count - 1)))],
            "p95": ordered[min(count - 1, int(0.95 * (count - 1)))],
            "max": ordered[-1],
        }

    def diagnostics(self, optimizer_step: int) -> dict[str, Any]:
        # return detached queue diagnostics; first reuse after step s has age one
        step = int(optimizer_step)
        image_ages = [step - value for value in self.image_insertion_steps]
        text_ages = [step - value for value in self.text_insertion_steps]
        return {
            "image_age": self._summary(image_ages) if self.enable_image else None,
            "text_age": self._summary(text_ages) if self.enable_text else None,
            "image_exposures": self._summary(self.image_exposures)
            if self.enable_image
            else None,
            "text_exposures": self._summary(self.text_exposures)
            if self.enable_text
            else None,
            "image_fill_fraction": (
                len(self.image_ids) / self.capacity if self.enable_image else None
            ),
            "text_fill_fraction": (
                len(self.text_image_ids) / self.capacity if self.enable_text else None
            ),
            "full": self.is_full("image") and self.is_full("text"),
        }


@dataclass(frozen=True)
class ContrastiveLossDiagnostics:
    total: torch.Tensor
    i2t: torch.Tensor
    t2i: torch.Tensor
    image_queue_positive_fraction: float | None
    text_queue_positive_fraction: float | None


# the queue variant. embeddings from earlier steps are stale by construction: the
# encoder has moved since they were written. this is the mechanism the study measures
def contrastive_loss_with_memory(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    image_ids: Sequence[Hashable],
    text_image_ids: Sequence[Hashable],
    memory: CrossBatchMemory | None = None,
    *,
    return_diagnostics: bool = False,
) -> torch.Tensor | ContrastiveLossDiagnostics:
    text_gallery = text_embeddings
    text_gallery_ids = list(text_image_ids)
    image_gallery = image_embeddings
    image_gallery_ids = list(image_ids)
    text_queue_start = len(text_embeddings)
    image_queue_start = len(image_embeddings)
    if memory is not None and memory.enable_text and memory.text_embeddings is not None:
        text_gallery = torch.cat((text_embeddings, memory.text_embeddings.to(text_embeddings.device)), dim=0)
        text_gallery_ids.extend(memory.text_image_ids)
    if memory is not None and memory.enable_image and memory.image_embeddings is not None:
        image_gallery = torch.cat((image_embeddings, memory.image_embeddings.to(image_embeddings.device)), dim=0)
        image_gallery_ids.extend(memory.image_ids)
    i2t_logits = logit_scale * image_embeddings @ text_gallery.t()
    t2i_logits = logit_scale * text_embeddings @ image_gallery.t()
    i2t_mask = positive_mask(image_ids, text_gallery_ids, i2t_logits.device)
    t2i_mask = positive_mask(text_image_ids, image_gallery_ids, t2i_logits.device)
    i2t = _multi_positive_direction(i2t_logits, i2t_mask)
    t2i = _multi_positive_direction(t2i_logits, t2i_mask)
    total = 0.5 * (i2t + t2i)
    if not return_diagnostics:
        return total

    def queue_positive_fraction(mask: torch.Tensor, start: int) -> float | None:
        queued = mask[:, start:]
        if queued.shape[1] == 0:
            return None
        # A resident entry is positive if it is positive for at least one
        # current query. This is detached diagnostic work on an existing mask.
        return float(queued.any(dim=0).float().mean().detach().cpu())

    return ContrastiveLossDiagnostics(
        total=total,
        i2t=i2t,
        t2i=t2i,
        image_queue_positive_fraction=queue_positive_fraction(
            t2i_mask, image_queue_start
        ),
        text_queue_positive_fraction=queue_positive_fraction(
            i2t_mask, text_queue_start
        ),
    )


def sigmoid_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    image_ids: Sequence[Hashable],
    text_image_ids: Sequence[Hashable],
) -> torch.Tensor:
    # sigLIP-style pairwise logistic loss, normalized by image count
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("sigmoid contrastive embeddings must be matrices")
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("image/text embedding dimensions must match")
    if image_embeddings.shape[0] != len(image_ids):
        raise ValueError("image id count does not match image embeddings")
    if text_embeddings.shape[0] != len(text_image_ids):
        raise ValueError("text id count does not match text embeddings")
    logits = logit_scale * (image_embeddings @ text_embeddings.t()) + logit_bias
    positives = positive_mask(image_ids, text_image_ids, logits.device)
    labels = positives.to(dtype=logits.dtype).mul(2).sub(1)
    return -torch.nn.functional.logsigmoid(labels * logits).sum() / max(
        1, image_embeddings.shape[0]
    )
