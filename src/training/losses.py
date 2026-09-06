from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


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
    """A short detached queue that adds negatives without changing inference cost."""

    def __init__(
        self,
        capacity: int,
        *,
        enable_image: bool = True,
        enable_text: bool = True,
        store_features: bool = False,
        store_semantic_embeddings: bool = False,
    ) -> None:
        self.capacity = max(0, int(capacity))
        self.enable_image = bool(enable_image) and self.capacity > 0
        self.enable_text = bool(enable_text) and self.capacity > 0
        # When set, the queue also retains the pre-projection representation of
        # each entry so the identification control can re-project residents
        # through the current projector.  Storage only; it does not by itself
        # change the loss.
        self.store_features = bool(store_features)
        # Optional embeddings from an independently frozen VLM.  These are
        # used only to identify semantically suspicious queued negatives; they
        # never receive gradients and never change the deployed model.
        self.store_semantic_embeddings = bool(store_semantic_embeddings)
        self.image_embeddings: torch.Tensor | None = None
        self.text_embeddings: torch.Tensor | None = None
        self.image_features: torch.Tensor | None = None
        self.text_features: torch.Tensor | None = None
        self.image_semantic_embeddings: torch.Tensor | None = None
        self.text_semantic_embeddings: torch.Tensor | None = None
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
        image_features: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
        image_semantic_embeddings: torch.Tensor | None = None,
        text_semantic_embeddings: torch.Tensor | None = None,
    ) -> None:
        if self.capacity == 0:
            return
        if self.store_features and (image_features is None or text_features is None):
            raise ValueError(
                "store_features queues require image_features and text_features"
            )
        if self.store_semantic_embeddings and (
            image_semantic_embeddings is None or text_semantic_embeddings is None
        ):
            raise ValueError(
                "semantic-filter queues require image and text semantic embeddings"
            )
        step = int(optimizer_step or 0)
        if self.enable_image:
            self.image_embeddings = self._append(self.image_embeddings, image_embeddings)
            if self.store_features:
                self.image_features = self._append(self.image_features, image_features)
            if self.store_semantic_embeddings:
                self.image_semantic_embeddings = self._append(
                    self.image_semantic_embeddings, image_semantic_embeddings
                )
            self.image_ids = (self.image_ids + list(image_ids))[-self.capacity :]
            self.image_insertion_steps = (
                self.image_insertion_steps + [step] * len(image_ids)
            )[-self.capacity :]
            self.image_exposures = (
                self.image_exposures + [0] * len(image_ids)
            )[-self.capacity :]
        if self.enable_text:
            self.text_embeddings = self._append(self.text_embeddings, text_embeddings)
            if self.store_features:
                self.text_features = self._append(self.text_features, text_features)
            if self.store_semantic_embeddings:
                self.text_semantic_embeddings = self._append(
                    self.text_semantic_embeddings, text_semantic_embeddings
                )
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
        """Return detached queue diagnostics; first reuse after step s has age one."""
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
    image_queue_filtered_fraction: float | None = None
    text_queue_filtered_fraction: float | None = None


def contrastive_loss_with_memory(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    image_ids: Sequence[Hashable],
    text_image_ids: Sequence[Hashable],
    memory: CrossBatchMemory | None = None,
    *,
    return_diagnostics: bool = False,
    image_reprojector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    text_reprojector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    semantic_image_embeddings: torch.Tensor | None = None,
    semantic_text_embeddings: torch.Tensor | None = None,
    queue_filter_mode: str = "none",
    queue_filter_threshold: float = 0.222935,
    queue_filter_seed: int = 0,
    queue_weight_mode: str = "none",
) -> torch.Tensor | ContrastiveLossDiagnostics:
    """Multi-positive InfoNCE, optionally extended by a cross-batch queue.

    Passing ``image_reprojector``/``text_reprojector`` switches the queue from
    stale to *fresh*: residents are re-projected from their stored
    pre-projection features through the current projector before entering the
    denominator.  The negative set, its ordering, and the positive masks are
    unchanged, so staleness is the only quantity that differs from the stale
    arm.  Re-projection is detached, matching the stale queue's semantics.
    """

    def _resident(
        stored_embeddings: torch.Tensor,
        stored_features: torch.Tensor | None,
        reprojector: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device,
    ) -> torch.Tensor:
        if reprojector is None:
            return stored_embeddings.to(device)
        if stored_features is None:
            raise ValueError(
                "fresh re-projection requested but the queue stored no features; "
                "construct CrossBatchMemory with store_features=True"
            )
        with torch.no_grad():
            return F.normalize(reprojector(stored_features.to(device)), dim=-1).detach()

    text_gallery = text_embeddings
    text_gallery_ids = list(text_image_ids)
    image_gallery = image_embeddings
    image_gallery_ids = list(image_ids)
    text_queue_start = len(text_embeddings)
    image_queue_start = len(image_embeddings)
    if memory is not None and memory.enable_text and memory.text_embeddings is not None:
        resident_text = _resident(
            memory.text_embeddings, memory.text_features, text_reprojector, text_embeddings.device
        )
        text_gallery = torch.cat((text_embeddings, resident_text.to(text_embeddings.dtype)), dim=0)
        text_gallery_ids.extend(memory.text_image_ids)
    if memory is not None and memory.enable_image and memory.image_embeddings is not None:
        resident_image = _resident(
            memory.image_embeddings, memory.image_features, image_reprojector, image_embeddings.device
        )
        image_gallery = torch.cat((image_embeddings, resident_image.to(image_embeddings.dtype)), dim=0)
        image_gallery_ids.extend(memory.image_ids)
    i2t_logits = logit_scale * image_embeddings @ text_gallery.t()
    t2i_logits = logit_scale * text_embeddings @ image_gallery.t()
    i2t_mask = positive_mask(image_ids, text_gallery_ids, i2t_logits.device)
    t2i_mask = positive_mask(text_image_ids, image_gallery_ids, t2i_logits.device)

    if queue_weight_mode not in {"none", "match_inbatch"}:
        raise ValueError(f"unsupported queue_weight_mode {queue_weight_mode!r}")
    if queue_filter_mode not in {"none", "semantic", "matched_random"}:
        raise ValueError(f"unsupported queue_filter_mode {queue_filter_mode!r}")

    def _queue_log_weight(inbatch: int, queued: int) -> float:
        if queue_weight_mode == "none" or queued <= 0:
            return 0.0
        # Give the entire queue the same nominal denominator mass as the
        # in-batch gallery.  This is fixed by counts, not tuned on outcomes.
        return float(torch.log(torch.tensor(inbatch / queued)).item())

    if text_gallery.shape[0] > text_queue_start:
        i2t_logits[:, text_queue_start:] += _queue_log_weight(
            text_queue_start, text_gallery.shape[0] - text_queue_start
        )
    if image_gallery.shape[0] > image_queue_start:
        t2i_logits[:, image_queue_start:] += _queue_log_weight(
            image_queue_start, image_gallery.shape[0] - image_queue_start
        )

    def _matched_random_mask(
        eligible: torch.Tensor, counts: torch.Tensor, *, seed: int
    ) -> torch.Tensor:
        """Select exactly ``counts[row]`` eligible cells using a local RNG."""
        if eligible.numel() == 0 or int(counts.max().item()) == 0:
            return torch.zeros_like(eligible)
        generator = torch.Generator(device=eligible.device)
        generator.manual_seed(int(seed))
        scores = torch.rand(
            eligible.shape, device=eligible.device, dtype=torch.float32, generator=generator
        ).masked_fill(~eligible, -1.0)
        width = int(counts.max().item())
        selected = scores.topk(width, dim=1).indices
        keep_rank = torch.arange(width, device=eligible.device).unsqueeze(0) < counts.unsqueeze(1)
        result = torch.zeros_like(eligible)
        result.scatter_(1, selected, keep_rank)
        return result & eligible

    def _filter_mask(
        query_semantic: torch.Tensor | None,
        resident_semantic: torch.Tensor | None,
        positives: torch.Tensor,
        start: int,
        *,
        seed: int,
    ) -> torch.Tensor | None:
        queued_positives = positives[:, start:]
        if queued_positives.shape[1] == 0 or queue_filter_mode == "none":
            return None
        if query_semantic is None or resident_semantic is None:
            raise ValueError(
                "queue filtering requires current and resident semantic embeddings"
            )
        query = F.normalize(query_semantic.float(), dim=-1)
        resident = F.normalize(resident_semantic.float(), dim=-1)
        suspicious = query @ resident.t() >= float(queue_filter_threshold)
        suspicious &= ~queued_positives
        if queue_filter_mode == "semantic":
            return suspicious
        eligible = ~queued_positives
        return _matched_random_mask(
            eligible, suspicious.sum(dim=1), seed=seed
        )

    text_filter = _filter_mask(
        semantic_image_embeddings,
        None if memory is None else memory.text_semantic_embeddings,
        i2t_mask,
        text_queue_start,
        seed=queue_filter_seed + 17,
    )
    image_filter = _filter_mask(
        semantic_text_embeddings,
        None if memory is None else memory.image_semantic_embeddings,
        t2i_mask,
        image_queue_start,
        seed=queue_filter_seed + 29,
    )
    if text_filter is not None:
        i2t_logits[:, text_queue_start:] = i2t_logits[:, text_queue_start:].masked_fill(
            text_filter, torch.finfo(i2t_logits.dtype).min
        )
    if image_filter is not None:
        t2i_logits[:, image_queue_start:] = t2i_logits[:, image_queue_start:].masked_fill(
            image_filter, torch.finfo(t2i_logits.dtype).min
        )
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
        image_queue_filtered_fraction=(
            None if image_filter is None else float(image_filter.float().mean().detach().cpu())
        ),
        text_queue_filtered_fraction=(
            None if text_filter is None else float(text_filter.float().mean().detach().cpu())
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
    """SigLIP-style pairwise logistic loss, normalized by image count.

    This deliberately operates on the current batch only.  The per-image
    normalization is part of the Wave 0 pre-registration and means the loss
    magnitude grows with the number of captions per image.
    """
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
