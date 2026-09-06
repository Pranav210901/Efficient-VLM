# Literature-map plan: frozen alignment and queue staleness

Date: 2026-07-26

## Decision

Determine which parts of the current Alignment project are established prior
art and whether a queue-age study can support a defensible dissertation
contribution.

## Scope

- Frozen, independently pretrained vision and text encoders.
- Lightweight projector-only image-text alignment.
- InfoNCE, multi-positive and sigmoid objectives.
- Cross-batch memories, feature drift, queue age and momentum correction.
- Vision-language teacher distillation.
- Efficient retrieval cascades and adaptive encoder routing.
- Pre-registration and reproducible ML experimentation.

This is a systematic relevance search, not a claim of bibliographic
exhaustiveness. Searches prioritised primary papers and official proceedings.

## Falsifiable hypotheses

1. Increasing detached queue age harms projector-only dual-frozen alignment
   even though additional fresh negatives would ordinarily help.
2. At matched age, degradation is more stable across batch sizes than at
   matched raw capacity.
3. An image-only queue directly perturbs the t2i loss branch more strongly,
   while a text-only queue directly perturbs the i2t branch more strongly.
   Downstream i2t/t2i R@1 direction is not pre-specified because both projectors
   receive gradients from the bidirectional objective.
4. No directly relevant prior paper reports the same cross-modal,
   dual-frozen, projector-only, modality-separated queue-age study.

Secondary, low-confidence full-queue hypothesis for unseen runs: if
modality-specific age dominates the coupled optimisation, t2i should degrade
more than i2t because the image queue is older. This sign is already
contradicted by the observed queue-zero/16,384 endpoints at batches 512 and
1,024, where i2t degraded more in absolute R@1. It is therefore not treated as
a clean preregistered claim; the single-modality branch tests are primary.

## Important pre-registration boundary

The existing queue-zero and queue-16,384 results, and the one-seed
0/1,024/4,096/8,192/16,384 curve, have already been observed. Hypotheses about
those results are retrospective. Only new matched-age, additional-seed and
single-modality experiments can be confirmatory.

## Search strategy

- Direct-competitor queries: frozen unimodal encoders, projector alignment,
  retrieval and zero-shot transfer.
- Mechanism queries: cross-batch memory, feature drift, stale embeddings,
  memory queues and moment/momentum correction.
- Component queries: sigmoid image-text loss, multiple positive captions,
  CLIP distillation and compact retrieval.
- System queries: bi-encoder cascades, mixtures of vision encoders and
  instance-level routing.
- Opposition queries: papers showing queues help, papers explicitly correcting
  drift, and closest methods using very large queue-free batches.

## Risk register

- “No paper exists” cannot be proven by search; claims must use
  “no directly matching study was found.”
- Batch size changes in-batch negative count as well as queue age.
- Queue capacity changes negative count and age simultaneously.
- Equal numerical image/text capacities imply unequal age when multiple
  captions are enqueued per image.
- Existing endpoint directionality cannot be presented as preregistered.
- XBM already ablates memory ratio and batch size; Adaptive Cross Batch
  Normalization already sweeps batch and memory size and corrects drift.
- SAIL and Freeze-Align sharply limit architecture-novelty claims.

## Stop criterion

Stop after direct competitors, queue/drift mechanisms, objective precedents,
distillation, routing and reproducibility each contain authoritative primary
sources, and an adversarial claim-boundary pass has been completed.
