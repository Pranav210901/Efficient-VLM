# Research plan: efficient frozen VLM alignment

Date: 2026-07-28

## Decision

Choose a dissertation-scale mechanism that can improve the efficiency of the
frozen DINOv3 ViT-S/16 + frozen text-encoder alignment system without merely
reproducing published encoder-compression work, while preserving the main
dissertation question and sub-5M trainable-parameter constraint.

## Falsifiable hypotheses

1. Static resolution reduction alone is heavily covered and is insufficiently
   novel unless used as a controlled baseline for a conditional mechanism.
2. The project-specific opportunity is asymmetric, retrieval-aware conditional
   computation: spend visual tokens only where they change alignment confidence.
3. A mechanism trained only around frozen encoders can recover most of the
   256-pixel retrieval performance while reducing measured end-to-end latency.
4. Parameter count is an inadequate efficiency proxy for this system; token
   count and modality-specific execution determine the practical frontier.

## Scope

- Primary sources and official implementations, mainly 2023–2026.
- Token reduction, conditional execution, routing, early exit, representation
  compression, quantization and retrieval-specific asymmetric execution.
- Exact overlap check against frozen SSL vision + frozen sentence encoder +
  lightweight cross-modal projector.
- Exclude wholesale backbone pretraining and research programs that cannot fit
  a dissertation-scale experimental phase.

## Required output

- Prior-art map with novelty risk.
- Ranked mechanisms with expected value, implementation scope and falsification
  test.
- A recommended minimal experiment and escalation path.
- Adversarial assessment and explicit limitations.

## Stop criteria

Stop searching when each recommended mechanism has at least three independent
source touchpoints or is explicitly labelled insufficiently triangulated, and
when no high-overlap paper is found by opposition queries.

## Risk register

- Apparent novelty may be only a renamed token-pruning or cascade method.
- Latency gains may disappear at batch 64 even when FLOPs fall.
- Dynamic routing overhead may exceed saved encoder work.
- Frozen-backbone accuracy loss may be irrecoverable with a small projector.
- Flickr30k-only clean evaluation limits generality.
