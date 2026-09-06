# 34 — Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency

- **arXiv:** 2506.10178
- **Submitted:** 2025-06-11
- **Authors:** Bill Psomas, Dionysis Christopoulos, Eirini Baltzi, Ioannis Kakogeorgiou, Tilemachos Aravanis, Nikos Komodakis, Konstantinos Karantzalos, Yannis Avrithis, Giorgos Tolias
- **Category:** **direct prior art — novelty-critical**
- **Source type:** preprint primary
- **Verification level:** `abstract_fetched_verbatim` (arxiv.org/abs fetched 2026-08-06)
- **Credibility:** 4 · **Recency:** 5 · **Bias:** 2

## Why this is the single most relevant paper in the corpus

It occupies the *same object* as the current `text_aggregation_study` and
`token_aggregator_scale_training` work: attention-based pooling over frozen
encoder features, evaluated explicitly on an **accuracy vs. parameter-efficiency
trade-off**. That is the project's own framing, published a year earlier.

## Verbatim abstract

> "As fine-tuning becomes impractical at scale, probing is emerging as the
> preferred evaluation protocol. However, standard linear probing can understate
> the capability of models whose pre-training optimizes local representations
> rather than an explicit global representation. This motivates attentive
> probing, an alternative that uses attention to selectively aggregate
> patch-level features. Despite growing adoption, attentive probing is still
> underexplored: existing approaches are often over-parameterized and
> computationally inefficient. In this work, we revisit attentive probing
> through the lens of the accuracy vs. parameter-efficiency trade-off. We
> present the first comprehensive study of existing methods, analyzing their
> design choices and benchmarking their performance. Building on these insights,
> we propose efficient probing (EP), a lightweight yet effective multi-query
> cross-attention mechanism that eliminates redundant projections and reduces
> the number of trainable parameters. Across multiple benchmarks and
> pre-training paradigms, EP consistently outperforms linear probing and
> previous attentive probing methods, and remains effective when combined with
> parameter-efficient fine-tuning. Beyond evaluation, our analysis uncovers
> emerging properties of EP, including complementary attention maps, which open
> new directions for leveraging probing beyond protocol design."

## Direct collisions with this project

| Project element | Paper's equivalent |
|---|---|
| `LearnedQueryPatchPool` (1 query, cross-attn, no patch↔patch) | "multi-query cross-attention mechanism that eliminates redundant projections" |
| Parameter budget as the gating criterion | "accuracy vs. parameter-efficiency trade-off" |
| Baseline = CLS passthrough / masked mean | "consistently outperforms linear probing" |
| Frozen SSL backbone (DINOv3) | "models whose pre-training optimizes local representations" |

## What it does *not* cover

- Cross-modal / contrastive retrieval — it is a **classification probing**
  paper. The project's setting is bidirectional image-text retrieval.
- A *text-side* aggregation counterpart.
- Latency (as opposed to parameter count) as a hard admission gate.
- Pre-registered hierarchical selection with a frozen fallback order.

These four are where any remaining novelty must live.

## Action required

Must be cited and explicitly differentiated. A dissertation that presents
learned-query pooling over frozen features as a novel efficiency mechanism
without engaging this paper is vulnerable in viva.
