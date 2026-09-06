# F1 — The token-aggregation mechanism is not novel

**Status:** H1 confirmed. Confidence: **high** (4 independent source types).

Learnable-query cross-attention pooling over encoder token states is an
established primitive with a decade-long lineage:

- **Set Transformer (2018)** — PMA with k learned seed vectors; k=1 is the
  project's exact configuration.
- **CoCa (2022)** — attentional poolers inside a contrastive image-text model.
- **Perceiver Resampler / Q-Former (2021–2023)** — the same block under other
  names, for compressing variable token sets to fixed size.
- **SigLIP / SigLIP 2 (2023–2025)** — a MAP head using a *learnable probe as
  attention query, plus LayerNorm and MLP*, applied to **both towers**. This is
  a line-for-line architectural match to `LearnedQueryTextPool`.
- **Attentive probing literature (2025–2026)** — studies this exact block on
  frozen features under a parameter-efficiency criterion.

## Consequence for the dissertation

`LearnedQueryPatchPool` and `LearnedQueryTextPool` are **implementation choices,
not contributions**. Presenting them as the mechanism is not defensible.

The aggregation chapter may still report the following methodological strengths,
but they should **not displace the queue dissociation as the novelty-bearing
claim**:

1. Pre-registered, hash-chained, **budget-gated selection** among aggregation
   variants with a frozen fallback hierarchy fixed before results are seen.
2. **Latency as a hard admission gate** (q3, not median) rather than parameter
   count alone — the attentive-probing literature gates on parameters only.
3. The **frozen heterogeneous pair** setting (self-supervised vision encoder +
   independently pretrained sentence encoder), where the probing literature uses
   single-modality classification.
4. A disciplined **null result**: reporting that text-side aggregation does *not*
   pay on short captions is publishable given the pre-registration, and is
   directly predicted by Sentence-BERT's mean-pooling finding.

The empirical aggregation gain remains a centrepiece result. Its architecture is
not presented as novel. The separate queue chapter carries the stronger novelty
statement; see [[F5_queue_claim_unoccupied]].

Related: [[F2_no_exact_occupant]], [[F4_mechanism_drift]]
