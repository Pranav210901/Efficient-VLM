# Adaptive Cross Batch Normalization

- Paper: *Adaptive Cross Batch Normalization for Metric Learning*
- Year/status: 2023 arXiv preprint
- URL: https://arxiv.org/abs/2303.17127
- Verification: full paper source inspected.
- Verified evidence: identifies distribution mismatch between accumulated and
  current embeddings, explicitly notes that large memory relative to small
  batches accumulates drift, sweeps batch size at fixed memory and memory size
  at fixed batch, and corrects drift through moment matching.
- Short excerpt: “ensure that the accumulated embeddings are up to date.”
- Relevance: direct prior art on capacity/batch interactions and stale-memory
  correction. It also reports cases where XBM is worse than no memory. The
  remaining difference is the dual-frozen, bidirectional, modality-asymmetric
  image-text setting.
