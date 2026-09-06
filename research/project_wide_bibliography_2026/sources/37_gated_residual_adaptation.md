# 37 — Zero/near-zero initialised residual gating

Grouped source. Establishes precedent for the project's
`gate = nn.Parameter(torch.tensor(-2.0))` → `gate.sigmoid()` construction used
in `LearnedQueryPatchPool`, `LearnedQueryTextPool`, `SpatialTokenAdapter` and
`ResidualProjectionHead`.

- **Category:** method foundation — training-stability primitive
- **Verification level:** `search_summary`
- **Credibility:** 5 · **Recency:** mixed · **Bias:** 2

## 37a — Flamingo (arXiv 2204.14198)

> "Flamingo uses a tanh-gating mechanism that multiplies the output of newly
> added layers by tanh(α) before adding it to the input representation from the
> residual connection, where α is a layer-specific learnable scalar initialized
> to 0. This ensures that at initialization, the model output matches that of
> the pretrained language model, improving training stability and final
> performance."

This is the closest precedent and the same argument the project makes: a new
module bolted onto a frozen pretrained backbone must start as a no-op.

## 37b — CaiT / Going deeper with Image Transformers (arXiv 2103.17239) — LayerScale

> "LayerScale is a learnable element-wise gating mechanism that adaptively
> scales skip connections, using a zero-initialized per-token and per-channel
> scaling tensor to control residual flow and ensure gradual information
> integration and stable optimization. LayerScale initializes all coefficients
> to 0, which resembles ReZero but with distinct learnable parameters for each
> channel."

## 37c — LLaMA-Adapter (arXiv 2303.16199), ICLR 2024

Zero-initialised attention with a learnable gating factor, same motivation.

## Difference in the project's variant

The project uses `sigmoid(−2.0) ≈ 0.1192`, i.e. a **small but non-zero** start,
rather than the exactly-zero start of Flamingo/LayerScale/ReZero. That is a
minor design difference, not a novel mechanism, and should be described as an
implementation choice with a citation to this lineage rather than as a
contribution.

**H3 confirmed, high confidence** — the construction has clear precedent.

## Caveat worth noting in the write-up

A gate initialised at 0.12 rather than 0.0 means the aggregator is *not* an
exact no-op at initialisation. The claim "training begins from the baseline's
behaviour" is approximate. If exact baseline-equivalence at init matters for the
fairness argument, initialising the gate at a large negative value (or 0 under
tanh) would make it exact and is trivially defensible against this literature.
