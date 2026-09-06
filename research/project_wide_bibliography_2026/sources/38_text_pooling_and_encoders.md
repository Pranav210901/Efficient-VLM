# 38 — Text pooling strategy and text-encoder choice

Grouped source supporting the `text_aggregation_study` M/E arms.

- **Category:** direct baseline justification
- **Verification level:** `search_summary`
- **Credibility:** 5 · **Recency:** mixed · **Bias:** 2

## 38a — Sentence-BERT (arXiv 1908.10084)

Establishes the masked-mean baseline the project's T0 cells use, and reports
that mean pooling is the strongest simple choice:

> "SBERT uses three types of pooling strategies: CLS ... Mean ... and Max."

> "The mean-pooling approach was best performing for both NLI and STSb datasets."

> "When trained with the regression objective function, the pooling strategy has
> a large impact, with the MAX strategy performing significantly worse than MEAN
> or CLS-token strategy."

**Directly supports the project's pre-registered hypothesis** that the text-side
gain will be small: MiniLM was *trained* with mean pooling, so replacing its
pooling at alignment time fights the pretraining objective. The preregistration
already states this — this is the citation for it.

## 38b — Comparative Analysis of Pooling Mechanisms in LLMs (arXiv 2411.14654)

Recent systematic comparison of pooling mechanisms; use for the claim that
pooling choice is empirically consequential and not settled.

## 38c — Comparison and Combination of Sentence Embeddings from Different Supervision Signals (arXiv 2202.02990)

Supporting evidence on pooling/supervision interaction.

## 38d — E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training (arXiv 2212.03533)

The provenance of the project's `e5_small_v2` arm and its `"passage: "` prefix.

> "E5 is a family of state-of-the-art text embeddings trained in a contrastive
> manner with weak supervision signals from a curated large-scale text pair
> dataset called CCPairs."

> "The first model to outperform the strong BM25 baseline on BEIR without
> labeled data, and when fine-tuned, obtained the best results on MTEB, beating
> existing embedding models with 40x more parameters."

**Relevance:** justifies why E5 is a reasonable challenger to MiniLM in the
between-encoder comparison, and why the prefix handling in
`TEXT_PREFIX_DEFAULTS` is required rather than cosmetic.

## 38e — Improving Text Embeddings with Large Language Models (arXiv 2401.00368)

E5 follow-up lineage; relevant only if the encoder sweep is extended.
