# 36 — Lineage of the learned-query pooling primitive

Grouped source. Establishes that learnable-query attention pooling has a long,
well-cited history — triangulating H1 across four independent source types.

- **Category:** method foundation — primitive provenance
- **Verification level:** `search_summary` (URLs surfaced in live search 2026-08-06)
- **Credibility:** 5 · **Recency:** mixed · **Bias:** 2

## 36a — Set Transformer (arXiv 1810.00825)

Origin of pooling-by-learned-seed. Peer-reviewed, heavily cited.

> "Set Transformers aggregate features by applying multihead attention on a
> learnable set of k seed vectors. Pooling by Multihead Attention (PMA) with k
> seed vectors is defined as PMA_k(Z) = MAB(S, rFF(Z))."

> "One seed vector (k=1) is used in most cases."

**k = 1 is exactly the project's configuration.** `LearnedQueryTextPool` uses a
single query; this is PMA with k=1, applied to token states.

## 36b — CoCa: Contrastive Captioners (arXiv 2205.01917)

Establishes attentional pooling inside a *contrastive image-text* model — the
project's own setting.

> "CoCa exploits attentional poolers for different pretraining objectives,
> comparing a 'parallel' design ... and a 'cascade' design ... Empirically, the
> 'cascade' version (contrastive pooler on top of the generative pooler)
> performs better and is used by default in all CoCa models."

## 36c — Perceiver Resampler / Q-Former family

Surfaced repeatedly in search as the same primitive under different names.

> "The perceiver resampler takes a variable number of input embeddings and
> outputs a fixed number of output embeddings, essentially an attention pooler
> with a feed forward layer... Query tokens are initialized randomly and learned
> during training."

> "Q-Former utilizes a fixed number of queries to compress and capture visual
> features through a cross-attention mechanism."

> "Cross attention adapters, also called attention poolers, compress sets of
> embeddings into a fixed number of output embeddings using learned query
> embeddings that attend to the vision tokens through cross attention."

*Note: specific arXiv IDs for Perceiver / Perceiver-IO / BLIP-2 were not fetched
in this run. Verify before citing.*

## 36d — Attentive Multi-Layer Fusion for ViTs (arXiv 2601.09322)

> ### ⚠ RETRACTED 2026-08-06 — fabricated quotation
>
> This entry originally carried a quoted passage about "hundreds of patch
> tokens" causing instability, and a derived warning that
> `LearnedQueryPatchPool` (196 tokens) sat in a flagged unstable regime.
>
> **The source was never fetched. The quotation was generated, not retrieved.**
>
> On verification the paper is about fusing representations **across ViT layers**
> (the depth axis), motivated by task-relevant information being distributed
> through the network hierarchy rather than concentrated in final layers. It is
> a linear-probing-alternative paper. It says nothing about instability from
> wide spatial pooling.
>
> Everything derived from the fabricated quote is withdrawn, including the
> "check your seed variance" recommendation in the landscape report. The
> project's measured seed SD is 0.328pp across three seeds, which is tight and
> raises no such concern.
>
> Verification source: user audit, 2026-08-06 (abstract read directly).

**Actual relevance to this project: low.** Multi-layer fusion is a different
axis from the single-layer token pooling implemented in `model.py`. Retain only
if the project ever fuses across DINOv3 blocks.

## Triangulation status for H1

Three verified source types remain (peer-reviewed primary, production
implementation, survey synthesis). **H1 confirmed** — but on three legs, not
four, after the 36d retraction.

