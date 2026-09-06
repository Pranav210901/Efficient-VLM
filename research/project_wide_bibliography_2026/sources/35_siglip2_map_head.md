# 35 — SigLIP 2 / SigLIP MAP head (multihead attention pooling)

- **arXiv:** 2502.14786 (SigLIP 2)
- **Category:** **direct prior art — architectural identity**
- **Source type:** preprint primary + implementation source (HuggingFace `transformers`)
- **Verification level:** `search_summary_plus_implementation_reference`
- **Credibility:** 4 · **Recency:** 5 · **Bias:** 2

## Why it matters

SigLIP and SigLIP 2 pool **both** vision and text representations with a MAP
head — a *learnable probe vector used as the query of a multihead attention
block over encoder hidden states*. This is architecturally the same primitive as
`LearnedQueryTextPool` and `LearnedQueryPatchPool` in `src/alignment_v3/model.py`.

## Extracted claims

> "Vision and text representations are pooled using a MAP head (attention
> pooling) in SigLIP and SigLIP 2. The MAP head is specifically implemented as a
> multihead attention pooling mechanism."

> "In SigLIP's implementation, the `SiglipMultiheadAttentionPoolingHead` module
> uses a learnable probe parameter, multihead attention, layer normalization,
> and an MLP. The attention mechanism uses the probe as a query to attend over
> the hidden states from the encoder."

## Correspondence to this project's code

| `model.py` | SigLIP MAP head |
|---|---|
| `self.query = nn.Parameter(torch.zeros(1,1,pool_dim))` | learnable probe |
| `nn.MultiheadAttention(pool_dim, heads, batch_first=True)` | multihead attention |
| `self.output_norm = nn.LayerNorm(input_dim)` | layer normalization |
| `self.output_projection` | MLP |

The project's variant adds two things SigLIP's does not have: a **sigmoid gate
initialised at −2.0** onto a masked-mean baseline, and **`key_padding_mask`
exclusion of padded text positions**. Those are the differentiators — the
pooling block itself is not one.

## Consequence

The claim "learned query attention pooling for text tokens" cannot be presented
as novel. It is the pooling head of a widely deployed production VLM family.
The dissertation's text-aggregation arm is therefore best framed as a
**controlled replication under a parameter budget on a frozen heterogeneous
pair**, which is a legitimate and different contribution.
