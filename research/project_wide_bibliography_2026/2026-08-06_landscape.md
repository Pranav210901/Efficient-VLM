# Aggregation-scoped literature: relevance-ranked bibliography and novelty assessment

**Run date:** 2026-08-06 · **Revised:** 2026-08-06 after user audit
**Corpus:** 54 ranked entries (38 inherited, 16 new) · **Genre:** landscape · **Depth:** deep

---

> ## ⚠ Corrections applied after audit — read first
>
> **1. One entry was a fabricated quotation.** Rank 20 (`arXiv:2601.09322`)
> carried a quoted passage and a derived technical warning for a source that was
> never fetched. The paper is about fusion **across ViT layers**, not spatial
> pooling instability. The entry is retracted (`sources/36`, §36d), the derived
> "check your seed variance" recommendation is withdrawn, and H1 now rests on
> three verified legs rather than four.
>
> **2. Quotes in this document are not uniformly retrieved.** Only rank 1 was
> fetched and quoted from source. The Flamingo passage at rank 10 is
> memory-sourced and formatted as if retrieved. Treat every quotation here as
> unverified unless explicitly marked otherwise, and re-derive before citing.
>
> **3. The title and scope of this document were wrong.** It was named
> "project-wide" but its search was scoped to token aggregation and text
> pooling. It does **not** assess the queue/drift work. See *Scope limitation*
> below — this is the most consequential defect, larger than the citation error.
>
> **4. Objection 3 assumed a null that did not occur.** Text aggregation
> *passed*: `M_T1` = 0.5448718 vs `historical_c4_mean` = 0.5386588, a **0.621pp**
> gain against a pooled-SD threshold of 0.00318 — **1.95σ**. Corrected below.
>
> **5. Parameter and latency wording tightened.** C4 contains a **1.25M token
> aggregator within a 2.73M trainable inference stack**; 2.73M is not the
> aggregator size. The **0.204 ms** delta is the pre-training/random-graph
> admission profile. The completed trained-weight profile measured **0.438 ms**
> against its contemporaneous CLS control, and is the value used for the final
> model. Both remain in the repository because they belong to different stages.

---

## Scope limitation (added post-audit)

**This document assesses the novelty of the token-aggregation and text-pooling
work only.** Its searches were scoped to those areas. It then generalised into a
claim about the whole project's contribution, which it had no corpus to support.

Absent entirely from this corpus, and therefore never assessed against prior art:

- queue staleness and cross-batch memory pathology
- projector drift / cross-modal feature staleness
- **modality-asymmetric falsification** — matched cumulative drift failing to
  predict matched degradation

That last item is plausibly the project's strongest asset and it was never
searched. Preliminary checks indicate MoCo, XBM and EfficientCLIP establish queue
staleness as known, but nothing was found on the modality asymmetry. **That is an
unverified gap, not an established one.**

**Required next action:** commission a separate landscape run scoped to queue
staleness, cross-batch memory, feature drift, and modality asymmetry in
contrastive alignment. That run — not this one — determines the publishable
claim. Either outcome (gap confirmed or gap killed) is worth more than anything
in this document.

---

## Executive summary

Three things this run establishes about **the aggregation work**, in order of
importance:

1. **The token-aggregation mechanism now in flight is not novel.** Learnable-query
   attention pooling over frozen encoder features is the SigLIP MAP head, the CoCa
   attentional pooler, Set Transformer PMA with k=1, and the subject of a June 2025
   paper that studies it under *the project's own accuracy-vs-parameter-efficiency
   framing*. Confidence: high, four independent source types.

2. **Routing was scoped out on measurement, not abandoned by drift.**
   *(Revised — the original "drift" framing was wrong.)* The 2026-07-28 landscape
   prescribed Stage A controls before any router. Those controls ran, and returned
   a redirecting result: the frozen image tower dominates full-stack latency
   (text = 1.49 ms of 10.75 ms). C4 adds a **1,252,225-parameter aggregator**
   inside a **2,730,628-parameter trainable inference stack**. Its pre-training
   admission profile added 0.204 ms; its completed trained-weight controlled
   profile added **0.438 ms**. The binding constraint is parameters, not latency,
   so the efficiency motivation for routing collapsed. See
   `findings/F4_scoping_decision.md`.

3. **The pooling block is not the contribution — but the protocol is not
   automatically the contribution either.** *(Revised.)* "No paper combines these
   five things" is novelty-by-conjunction, and H2 is only medium-confidence
   absence-of-evidence. The protocol is what *enabled* the findings; it is not
   itself the headline. The headline is whichever empirical finding survives — and
   the strongest candidate was never searched (see *Scope limitation*).

4. **The queue drift–damage dissociation is the novelty-bearing empirical
   contribution.** The completed aggregation result remains the empirical
   centrepiece because it is the largest clean positive gain, but attention
   pooling itself is crowded prior art. The separate queue chapter carries the
   defensible novelty claim: cumulative projector drift does not determine queue
   damage, and the more-damaged modality reverses with queue age. A 288-work
   forward-citation screen of XBM and CODER found no prior report of that
   per-modality matched-drift dissociation; CODER, the closest apparatus match,
   was ruled out at full-text level. This is a bounded prior-art claim, not proof
   of universal novelty. See `findings/F5_queue_claim_unoccupied.md`.

## Contribution hierarchy after the queue-specific search

1. **Empirical centrepiece:** learned patch-token aggregation produces the
   largest completed positive improvement under the frozen-backbone and latency
   budget. The block is established prior art; the result is not sold as a new
   pooling architecture.
2. **Novelty-bearing chapter:** the queue factorial and drift dissociation. The
   contribution statement should include:

   > The study shows that cumulative projector drift does not determine
   > cross-batch-memory damage: conditions matched on drift differ by 4.74pp,
   > and the modality suffering greater degradation reverses with queue age—a
   > dissociation not reported in prior cross-batch-memory work.

3. **Methods discussion:** batch-scaling, padding, fusion and FLOP-accounting
   corrections are protocol findings, not substitutes for either contribution.

---

## How "relevance" was defined

The request was "rank by relevancy," but relevance is only meaningful against a
decision. The decision here is *what must the dissertation cite and differentiate
from, and what threatens its claim*. So the ranking is by **threat-and-obligation**,
not topical similarity. A 2025 preprint that occupies your object outranks a
10,000-citation paper that is merely background.

Four tiers:

| Tier | Meaning | Count |
|---|---|---|
| **1** | Novelty-critical. Must cite, must differentiate. | 8 |
| **2** | Core method foundations. Must cite. | 14 |
| **3** | Baselines, controls, supporting evidence. | 21 |
| **4** | Methodology, framing, peripheral. | 11 |

Full machine-readable index: `sources.csv`.

---

## Tier 1 — Novelty-critical (ranks 1–8)

### 1. Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency
`arXiv:2506.10178` · June 2025 · *verified verbatim*

**The closest paper in the corpus.** It studies attention pooling over frozen
encoder features explicitly as an accuracy-vs-parameter-efficiency trade-off, finds
existing approaches "over-parameterized and computationally inefficient," and
proposes a lightweight multi-query cross-attention pooler that "eliminates
redundant projections and reduces the number of trainable parameters."

That is the `token_aggregator_scale` study's thesis, published fourteen months
earlier.

**What it does not cover, and where your remaining room is:** it is a
*classification probing* paper — not cross-modal retrieval, no text-side
counterpart, no latency gate, no pre-registration. Those four gaps are now
load-bearing for the contribution.

### 2. SigLIP 2 — MAP head
`arXiv:2502.14786`

SigLIP pools **both** towers with a MAP head: a learnable probe used as attention
query, plus LayerNorm and MLP. Compare to `model.py`:

| `LearnedQueryTextPool` | `SiglipMultiheadAttentionPoolingHead` |
|---|---|
| `self.query = nn.Parameter(...)` | learnable probe |
| `nn.MultiheadAttention(...)` | multihead attention |
| `nn.LayerNorm(...)` | layer normalization |
| `output_projection` | MLP |

Your additions — the sigmoid gate onto a masked-mean baseline, and
`key_padding_mask` exclusion of padding — are real differences, but they are
refinements to a standard block, not a new one.

### 3–6. Freeze-Align · SAIL · ShareLock · STRUCTURE
*(inherited, ranks unchanged from prior corpus)*

The direct competitor set for frozen-tower alignment. Prior finding stands: frozen
towers, MLP projectors and contrastive retrieval are **baselines, not your
algorithm**.

### 7–8. CoCa · Set Transformer
`arXiv:2205.01917` · `arXiv:1810.00825`

The provenance chain. Set Transformer's PMA with k=1 seed vector *is* your
single-learned-query pool. CoCa put it inside a contrastive image-text model in
2022.

---

## Tier 2 highlights (ranks 9–22)

**Sentence-BERT (`1908.10084`, rank 9)** matters more than its age suggests. It
reports mean pooling as the best simple strategy — which is the published basis for
your own pre-registered prediction that the text-side gain will be small. Your
preregistration already reasons this way ("MiniLM was pretrained with mean
pooling"); this is the citation for it.

**Flamingo (`2204.14198`, rank 10)** is the precedent for your gate. Flamingo
multiplies new-layer output by `tanh(α)` with α initialised to 0 so that "at
initialization, the model output matches that of the pretrained language model."
Your `sigmoid(−2.0) ≈ 0.119` is the same idea with a small non-zero start.

> **Worth noting:** because 0.119 ≠ 0, your aggregator is *not* an exact no-op at
> init. The artifact and the methodology both describe it as "near-no-op," which is
> accurate — but if exact baseline-equivalence matters to the fairness argument,
> a zero-init tanh gate would make it exact and is trivially defensible against
> this literature.

**Reproducible scaling laws (`2212.07143`, rank 13)** is your strongest defence on
the 26–38pp gap to OpenCLIP/MobileCLIP2/SigLIP2. That gap is substantially a
data-scale gap, and this paper quantifies the power law. Use it to pre-empt a
reviewer reading the gap as architectural failure.

**~~Attentive Multi-Layer Fusion (`2601.09322`, rank 20)~~ — RETRACTED.**
This entry asserted a quoted warning about instability when pooling over hundreds
of patch tokens. The quote was fabricated; the source was never fetched and is
about cross-layer fusion, not spatial pooling. The derived recommendation to check
seed variance is withdrawn — measured SD is 0.328pp across three seeds, which is
tight and raises no concern. See `sources/36` §36d.

---

## Hypothesis assessment

| # | Hypothesis | Verdict | Confidence |
|---|---|---|---|
| H1 | Learned-query pooling is established, not novel | **Confirmed** | High — 3 source types (was 4; one retracted) |
| H2 | No paper occupies the exact protocol setting | **Not falsified** | Medium — absence of evidence |
| H3 | Gate-at-−2.0 has clear precedent | **Confirmed** | High |
| H4 | ~~Mechanism drifted from routing to aggregation~~ | **Refuted** | Repo + timestamp evidence |

**H4 refuted.** Experiment timestamps show the prescribed Stage A controls ran in
order, and returned a measurement that removed routing's motivation. This was a
scoping decision on evidence, not drift. Rewritten as
`findings/F4_scoping_decision.md`.

**H2 is the weakest surviving claim** — absence-of-evidence across three search
rounds, not a novelty guarantee. It should not carry the contribution.

**Untested hypothesis, and the important one:** no hypothesis in this run
addressed queue staleness or modality-asymmetric drift. See *Scope limitation*.

---

## Adversarial pass

**Objection 1 — "This is the SigLIP MAP head with a gate."**
The strongest objection, and largely correct at the block level. Survivable only if
the claim moves to the protocol: pre-registered budget-gated selection over a
frozen heterogeneous pair, with latency as a hard gate. If the write-up leads with
the module, this objection wins.

**Objection 2 — "The attentive-probing paper already did the efficiency study."**
Partly correct. The defence is the setting: classification probing on a single
modality is not bidirectional cross-modal retrieval, and parameter count is not
latency. Both defences must be *demonstrated*, not asserted — which means the
latency gate needs to do visible work in the results.

**Objection 3 — ~~"You are reporting a null result."~~ CORRECTED.**
The original text reasoned about text aggregation *failing* to beat masked mean.
It did not fail. `M_T1` reaches 0.5448718 against `historical_c4_mean` 0.5386588
— a **0.621pp gain**, clearing the pooled-SD threshold of 0.00318 at **1.95σ**.
The pre-registered-null contingency did not fire.

The live objection is different and narrower: **does that margin survive the
sealed test?** The gain was measured under Flickr-validation epoch selection, and
the arm report discloses the double-selection exposure explicitly
(`selection_gain_pp = 0.128`). A 1.95σ validation margin with acknowledged
selection pressure is not a safe margin. That, not a null, is what needs
defending.

**Objection 4 — "Why these nine failure categories?"**
`probe1_error_decomposition` defines its taxonomy ad hoc. `arXiv:2407.15239` has a
published taxonomy for image-text retrieval granularity. Align to it, or justify
the divergence explicitly.

**Counter-evidence actively sought and not found:** no paper combining frozen
heterogeneous encoders + dual-tower learned aggregation + pre-registered latency
gating. Searched via opposition queries; see `plan.md`.

---

## Recommended actions *(revised post-audit)*

1. **Commission the queue/drift landscape run.** Highest value action in this
   document. Scope: queue staleness, cross-batch memory, feature drift, modality
   asymmetry in contrastive alignment. It decides the publishable claim; this
   document cannot.
2. **Add Tier 1 citations to related work now.** Ranks 1, 2, 7, 8 uncited is a
   genuine viva liability and each takes minutes. Unambiguous, do it first.
3. **Reframe away from the module — but do not land on the protocol.**
   Cite MAP/PMA/CoCa in the aggregation chapter and present the block as
   engineering under a budget. The protocol is what *enabled* the finding, not the
   finding itself. Novelty-by-conjunction is a weak headline.
4. **Keep aggregation; document routing as a scoping decision.** Building a
   conditional router now reopens a closed architecture search, needs its own
   preregistration, and yields a thin final chapter. Write it as "routing was
   scoped out under the measured latency and parameter budget; static aggregation
   was selected instead," and place routing in future work beside the
   queue-asymmetry probes.
5. **Re-derive every quotation in this document from fetched text, or delete it.**
   Only rank 1 is currently safe.
6. **Re-run the Tier 1 search before submission.** H2 is the claim most likely to
   decay.

---

## Method notes and limitations

- **Existing-work check honoured.** 38 sources inherited from two prior folders,
  re-ranked rather than re-searched. Search budget spent only on the seven areas
  the code entered after 2026-07-28.
- **Sub-agent fan-out was not used.** This session's configuration prohibits
  spawning agents, so search rounds ran inline in parallel batches. Coverage is
  unaffected; wall-clock time was longer than a fan-out run.
- **Verification is uneven and recorded per-row.** Only rank 1 was fetched and
  quoted verbatim from its source page. Most new entries are
  `search_summary` — the arXiv IDs all appeared in live search results, but
  abstracts were not individually fetched. **Verify any ID before it enters a
  submitted bibliography.**
- **Perceiver / Perceiver-IO / BLIP-2 / DINOv2 / ToMe** are referenced by name in
  `sources/36` but their arXiv IDs were *not* confirmed in this run and are
  deliberately absent from `sources.csv`.
