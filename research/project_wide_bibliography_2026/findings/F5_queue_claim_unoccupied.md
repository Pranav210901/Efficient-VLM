# F5 — The drift-insufficiency claim is unoccupied

**Run:** 2026-08-06, targeted search round (4 queries + 2 verification fetches).
**Status:** claim **not falsified**. Confidence: **medium** (absence of evidence,
single search round).

## The claim under test

> Cumulative projector drift is *insufficient* to explain memory-queue damage in
> cross-modal contrastive alignment, and the residual factor is modality-dependent
> with an age-dependent sign.

## Dissertation position

This finding is not a generic negative or a methodological footnote. It warrants
its own queue chapter and an explicit line in the dissertation contribution
statement:

> The study shows that cumulative projector drift does not determine
> cross-batch-memory damage, with the modality that suffers most reversing
> between queue ages—a dissociation not reported in prior cross-batch-memory
> work.

Token aggregation remains the empirical centrepiece because it provides the
largest clean positive gain. The novelty hierarchy is different: attentive token
pooling is established, whereas the per-modality matched-drift dissociation was
not found in the screened queue literature.

Internal evidence (`results/queue_factorial/interpretation/`):
image-only age 16 (drift 0.08613 → 13.08pp degradation) vs text-only age 64
(drift 0.08606 → 17.82pp). Matched drift, 4.74pp gap, ~20× seed SD.
`matched_drift_claim: GENERAL` at 4/4 tolerances; `seed_replication: HIT`.

## The kill criterion

A paper that measures drift **and** degradation **per modality** across queue ages
and reports the dissociation. Not met by anything found.

## Closest candidates, both verified and ruled out

**CSMCIR — CoT-Enhanced Symmetric Alignment with Memory Bank for CIR**
`arXiv:2601.03728` · Jan 2026, rev. May 2026 · *abstract fetched*

The nearest miss. Explicitly identifies memory-bank staleness with age
("representations become misaligned with current batch representations as the
model states evolve") and proposes an entropy-based temporally dynamic memory bank.

Ruled out because: (a) task is **composed** image retrieval, not bidirectional
dual-encoder retrieval; (b) drift is **not measured as a separate quantity** from
degradation; (c) **no image-side vs text-side memory bank comparison**; (d) no
dissociation reported. It treats staleness as a problem to fix, not as an
explanation to test.

**Rethinking Deep Contrastive Learning with Embedding Memory**
`arXiv:2103.14003` · Zhang, Wang, Huang, Scott, 2021 · *abstract fetched*

Ruled out: single-modality metric learning on visual retrieval; no drift
measurement; no modality comparison.

## The finding's ideal target: XBM's slow-drift assumption

Cross-Batch Memory (CVPR 2020) justifies memory banks with an explicit empirical
claim:

> "The embedding of an instance actually drifts at a relatively slow rate
> throughout the training process, suggesting that deep features of a mini-batch
> computed at past iterations can considerably approximate those extracted by the
> current model."

**This is the assumption the queue_factorial result falsifies the sufficiency of.**
Slow drift is offered as the reason stale entries are safe. The result shows that
in the cross-modal dual-encoder regime, drift magnitude does not determine damage
— two conditions at identical drift differ by 4.74pp, and which modality suffers
more flips with queue age.

That gives the contribution a named, canonical, well-cited target rather than a
free-floating novelty claim. Position it as a boundary condition on XBM's
justification, not as a refutation of XBM.

Also relevant, not occupying: XBM's own image-text ablations vary **queue size and
batch size against performance** — never drift against degradation, never
per-modality.

---

# Forward-citation sweep — 2026-08-06, round 2

Ran 4 additional queries deliberately outside "queue" vocabulary (modality gap,
asymmetric encoder dynamics, momentum staleness) on the theory that an occupant
might not use memory-bank language. **Claim still unoccupied.** Confidence raised
from medium to **medium-high**.

## The decisive ruling-out: CODER

**CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for
Image-Text Retrieval** · `arXiv:2208.09843` · ECCV 2022 · *full paper fetched*

This is the closest structural match that exists: **dual dynamic
modality-specific memory banks** for bidirectional image-text retrieval. It has
exactly the apparatus needed to find this result.

Checked against all four criteria, from the experimental sections:

| Criterion | CODER |
|---|---|
| Measures drift/staleness as a numeric quantity separate from accuracy | **No** — Recall@K only |
| Ablates image bank vs text bank separately | **No** — component-level ablations only |
| Varies bank age/size with per-modality degradation | **No** |
| Matches drift across modalities and compares degradation | **No** |

**This strengthens the contribution rather than merely failing to kill it.** The
field had the apparatus — modality-specific memory banks in cross-modal retrieval,
published at ECCV — and did not ask whether the two banks fail differently, or
whether drift explains the failure. The question was available and went unasked.

## Also ruled out this round

- **AsCL** (`arXiv:2405.10029`) — "asymmetry-sensitive" but concerns *semantic*
  asymmetry types between modalities, not queue staleness dynamics. Different object.
- **CSMCIR**, **Rethinking Deep Contrastive Learning** — ruled out in round 1.

## NEW RISK — the modality gap is a rival explanation

The sweep surfaced a substantial literature on the **modality gap** in CLIP-style
models (Liang et al.; "Understanding the Modality Gap in CLIP", OpenReview;
`arXiv:2406.17639`; `arXiv:2502.04263`). Established findings:

> Each modality encoder constrains its embeddings to a narrow cone; at
> initialisation the average intra-modal cosine similarity is very high, and the
> cones are highly separable across random initialisations. The contrastive
> objective preserves and worsens this separation.

**The objection this enables:** *"Your modality-dependent damage is just the known
modality gap. Image and text embeddings live in differently-shaped regions, so of
course equal drift magnitude means different things in each — you have
re-described cone geometry, not discovered a new phenomenon."*

This is now the strongest objection to the **interpretation** of the finding. It
does not threaten the measurement — the dissociation is real either way — but it
threatens the claim that a *new* modality-specific factor is required.

**Pre-empt it.** Two options, cheap to expensive:
1. Cite the modality-gap literature and argue the **age-dependent sign flip** is
   not explicable by static cone geometry. A fixed geometric difference predicts a
   fixed ordering; yours reverses between age 4 and age 16. That is the strongest
   available defence and it costs nothing but the argument.
2. If time permits, measure intra-modal cosine spread per tower and show the
   dissociation survives normalising drift by it.

Option 1 alone is probably sufficient and uses the `modality_asymmetry: MISS`
result as an asset rather than a blemish.

## Confidence caveats

- Two search rounds, six targeted queries, three full verifications. Still absence
  of evidence rather than proof of absence — but the closest-apparatus paper
  (CODER) was checked at full-text level, which is the strongest single check
  available short of an exhaustive citation graph.
- **Unverified claim, do not cite:** a search summary attributed to
  `arXiv:2303.11866` (Khan & Fu, ICLR 2023) a finding that alignment changes
  normalisation parameters most in the *shallowest text-encoder layers* and the
  opposite for the image encoder. **The abstract fetch did not confirm this.** If
  that finding is real it is directly supportive of encoder-side asymmetry, but it
  must be verified in the full PDF before use.
---

# Citation-graph sweep — 2026-08-06, round 3

**Claim still unoccupied.** Confidence raised to **high (bounded)** — see the
coverage caveat, which is genuine and matters.

Semantic Scholar hard-rate-limited this host (8 backoffs to 90s, all HTTP 429).
Ran the sweep on **OpenAlex** instead — same citation graph, open public tier.
Script: `scratchpad/oa_sweep.py`.

## Method

Screened every work citing **XBM** (`W3035014997`, CVPR 2020) and every work
citing **CODER** (`W4312934263`, ECCV 2022 — the closest-apparatus paper), by
three-stage vocabulary match over title + abstract:

1. cross-modal / image-text / vision-language
2. memory bank / queue / cross-batch / momentum contrast
3. drift / staleness / asymmetry / modality-specific / modality gap

Tier 1 = all three stages matched.

## Result

| Target | Citing works screened | Tier 1 hits |
|---|---|---|
| XBM (CVPR 2020) | 260 | **1** |
| CODER (ECCV 2022) | 28 | **0** |
| **Total** | **288** | **1** |

**The single Tier-1 hit is out of domain.** "Learning Memory-Augmented
Unidirectional Metrics for Cross-modality Person Re-identification" (CVPR 2022)
— memory bank + modality gap + modality-specific, but the modality pair is
**visible/infrared imagery** and the task is person re-identification. No text
encoder, no caption retrieval, no dual-encoder image-text space. Ruled out on
domain, from title and abstract; not fetched at full text.

**Nobody extended CODER's dual memory banks into drift analysis.** Of its 28
citers, the nearest are USER (momentum contrast + queue for image-text, no drift
vocabulary) and ReAL (memory bank for image-text negatives, no drift vocabulary).
Both use the apparatus; neither interrogates it.

## Coverage caveat — read this before claiming exhaustiveness

Two real limits, and the second is the one that could bite:

1. **OpenAlex reports 253 citations for XBM.** Google Scholar reports
   substantially more. OpenAlex's coverage of CS conference proceedings is
   incomplete, so this is systematic over OpenAlex's index, **not exhaustive over
   all literature**.

2. **Abstract-level keyword screening has a known false-negative mode — and this
   project already hit it.** CODER does *not* use drift or staleness vocabulary in
   its abstract. It was found by structural reasoning ("dual modality-specific
   memory banks") and ruled out only by reading its full experimental section. A
   paper that runs this analysis in an unheadlined ablation would pass straight
   through this filter.

   Mitigation already in place: the round-2 manual search deliberately used
   non-queue vocabulary, and the one paper with the right apparatus was read in
   full. But the residual risk is real and should be stated in the thesis rather
   than hidden.

## Standing conclusion

Across three rounds — 6 targeted manual searches, 288 systematically screened
citing works, and 4 full-text or abstract verifications — no paper measures
projector drift separately from degradation, per modality, across queue ages, in
cross-modal dual-encoder retrieval.

**The claim is defensible.** State it with the coverage caveat attached, not as an
absolute priority claim.

## Consequence

This is the strongest available contribution. Aggregation, protocol and the
efficiency negative become supporting apparatus around it.

Related: [[F1_aggregation_not_novel]], [[F2_no_exact_occupant]], [[F4_scoping_decision]]
