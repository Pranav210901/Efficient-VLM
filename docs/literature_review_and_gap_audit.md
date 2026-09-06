# Literature review and gap audit

> **Superseded for current status.** This audit predates the authorised
> Flickr30k final-test run and is preserved as a dated decision record. See
> `docs/literature_gap_closure_20260810.md`, `docs/literature_review.tex`, and
> `docs/completion_status.md` for the current submission-facing account.

**Compiled:** 2026-08-10 · **Status:** integrated into `dissertation.md`; retained as the detailed audit trail
**Corpus:** 54 ranked local entries across three search rounds (2026-07-26,
2026-07-28, 2026-08-06), a 288-work citation screen, and a 20-query refresh with
322 raw records (273 unique in the targeted set)
**Scope:** what the literature establishes, which gaps this project has closed,
which remain open, and which of those are closeable with the assets that already
exist.

The ranked 25-gap disposition and comparison hierarchy are in
[`literature_gap_closure_20260810.md`](literature_gap_closure_20260810.md). That
document supersedes stale status labels in this historical narrative where they
conflict.

---

## ⚠ Read before citing anything in this document

This review is assembled from the project's own corpus in
[`docs/research/alignment_queue_prior_art/`](research/alignment_queue_prior_art/),
[`research/efficient_frozen_vlm_2026/`](../research/efficient_frozen_vlm_2026/) and
[`research/project_wide_bibliography_2026/`](../research/project_wide_bibliography_2026/).
Three inherited defects carry forward and must be respected:

1. **Verification is uneven.** Each source carries a `verification_level` in its
   `sources.csv`. Values range from `full_text_and_supplement` (safe) through
   `primary_abstract` down to `search_summary` (arXiv ID appeared in a search
   result; abstract never fetched). **Anything marked `search_summary` must be
   fetched and verified before it enters a submitted bibliography.**
2. **One entry was fabricated and retracted.** Rank 20 (`arXiv:2601.09322`,
   "Attentive Multi-Layer Fusion") carried a quoted passage for a source that was
   never fetched, and the paper concerns cross-layer fusion, not spatial pooling.
   It is retracted and excluded here.
3. **Some arXiv IDs were never confirmed.** Perceiver, Perceiver-IO, BLIP-2,
   DINOv2 and ToMe are referred to by name in the corpus but their identifiers
   were deliberately left out of `sources.csv`. They are named in this review as
   lineage, not cited with identifiers.

All novelty statements below are **bounded prior-art claims over a screened
corpus**, never proofs of absence.

---

## 1. What the literature establishes

### 1.1 Frozen unimodal alignment is an established method, not a contribution

This is the most important framing fact in the review, and it constrains
everything the project may claim architecturally.

| Work | Venue | What it establishes | Verification |
|---|---|---|---|
| **LiT** | 2022 | Locked image tower + trained text tower; the precursor to freezing at all | `primary_abstract` |
| **ASIF** | 2023 | Alignment without training, via relative representations | `primary_abstract` |
| **dino.txt** | 2024 | Text alignment onto a frozen self-supervised DINO vision tower — the closest vision-side analogue to this project | `primary_abstract` |
| **Freeze-Align** (Maniparambil et al.) | CVPR 2025 | Frozen independently pretrained encoders + simple MLP projectors + pair-compatibility analysis + retrieval evaluation | `selected_fulltext_sections` |
| **SAIL** (Zhang et al.) | CVPR 2025 | Frozen towers, trainable alignment layer only, refined sigmoid loss, multiple positive captions, pre-encoded features, batch 32,768 | `full_text_and_supplement` |
| **STRUCTURE** | NeurIPS 2025 | Frozen unimodal alignment with geometry preservation under limited paired data | `primary_abstract` |
| **ShareLock** | TMLR 2026 | Frozen backbones, lightweight alignment, reusable precomputed features | `primary_abstract` |
| **SOTAlign** | arXiv 2026 | Semi-supervised frozen alignment using paired and unpaired data | `primary_abstract` |
| **EBind** | 2025 | Frozen encoders + projectors for multi-space binding (peripheral) | `search_summary` |

**Consequence.** Frozen towers, MLP projectors, contrastive retrieval training and
pair-compatibility screening are **baselines and implementation choices**, not this
dissertation's algorithm. This was settled in
[`F1_architecture_prior_art.md`](research/alignment_queue_prior_art/findings/F1_architecture_prior_art.md)
at high confidence and has not been disturbed by any later round.

Theoretical backing for why aligning independently trained encoders works at all
comes from the **Platonic Representation Hypothesis** (2024) and *Do Vision and
Language Encoders Represent Similarly?* (2024) — both used here as justification
for encoder-pair selection, not as contributions.

### 1.2 Contrastive objectives and negative handling are mature

CLIP-style InfoNCE, **SigLIP**'s sigmoid pairwise loss (2023), **UniCL** (2022) and
**SupCon** (2020) multi-positive formulations keyed on image identity, **FFF**
(2024) on loss and caption handling, **CLIP-Lite** (2023) on negative efficiency,
and **Hard Negative Mixing** (2020) on synthesising negatives in embedding space
all predate this work. The project implements several of these as *arms of a
factorial*, which is the correct posture: the loss family is a controlled variable,
not a claim.

This matters for interpreting **R01 (Wave 0)**: the finding that loss-family
differences accounted for ~1pp while queue removal accounted for ~19pp is only
publishable *because* the loss families are established and correctly implemented.
It is a negative result about a well-specified space.

### 1.3 Memory queues: staleness is known, the mechanism is not decomposed

This is where the project's defensible novelty lives, so the prior art needs to be
stated precisely.

**Momentum Contrast (MoCo, CVPR 2020)** introduced the FIFO negative dictionary
with a slowly-evolving momentum key encoder. The momentum encoder exists
specifically to keep queued keys *consistent*. The project's queue has **no such
mechanism** — it is a detached student-projector queue — which is a genuine
regime difference, though not by itself a contribution.

**Cross-Batch Memory (XBM, CVPR 2020)** is the load-bearing prior art. Inspected at
full text and supplement, it provides:
- a FIFO historical-embedding memory;
- explicit drift measurement at intervals 10/100/1,000;
- a memory-ratio sweep at fixed batch size;
- a batch-size study, reporting reduced batch sensitivity with XBM;
- saturation at moderate memory sizes.

And it provides the justifying claim this project targets:

> "The embedding of an instance actually drifts at a relatively slow rate
> throughout the training process, suggesting that deep features of a mini-batch
> computed at past iterations can considerably approximate those extracted by the
> current model."

Its regime is **supervised unimodal metric learning**, and it does not separate two
modality-specific queues.

**Adaptive Cross Batch Normalization (ACBN, 2023)**, inspected at full text,
already identifies distribution mismatch between accumulated and current
embeddings, notes that large memory relative to small batches accumulates drift,
sweeps batch size at fixed memory and memory at fixed batch, reports cases where
XBM is **worse than no memory**, and corrects drift by moment matching.

**Therefore, and this must be stated plainly in the thesis:** a generic queue-capacity
curve is **not novel**; a capacity×batch interaction is **not novel**; "stale
features hurt" is **not novel**. Q01's dose-response curve is confirmatory of
known directionality.

What the screened corpus does **not** contain is any work that measures *drift*
as a quantity separate from *degradation*, **per modality**, **across queue ages**,
in bidirectional cross-modal dual-encoder retrieval. See §2.1.

The two closest structural matches were both ruled out at source:
- **CODER** (ECCV 2022) — dual dynamic *modality-specific* memory banks for
  image-text retrieval. Full paper fetched. It has exactly the apparatus needed
  and never asks the question: Recall@K only, component-level ablations only, no
  per-modality bank ablation, no drift-versus-degradation analysis.
- **CSMCIR** (`arXiv:2601.03728`, 2026) — identifies memory-bank staleness with age
  and proposes an entropy-based dynamic bank, but for *composed* image retrieval,
  and never measures drift separately from degradation.

### 1.4 Learned-query token aggregation is a decade-old primitive

The project's `LearnedQueryPatchPool` / `LearnedQueryTextPool` sit on a long,
crowded lineage:

- **Set Transformer (2018)** — Pooling by Multihead Attention with *k* learned seed
  vectors. `k=1` is the project's exact configuration.
- **Perceiver / Perceiver Resampler / Q-Former (2021–2023)** — the same block under
  other names, compressing variable token sets to fixed size. *(IDs unconfirmed;
  cite by name only until verified.)*
- **CoCa (2022)** — attentional poolers inside a contrastive image-text model.
- **SigLIP / SigLIP 2 (2023–2025)** — a MAP head using a learnable probe as
  attention query, plus LayerNorm and MLP, applied to **both** towers. This is a
  line-for-line architectural match to `LearnedQueryTextPool`.
- **Attention, Please! Revisiting Attentive Probing** (`arXiv:2506.10178`, June
  2025) — the single closest paper in the corpus, and the only one fetched and
  quoted verbatim. It studies attention pooling over *frozen* encoder features
  explicitly as an accuracy-versus-parameter-efficiency trade-off, finds existing
  approaches "over-parameterized and computationally inefficient," and proposes a
  lightweight multi-query cross-attention pooler that "eliminates redundant
  projections and reduces the number of trainable parameters."

That last item is the `token_aggregator_scale` study's thesis, published fourteen
months earlier.

**The gating mechanism is also precedented.** **Flamingo** (2022) multiplies new-layer
output by `tanh(α)` with α initialised to 0; **LLaMA-Adapter** (2023) uses zero-init
gated attention; **CaiT/LayerScale** (2021) uses zero-init per-channel residual
gating. The project's `sigmoid(−2.0) ≈ 0.119` is the same idea with a small
non-zero start — which means the aggregator is **not an exact no-op at
initialisation**. "Near-no-op" is the accurate description and is what the
methodology says.

**On the text side**, **Sentence-BERT** (2019) reports mean pooling as the best
simple strategy — which is the published basis for the project's own pre-registered
prediction that the text-side gain would be small. *Comparative Analysis of Pooling
Mechanisms in LLMs* (2024) supports this.

**Consequence.** `E06`'s and `E08`'s architecture is **not** presented as novel. The
empirical result stands; the block does not.

### 1.5 Token reduction and efficiency: crowded, and directly tested

The [`efficient_frozen_vlm_2026`](../research/efficient_frozen_vlm_2026/) round
established that plain resolution reduction, token pruning/merging, early exit,
cascades, quantisation, Matryoshka dimensions and distillation are all established:
**ToMe** (2023), **DynamicViT** (2021), **MADTP** (CVPR 2024), **Patch Ranking**
(WACV 2025), **MoPE-CLIP** (CVPR 2024), **Attentive Mask CLIP** (ICCV 2023),
**Multiple-Exit Tuning** (2024), **ICAR** early-exit retrieval (2026),
**Bi-Encoder Cascades** (ICCVW 2023), **Matryoshka Representation Learning** (2022).
Several are marked `direct-conflict` in that round's `sources.csv`.

The refresh adds **PuMer** (ACL 2023), a closer multimodal pruning/merging
comparator than ToMe alone, plus **Dyna-ViT** and saliency-driven token merging
(CVPR 2026). TokenShift now supplies a completed fixed-merging latency and
accuracy test; it is a negative comparison within this established field.

Its [`F1_novelty_boundary.md`](../research/efficient_frozen_vlm_2026/findings/F1_novelty_boundary.md)
narrowed the defensible object to *query-independent, cross-modal-neighbourhood-preserving
conditional token computation inside a frozen SSL vision encoder*. **That object was
never built** — see §3.3. What replaced it is static aggregation, which is §1.4
prior art.

### 1.6 Distillation and data scaling are context, not claims

**TinyCLIP** (2023), **CLIP-KD** (CVPR 2024), **DIME-FM** (2023), **MCAD** (2024),
**MobileCLIP** (2024) and **MobileCLIP2** (2025) establish CLIP distillation.
`E02` and `E05` are controlled studies *within* this established space; `E05`'s
finding that teacher benchmark strength does not predict student transfer is the
interesting part, and it is a negative result about teacher selection, not a new
distillation method.

For parameter-efficient adaptation, **VL-Adapter**, low-rank CLIP adaptation,
**B-HFA** and **HALoRA** precede or directly overlap FreezeShift. HALoRA is the
highest-risk recent comparator because it allocates low-rank capacity across a
dual encoder while modelling modality asymmetry. FreezeShift is therefore a
bounded tower ablation and upper bound, not an algorithmic PEFT contribution.

On data, **Reproducible scaling laws for CLIP** (`arXiv:2212.07143`) is the
project's strongest defence of its reference gap: M_T1 reaches **54.487%**
Flickr-validation mean bidirectional R@1 against a locally evaluated OpenCLIP
reference at **69.94%**, a ~15.5pp gap. That gap is substantially a *data-scale*
gap, and this paper quantifies the power law. **VeCLIP** (2023), **Filter & Align**
(2023), **Dataset Growth** (2024) and **Differential-informed Sample Selection**
(2025) all support curation-over-volume, which is exactly what `D02`/`D03` found
independently.

### 1.7 Methodology literature legitimises the protocol

**Preregistering NLP Research** (2021), **Underspecification** (2021), **Sources of
Irreproducibility** (2022) and the **NeurIPS reproducibility programme** (2021)
provide the published basis for the hash-chained preregistration, multi-seed
requirement and sealed-test discipline. These justify the protocol; they do not
make the protocol a contribution.

---

## 2. Gaps this project has CLOSED

### 2.1 ✅ The sufficiency of XBM's slow-drift justification — falsified in the cross-modal regime

**This is the project's novelty-bearing contribution.**

**The gap.** XBM justifies memory banks on slow drift. Nothing in the screened
corpus tests whether drift magnitude actually *determines* damage, per modality,
across ages, in bidirectional cross-modal retrieval.

**What closed it.** `Q02`/`Q03`
([`results/queue_factorial/`](../results/queue_factorial/)), a preregistered
three-seed factorial with modality-separated queues and measured (not nominal)
eviction ages.

**Evidence:**

| Condition | Measured age | Cumulative drift | Degradation | SD |
|---|---:|---:|---:|---:|
| image-only | 16 | 0.08613 | 13.078pp | 0.229 |
| text-only | 64 → measured 13 | 0.08606 | **17.818pp** | 0.933 |

Matched drift to within 0.00007; **4.74pp** degradation gap; roughly 20× the seed
SD. `matched_drift_claim: GENERAL` at **4 of 4** tolerances (0.005 → 0.05), with two
independent matched pairs at the strictest tolerance and a third entering at 0.02.
Pooled drift-only R² = 0.124 against 0.675 (image) and 0.765 (text) within
modality — the curves demonstrably do not collapse.

**The dissociation is a difference in rate.** Regressing degradation on measured drift
separately by modality gives slopes of 5.553 pp per unit drift (image) against 131.442
(text) — text is roughly 24× more sensitive — while the *intercepts* run the other way
(12.008 image, 7.340 text). Restricting both towers to their overlapping drift interval
preserves the effect at a conservative 7.2×. See
[`queue_chapter_modality_gap_rebuttal.md`](queue_chapter_modality_gap_rebuttal.md).

⚠️ **Corrected 2026-08-08.** Earlier drafts of this claim, and F5 itself, described the
more-damaged modality as *reversing* with queue age. That reversal holds only on the
**nominal** age axis. On the **measured** eviction-age axis the text tower is more
damaged at every comparable point (at measured age 4: 13.717pp text vs 11.658pp image),
because the text queue's realised age falls far below its nominal setting — nominal 64
evicts at a measured age of 13. The sign-flip formulation must not be used.

**How hard the prior-art check was.** Three rounds: 6 targeted manual searches (round
2 deliberately avoiding queue vocabulary), 288 systematically screened citing works
of XBM and CODER via OpenAlex, 4 full-text or abstract verifications including
CODER at full text. One Tier-1 hit, out of domain (visible/infrared person re-ID).

**Claim strength.** Bounded prior-art claim, high confidence *within the screened
corpus*. Two honest coverage limits are recorded and must be stated in the thesis:
OpenAlex under-indexes CS proceedings, and abstract-level screening has a known
false-negative mode — CODER itself would have passed straight through it and was
caught only by structural reasoning.

**Suggested contribution sentence** (revised 2026-08-08; supersedes the wording in
[`F5`](../research/project_wide_bibliography_2026/findings/F5_queue_claim_unoccupied.md),
which uses the unsupported sign-flip formulation):

> Cumulative projector drift does not determine cross-batch-memory damage: conditions
> matched on drift differ by up to 4.74pp, and the text tower degrades roughly an order
> of magnitude faster per unit of drift than the image tower — a dissociation not
> reported in prior cross-batch-memory work.

### 2.2 ✅ The F3 confirmatory checklist — fully delivered

[`F3_claim_boundary.md`](research/alignment_queue_prior_art/findings/F3_claim_boundary.md)
ruled the original one-seed dose-response curve **observational** (capacity
confounds negative count with feature age) and specified five requirements for a
confirmatory claim. All five are present:

| Requirement | Delivered | Evidence |
|---|---|---|
| Matched-age tests across batch sizes | ✅ | 8 ages × batch 512 and 1024 |
| Three seeds | ✅ | `seed_replication: HIT` — all >1pp orderings held in all 3 seeds |
| Image-only and text-only queues | ✅ | separate `queue_mode` arms |
| Actual age logging for both modalities | ✅ | `measured_eviction_age_steps`; text nominal-64 measured as 13 |
| Fixed optimiser/evaluation protocol | ✅ | `frozen_predictions_sha256_unchanged_by_script: true` |

The acknowledged residual — batch size still changes in-batch negative count — is
explicitly handled by the age-0 control row and disclosed as such.

### 2.3 ✅ ACBN's "memory can be worse than none" — sharpened to a threshold

ACBN reports *cases* where XBM underperforms no memory. `Q02` supplies the
boundary in this regime: harm begins at **age 1** (−6.46pp at batch 1024, −5.60pp
at batch 512, against matched queue-free controls). There is no empirically safe
non-zero queue age for projector-only cross-modal alignment. That is a stronger,
more specific statement than the prior art contains.

### 2.4 ✅ Preregistration discipline actually held under pressure

Three separate events demonstrate this, and together they are reportable
methodology:

- `F02` (alignment_v3) failed its compatibility gate and was **preserved as a
  negative screen** rather than having its threshold relaxed.
- `X02` (guarded hard-negative mining) returned
  `INFEASIBLE_UNDER_PREREGISTERED_GUARD`; no training was authorised and no
  threshold was moved after observation.
- `Q02` scored its own `modality_asymmetry` prediction as a **MISS** and assigned no
  partial credit — the reversal became the result rather than being reframed.

Against the reproducibility literature (§1.7) this is demonstrated, not asserted,
compliance. It is not a novelty claim, but it is a defensible methods contribution.

### 2.5 ✅ Curation-over-volume — independently reproduced in the frozen-projector regime

`D01`/`D02`/`D03` reproduce the curation-over-volume finding of §1.6 in a regime
those papers did not test (frozen encoders, <5M trainable parameters, fixed
compute): full COCO beat a 25% subset by 15.03pp at matched updates; a qualified
CC3M mirror **underperformed** the COCO control by 8.13pp at matched compute and
5.89pp after four passes; mixing recovered most of the loss (53.133% vs 47.209%).
Correctly, no universal claim about CC3M is made, because the extended arm hit its
ceiling without converging.

### 2.6 ✅ Measurement-artifact corrections that changed conclusions

`E01` is a real methods finding: Flickr30k was promoted to primary because COCO had
selection leakage, and dynamic padding, deployed MobileCLIP2 fusion and corrected
operator-level FLOPs replaced misleading first-pass measurements. Related, `E03`
notes DINOv3's dynamic rotary encoding avoids the learned-position interpolation
artifacts that resolution changes usually introduce.

---

## 3. Gaps still OPEN

Each is tagged with feasibility against **assets that already exist**.

### 3.1 ✅ The modality-gap rival explanation — INTEGRATED 2026-08-08

**Status:** draft written to
[`queue_chapter_modality_gap_rebuttal.md`](queue_chapter_modality_gap_rebuttal.md);
citations verified at source; conservative slope-based argument integrated into
the dissertation discussion. Supervisor review remains appropriate before submission.

**The gap.** Liang et al. (`arXiv:2203.02053`, NeurIPS 2022) establish that each encoder
confines embeddings to a narrow cone and that contrastive training preserves the
separation. This enables the strongest available objection to §2.1:

> "Your modality-dependent damage is just the known modality gap. Image and text
> embeddings live in differently-shaped regions, so of course equal drift means
> different things in each. You have re-described cone geometry."

**Current state.** `grep -ri "modality gap|intra-modal|cone"` across `docs/`,
`experiments/`, `results/`, `src/` and `reports/` returns **zero hits**. The defence
does not exist anywhere in the repository.

**The argument used.** Not F5's sign-flip argument, which the measured-age data does not
support (see the ⚠️ note in §2.1). Instead: a static geometric difference is an
**intercept**, predicting a roughly constant offset between towers. The fitted
relationships are a **slope** story — text has the *lower* intercept and a ~24× steeper
response to drift (7.2× under the conservative overlap-restricted estimate). Cone
geometry predicts the term that is small and negatively signed here; it does not predict
the term that dominates.

**Verification outcome — this check strengthened the finding.** All four modality-gap
candidates named in F5 were fetched. One was correct (Liang et al.), one was a duplicate
of it, and two were mischaracterised (`arXiv:2406.17639` is a mitigation paper;
`arXiv:2502.04263` concerns intra-modal misalignment). Two better mechanistic sources
were found and verified — `arXiv:2412.07909` (gradient-flow account) and
`arXiv:2510.03268` (dimension-collapse account). **Neither reports any per-modality
asymmetry**, and neither measures drift. The rival literature supplies a static
separation with no differential-sensitivity prediction, which is precisely the gap the
rebuttal exploits.

**Feasibility:** ✅ **Closed pending review.** Optional strengthening (measure intra-modal
cosine spread per tower and show the dissociation survives normalising drift by it)
remains feasible — checkpoints and `scripts/extract_embeddings.py` exist — and is
currently written into the draft as future work.

### 3.2 ✅ The latency gate was repaired — CLOSED 2026-08-08

**The gap.** [`freezeshift/runner.py:36`](../freezeshift/runner.py) hardcodes
`CEILING_MS = 9.480`, inherited from
[`configs/resolution_arm/selection.yaml`](../configs/resolution_arm/selection.yaml)
— an OpenCLIP measurement taken in a *different* session and allocation. Measured
M_T1 Q3 varies across sessions on the same node and GPU:

| Session | M_T1 Q3 | Verdict against 9.480 |
|---|---:|---|
| `text_aggregation_study/profiling` | 9.2086 ms | pass |
| README "repeated pooled" figure | ~8.998 ms | pass |
| `freezeshift/results/report/latency.json` | **9.4970 ms** | **fail** |

**The gate fails its own unmodified baseline.** Consequently the FreezeShift
rejections rest on a cross-session comparison. The vision arm measured **9.4826 ms
— faster than the contemporaneous M_T1 baseline** — while gaining +3.56pp, and was
nonetheless marked ineligible.

**Resolution.** Route (b) was executed as a declared post-observation
measurement repair. Slurm job 2297019 profiled OpenCLIP, M_T1, and all three
FreezeShift arms in the same allocation for ten independently warmed
repetitions. Every candidate was faster than OpenCLIP and every paired 95%
bootstrap interval lay entirely below zero. The original negative verdict is
preserved. Evidence: [`latency_amendment/results/report/report.md`](../latency_amendment/results/report/report.md).

The two routes considered before execution were:
- *(a) no new compute* — re-derive eligibility as a **paired same-allocation delta**
  against the contemporaneous M_T1 row, which `latency.json` already contains
  (`same_allocation: true`). This is the pattern `F4` already endorsed for
  `token_aggregator_scale`'s contemporaneous CLS control.
- *(b) one allocation* — re-profile the OpenCLIP reference in the same allocation
  via `src/alignment_v3/efficiency_frontier.py` to derive a fresh absolute ceiling.

⚠️ **Route (a) is a post-observation gate change**, which is precisely what `F02` and
`X02` refused to do. It must be documented as an **amendment on measurement-invalidity
grounds** — the ceiling is invalid because it disqualifies the unmodified baseline,
which is a measurement fault, not a result — in the style of the tie-policy
amendment already recorded in [`freezeshift/README.md`](../freezeshift/README.md).
**Route (b) avoids the problem entirely** and is the cleaner choice if an allocation
is available.

**Not affected:** the E5 rejections in `E08` are robust. E5 arms measured
9.9165–10.1992 ms Q3, far outside the ~0.3 ms session variance. That exclusion stands.

### 3.3 ✅ Token-reduction closure — COMPLETED 2026-08-10

The original 2026-07-28 plan proposed generic ToMe/ATS and routing work. Building
a new router would have reopened a closed architecture search, so the scoped
question was instead tested by TokenShift: fixed parameter-free 2x2 patch
merging at blocks 8 and 6, crossed with frozen M_T1 and dual FreezeShift parents.

Profiling showed real speedups, but the completed three-seed accuracy study
rejected every arm:

| Parent and merge | Validation R@1 | Change vs parent | Mean Q3 |
|---|---:|---:|---:|
| Frozen, block 8 | 41.907% | -12.581pp | 8.204 ms |
| Frozen, block 6 | 38.537% | -15.950pp | 7.573 ms |
| Dual LoRA, block 8 | 57.725% | -5.690pp | 8.203 ms |
| Dual LoRA, block 6 | 52.554% | -10.861pp | 7.556 ms |

The closest literature comparison is now PuMer rather than ToMe alone: PuMer
shows that adaptive multimodal pruning/merging can preserve accuracy much better.
The correct conclusion is narrow—fixed spatial averaging made this model faster
but moved it off the accuracy--latency frontier. Adaptive routing remains future
work, not an unperformed requirement for this dissertation.

### 3.4 ✅ Related work — WRITTEN 2026-08-08

**The gap.** [`docs/full_completion_report.md`](full_completion_report.md) and
[`docs/supervisor_review.md`](supervisor_review.md) contain **zero** occurrences of
`arXiv`, `et al`, SigLIP, CoCa, Set Transformer, Perceiver, Q-Former or attentive
probing. The SigLIP-2 MAP head — a line-for-line match to `LearnedQueryTextPool` —
appears in no write-up-facing document. This was recommended action #2 on
2026-08-06 and was described there as "a genuine viva liability."

**Resolution:** the dissertation now contains a bounded related-work chapter
using verified primary records for frozen alignment, cross-batch memory,
modality gap, learned-query pooling, token merging, and scaling. Unverified
`search_summary` records were not promoted into the submitted reference set.

### 3.5 🟠 M_T1's headline margin is unconfirmed on sealed test — CLOSEABLE ONCE, IRREVERSIBLE

**The gap.** M_T1's text-aggregation gain is **+0.621pp** over `historical_c4_mean`
against a pooled-SD threshold of 0.00318 — **1.95σ** — measured under
Flickr-validation epoch selection, with double-selection exposure disclosed
(`selection_gain_pp = 0.128`). A 1.95σ validation margin with acknowledged
selection pressure is not a safe margin. Flickr30k test remains correctly sealed
(`flickr_test_used: false` in both the text study and FreezeShift).

**Feasibility:** ✅ **Closeable with one evaluation run** — but it is **one-shot and
irreversible**, and [`docs/evaluation_protocol.md`](evaluation_protocol.md) states
that until this happens the project must say fully untouched external retrieval
validation is unavailable.

**Sequencing constraint:** do not break the seal until §3.2 is resolved. If the
latency gate is re-derived and a FreezeShift arm becomes eligible, the model that
should be tested changes.

**Current state:** the sequencing constraint is now satisfied. Section 3.2 is
closed and `final_model_freeze.md` selects dual FreezeShift as the one-shot test
candidate. The seal remains closed pending explicit author/supervisor approval.

### 3.6 ✅ FreezeShift (X03) — DOCUMENTED AND REGISTERED 2026-08-08

**The gap.** [`experiments/registry.yaml`](../experiments/registry.yaml) and
[`experiments/README.md`](../experiments/README.md) both mark `X03` as
`READY_NOT_RUN` ("a nine-run LoRA study is prepared"). It has completed:

| Arm | Val mean R@1 | Gain vs M_T1 | Params | Q3 latency | Marked eligible |
|---|---:|---:|---:|---:|---|
| M_T1 (baseline) | 54.487% | — | 2.90M | 9.497 ms | — |
| vision | 58.047% | **+3.56pp** | 4.08M | 9.483 ms | ❌ |
| text | 59.724% | **+5.24pp** | 3.68M | 9.641 ms | ❌ |
| dual | **63.416%** | **+8.93pp** | 4.86M | 9.605 ms | ❌ |

All three clear the 5M budget and all three exceed pooled SD substantially. None
beats the OpenCLIP reference (69.94%).
[`freezeshift/results/report/report.md`](../freezeshift/results/report/report.md)
is **two lines long**.

**Resolution:** the result-level report, final model-freeze record, canonical
index, and registry now report the completed study. The repaired latency rule
selects the dual arm as the final development adaptation candidate; the fully
frozen claim remains attached to M_T1.

### 3.7 ✅ Probe1 taxonomy boundary — CLOSED IN WRITING 2026-08-10

**The gap.** `X01` (`probe1_error_decomposition`) defines its nine failure
categories ad hoc. `arXiv:2407.15239` has a published taxonomy for image-text
retrieval granularity. A reviewer will ask "why these nine?"

**Resolution.** Probe1 is now described as a diagnostic of annotation validity and
ranking severity, not a comprehensive linguistic-granularity taxonomy. Its
model-assisted labels—fine-grained miss, underspecified caption, defensible
label, retrieved-also-correct and atypical positive—answer whether a nominal
error is evidentially clean. The published fine-grained taxonomy answers which
linguistic property is stressed. Those are complementary axes, so the nine local
categories are not represented as a replacement standard. A full remap remains
optional reanalysis rather than a missing thesis claim.

### 3.8 🟡 The residual queue mechanism is unidentified — PARTIALLY CLOSEABLE

**The gap.** `Q03` establishes that drift is necessary but not sufficient, and names
three unseparated candidates: gradient sensitivity, representation geometry, and
false-negative structure. None is tested. The current wording — "no new mechanism
is claimed from these data" — is honest and correct.

The 2026 refresh increases the importance of false negatives: FALCON explicitly
balances hard and false negatives in vision-language alignment, while PCME notes
that COCO annotations are non-exhaustive. These works strengthen the alternative
explanation but do not report per-modality queue-age drift--damage curves.

**Feasibility:** 🟡 **Partial.**
- ✅ *Known same-image recurrences* are resolved by existing instrumentation. The
  loss constructs the multi-positive mask after the queue is appended, and the
  non-zero queue-positive diagnostic confirms that repeated image identities are
  treated as positives. See
  [`known_positive_audit.md`](../results/queue_factorial/interpretation/known_positive_audit.md).
- 🟡 *Unannotated cross-image semantic false negatives* are not enumerable from
  image IDs alone and require semantic or human relevance judgements.
- ✅ *Representation geometry* overlaps with §3.1's optional intra-modal spread
  measurement — one analysis serves both.
- ❌ *Gradient sensitivity* requires instrumented retraining across the factorial and
  is a separate study.

**Recommendation:** do not attempt full mechanism separation. The dissociation is
the contribution; the residual semantic, geometry and gradient mechanisms are
legitimate future work.

### 3.9 ⚪ Not closeable within this project

Recorded so the thesis can say so explicitly rather than leaving them looking
overlooked:

| Gap | Why it cannot close here |
|---|---|
| **Universal claim about CC3M transfer** | `D02`'s extended arm hit its compute ceiling without converging. A universal claim needs convergence-scale compute the project does not have. Current bounded wording is correct. |
| **Tuned multi-teacher mixtures** | `E05` shows equal 0.5/0.5 mixing does not beat MobileCLIP2 alone but explicitly does not rule out tuned mixtures. Tuning is a new search with its own preregistration. |
| **Exact no-op gate initialisation** | A zero-init `tanh` gate would make baseline-equivalence exact and is trivially defensible against Flamingo/LayerScale, but changing it requires retraining every aggregation arm. "Near-no-op" is accurate; defend in text. |
| **Closing the ~15.5pp gap to OpenCLIP** | Substantially a data-scale gap. `arXiv:2212.07143` is the citation that pre-empts a reviewer reading it as architectural failure. |
| **Proof of novelty** | Every §2 claim is absence-of-evidence over a screened corpus. `F2`'s novelty-by-conjunction ("no paper combines these five things") is the weakest surviving claim and must not carry the contribution. |

---

## 4. Recommended contribution hierarchy

Derived from the corpus, in the order the thesis should assert them:

1. **Novelty-bearing claim — the queue chapter.** Cumulative projector drift does
   not determine cross-batch-memory damage: matched-drift conditions differ by up
   to 4.74pp, and the text tower degrades about 7.2× faster per unit drift in the
   overlap-restricted comparison. Positioned as a **boundary condition on XBM's
   justification**, not a refutation of XBM. Coverage caveat attached.
2. **Empirical centrepiece — token aggregation.** The largest clean positive gain
   under the frozen-backbone and budget constraints (`E06`: mean pooling recovers
   ~1.33pp, learned-query ~6.08pp, transformer ~6.91pp — access to patch information
   alone is insufficient; learning *which* patches to keep is what pays). The block
   is established prior art and is **not** sold as a new architecture.
3. **Methods findings.** Batch-scaling, padding, fusion and FLOP-accounting
   corrections; the demonstrated preregistration discipline of §2.4; the
   curation-over-volume reproduction of §2.5.
4. **Bounded negative results.** Teacher strength does not predict student transfer
   (`E05`); LR retuning did not improve the selected model (`E09`); hard-negative
   mining was infeasible under its guard (`X02`); FLOP reduction does not translate
   into batched latency for frozen SSL encoders (§3.3).

**Framing rule, from `F1`/`F2`:** the contribution is **methodological and empirical,
not architectural**. Lead with the module and Objection 1 ("this is the SigLIP MAP
head with a gate") wins.

---

## 5. Completion record for the suggested work

| # | Action | Cost | Blocks |
|---|---|---|---|
| 1 | Write the modality-gap rebuttal (§3.1) | **Complete and integrated** | supervisor review |
| 2 | Write the routing scoping statement (§3.3) | **Complete** | none |
| 3 | Repair the latency gate (§3.2) | **Complete; route (b), job 2297019** | none |
| 4 | Write FreezeShift report and fix registry (§3.6) | **Complete** | none |
| 5 | Verify core sources and write related work (§3.4) | **Complete** | refresh before submission |
| 6 | Sealed-test evaluation (§3.5) | **Not run; deliberately pending approval** | final author/supervisor decision |
| 7 | Optional false-negative structure measurement (§3.8) | Future work | not a submission blocker |
| 8 | Complete TokenShift accuracy closure (§3.3) | **Complete; all four merge arms rejected** | none |
| 9 | Add compact/field-anchor/frozen-peer comparison hierarchy | **Complete** | no cross-split ranking |
| 10 | Refresh 2025--2026 PEFT, token and negative-selection work | **Complete to 2026-08-10** | repeat at formal submission |

The protocol amendment is documented in `latency_amendment/`. The one-shot test
remains last and is the only outstanding empirical action required for a
confirmatory test claim.

---

## 6. Maintenance

- **Re-run the Tier-1 search immediately before submission.** `F2` is the claim most
  likely to decay; the corpus itself notes that 2026 work is still appearing.
- **Verify before citing.** See the warning box at the top. Verification status is
  per-row in each `sources.csv`.
- **Do not cite** the claim attributed to `arXiv:2303.11866` (Khan & Fu, ICLR 2023)
  about alignment changing normalisation parameters most in shallow text-encoder
  layers. It appeared in a search summary, the abstract fetch did not confirm it,
  and it remains unverified. If real it would support encoder-side asymmetry, but it
  must be read in full first.

## 7. Provenance

| Round | Date | Scope | Output |
|---|---|---|---|
| 1 | 2026-07-26 | Frozen alignment, queue prior art | 33 sources, findings F1–F3 |
| 2 | 2026-07-28 | Efficient frozen VLM, token reduction | 14 sources, novelty boundary |
| 3 | 2026-08-06 | Token aggregation, text pooling | 54 ranked entries, findings F1/F2/F4 |
| 3b | 2026-08-06 | Queue/drift targeted + citation graph | F5; 288 works screened via OpenAlex |
| 4 | 2026-08-10 | 25-gap decomposition refresh | 20 searches, 322 raw records; primary-source rescue verification |
| — | 2026-08-10 | This document: consolidation and gap audit | [`literature_gap_closure_20260810.md`](literature_gap_closure_20260810.md) |

Round 3's own scope limitation is important and is preserved here: it was titled
"project-wide" but searched only aggregation and text pooling, then generalised into
a whole-project novelty claim it had no corpus to support. Round 3b was commissioned
to fix exactly that, and it is what produced the project's strongest finding.
