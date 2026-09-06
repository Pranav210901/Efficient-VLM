# Draft: modality-gap rebuttal for the queue chapter

**Status:** Integrated into `dissertation.md` on 8 August 2026; supervisor review pending.
**Purpose:** pre-empt the strongest objection to the drift-dissociation finding,
identified in [`F5_queue_claim_unoccupied.md`](../research/project_wide_bibliography_2026/findings/F5_queue_claim_unoccupied.md).
**Placement:** dissertation queue results and discussion, after the
drift-regression result and before mechanism limitations.

---

## ⚠ Correction to the F5 framing — read before using this draft

F5 recommends defending the finding by arguing that *"the age-dependent sign flip is
not explicable by static cone geometry — a fixed geometric difference predicts a fixed
ordering; yours reverses between age 4 and age 16."*

**That defence is fragile and should not be used as written.** The reversal exists on
the **nominal** queue-age axis, but not on the **measured** eviction-age axis. From
[`drift_regression.json`](../results/queue_factorial/interpretation/drift_regression.json):

| Queue mode | Nominal age | **Measured** age | Drift | Degradation |
|---|---:|---:|---:|---:|
| image-only | 4 | 4 | 0.01971 | 11.658pp |
| image-only | 16 | 16 | 0.08613 | 13.078pp |
| image-only | 64 | 64 | 0.31981 | 13.654pp |
| text-only | 4 | **1** | 0.00203 | 4.936pp |
| text-only | 16 | **4** | 0.02186 | 13.717pp |
| text-only | 64 | **13** | 0.08606 | 17.818pp |

The text queue's realised age is far below its nominal setting. The apparent flip at
nominal age 4 compares **image at measured age 4** against **text at measured age 1** —
an unmatched comparison. At the one measured age where both modalities are observed
(age 4), text is the more damaged modality: 13.717pp versus 11.658pp. On the measured
axis there is no reversal at all; text is more damaged at every comparable point.

An examiner who reads the measured-age column — which the project logged precisely so
this could be checked — will dismantle the sign-flip argument in one question.

**The argument below uses a different and stronger leg: a slope difference, not a sign
flip.** It is robust to the age-axis problem because it is defined on the drift axis,
where both modalities are directly comparable.

---

## Draft text

### Is the dissociation just the modality gap?

A natural objection to the preceding result is that it restates a known property of
contrastive vision-language models rather than identifying a new one. Liang et al.
[Mind the Gap, NeurIPS 2022] establish that "the representation of a common deep neural
network is restricted to a narrow cone," with the consequence that "the representations
of the two modalities are clearly apart when the model is initialized," and that
contrastive optimisation subsequently "keeps the different modalities separate by a
certain distance." Eslami and de Melo [AlignCLIP, 2024] describe the resulting space as
"overly sparse and disconnected, with different modalities being densely distributed in
distinct subregions of the hypersphere." If image and text embeddings occupy differently
shaped regions of the space, then a given cosine displacement plausibly carries
different retrieval consequences in each. On that reading, the observation that equal
cumulative drift produces unequal degradation is a restatement of cone geometry, and no
modality-specific *dynamic* factor is required to explain it.

The objection is well posed and it does not threaten the measurement. The dissociation
reported above is real under either interpretation. What it threatens is the inference
that something beyond static geometry is at work. We therefore address it directly.

**A static geometric difference between the two cones is an intercept, not a slope.**
If the modality gap were the whole explanation, it would predict that text-side and
image-side degradation differ by an approximately constant offset — a fixed penalty
reflecting a fixed difference in how the two embedding regions are shaped — while
responding to accumulated drift at comparable rates. The fitted relationships do not
have that form.

Regressing degradation on measured cumulative drift at eviction, separately by
modality (n = 9 per modality, three conditions × three seeds), gives:

| Modality | Intercept (pp) | Slope (pp per unit drift) | R² | RMSE (pp) |
|---|---:|---:|---:|---:|
| Image | 12.008 | **5.553** | 0.675 | 0.495 |
| Text | 7.340 | **131.442** | 0.765 | 2.615 |

The intercept difference runs in the *opposite* direction to the observed damage: the
text tower starts from a **lower** baseline degradation, not a higher one. The
divergence is carried almost entirely by the slope, which is roughly **24 times
steeper** for text. In the pooled model with an interaction term, this appears as a
small negative main effect for modality (`is_text` = −4.669) against a large positive
interaction with drift (`drift × text` = +125.889).

This is a difference in *sensitivity to drift*, not a difference in baseline position.
Static cone geometry predicts the term that is small and negatively signed here; it does
not predict the term that dominates.

**The slope difference is not an artifact of unequal drift ranges.** Because the two
modalities span different drift ranges — image 0.020 to 0.320, text 0.002 to 0.086 — the
full-range fits are not automatically comparable. Restricting both modalities to their
overlapping drift interval (0.002 to 0.086) and taking simple two-point slopes over that
interval preserves the effect: 21.4 pp per unit drift for image against 153.3 for text,
a ratio of **7.2×**. The conclusion is therefore robust to the range restriction,
although the ratio is smaller than the full-range fits suggest, and we report the
restricted figure as the conservative one.

**The consequence for interpretation.** The observed pattern is that the two towers
degrade at materially different *rates* per unit of representational movement, with the
text tower far more sensitive. A fixed geometric separation between modality cones does
not produce that pattern; it produces an offset. Explaining a rate difference requires a
factor that is itself dynamic — candidates include differential gradient sensitivity in
the two projection heads, differences in false-negative structure between the modalities
under multi-caption supervision, and modality-specific curvature of the local
representation geometry. The present factorial does not separate these, and we claim no
mechanism from these data.

**The mechanistic modality-gap literature does not report this asymmetry.** The two works
that analyse the gap's causes rather than merely measuring it — Yaras et al. [Explaining
and Mitigating the Modality Gap, 2024/2026], which characterises gap emergence through
gradient-flow dynamics and attributes it to mismatched pairs and the learnable
temperature, and Yi et al. [Decipher the Modality Gap, 2025], which derives the gap from
dimension collapse and subspace constraints — both treat the two modalities
symmetrically. Neither reports differential sensitivity, differential degradation rates,
nor any per-tower asymmetry. The modality-gap account therefore supplies a static
geometric separation but no prediction that one tower should be an order of magnitude
more sensitive to representational movement than the other.

We note that the modality gap and this result are not rivals in the strong sense. Cone
geometry may well be the *reason* the two towers have different drift sensitivities. The
claim here is narrower and survives that possibility: drift magnitude alone does not
determine damage, and a model that treats stale entries as safe whenever measured drift
is small — the justification offered for cross-batch memory — is therefore unsound in
the cross-modal dual-encoder regime, because the same drift carries an order of magnitude
different cost depending on which tower produced it.

### Limitations of this analysis

Three limitations bear on the strength of the argument.

First, the per-modality regressions are **descriptive and post-hoc**. They were not
preregistered, and they rest on six conditions with three seeds each. The interaction
model reaches R² = 0.765 with RMSE 1.88pp, which is adequate for establishing that a
single pooled curve fails but not for estimating rate parameters precisely.

Second, the nominal queue-age setting and the **measured** eviction age diverge
substantially in the text-only arm, where a nominal age of 64 corresponds to a measured
residence of 13 steps. All comparisons in this section are therefore stated on the drift
axis or on measured age, never on the nominal setting. The preregistered
`modality_asymmetry` prediction was scored on the nominal axis and recorded as a MISS;
that scoring stands and is not revisited here.

Third, this analysis addresses whether static geometry *alone* explains the dissociation.
It does not establish which dynamic factor is responsible, and it does not test the
modality-gap account directly. A direct test — measuring intra-modal cosine spread in
each tower and asking whether the dissociation survives normalising drift by it — is
feasible with the existing checkpoints and is left as future work.

---

## Notes for the author

### Citations — verified 2026-08-08

All four F5 candidates were fetched at source. **Two were mischaracterised in F5 and one
was a duplicate.** Corrected record:

| F5 candidate | What it actually is | Verification | Use |
|---|---|---|---|
| "Liang et al." | **arXiv:2203.02053** — *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, Liang, Zhang, Kwon, Yeung, Zou. NeurIPS 2022 (v1 Mar 2022, v2 Oct 2022) | ✅ `abstract_fetched_verbatim` | **Primary citation.** Both quotes in the draft are verbatim from this abstract. |
| "*Understanding the Modality Gap in CLIP* (OpenReview)" | **Not a separate paper.** OpenReview `S7Evzt9uit3` is the same Liang et al. work | ⚠️ duplicate — do not cite twice | Drop. |
| `arXiv:2406.17639` | **Not a gap-geometry paper.** *Mitigate the Gap: Investigating Approaches for Improving Cross-Modal Alignment in CLIP*, Eslami & de Melo, 25 Jun 2024. Proposes AlignCLIP; a **mitigation** paper | ✅ `abstract_fetched_verbatim` | Usable, for the "sparse and disconnected / distinct subregions" characterisation only. Do not cite as evidence about cone geometry at initialisation. |
| `arXiv:2502.04263` | **Not a gap-geometry paper.** *Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion*, Mistretta, Baldrati, Agnolucci, Bertini, Bagdanov, 6 Feb 2025 | ✅ `abstract_fetched_verbatim` | Optional and genuinely useful for a *different* point: the CLIP objective "does not enforce any intra-modal constraints," which is consistent with the two towers being free to develop different internal geometries. |

**Two additional sources found and verified during this check**, both stronger than the
two mischaracterised candidates:

| Source | Why it matters |
|---|---|
| **arXiv:2412.07909** — *Explaining and Mitigating the Modality Gap in Contrastive Multimodal Learning*, Yaras, Chen, Wang, Qu (Dec 2024, rev. Feb 2026) | The mechanistic account: gap emergence via gradient-flow dynamics, caused by mismatched pairs and the learnable temperature. ✅ verified. **Reports no per-modality asymmetry.** |
| **arXiv:2510.03268** — *Decipher the Modality Gap in Multimodal Contrastive Learning*, Yi, Douady, Chen (Sep 2025, rev. Oct 2025) | Derives the gap from dimension collapse and subspace constraints; gap converges to the smallest angle between hyperplanes. ✅ verified. **Reports no per-modality asymmetry.** |

**This check incidentally strengthened F5.** The modality gap was flagged as the
strongest rival explanation, and the natural worry was that someone in that literature
had already reported per-tower asymmetry. Both mechanistic treatments of the gap analyse
the two modalities symmetrically and neither measures drift or differential sensitivity.
The rival literature supplies a static separation and no asymmetry prediction — which is
exactly the shape of the argument the draft makes. Consider recording this as a
supplementary note under F5.

**What changed from the F5 recommendation.** F5's Option 1 (sign flip) is replaced by
the slope-versus-intercept argument for the reason given at the top. F5's Option 2
(intra-modal spread measurement) is retained but demoted to future work in the draft;
if you decide to run it, the third limitation should be deleted and the result folded
into the main argument, which would make this section considerably stronger.

**Knock-on edit required.** The contribution sentence proposed in F5 and reproduced in
[`literature_review_and_gap_audit.md`](literature_review_and_gap_audit.md) §2.1 says
"the modality suffering greater degradation reverses with queue age." On the measured
axis that is not supported. Suggested replacement:

> Cumulative projector drift does not determine cross-batch-memory damage: conditions
> matched on drift differ by up to 4.74pp, and the text tower degrades roughly an order
> of magnitude faster per unit of drift than the image tower — a dissociation not
> reported in prior cross-batch-memory work.

This is a weaker-sounding but more defensible claim, and it is the one the data
supports. The matched-drift result (§2.1 of the audit, 4/4 tolerances) is unaffected and
remains the primary evidence.
