# Queue factorial — interpretation

## Headline: no safe threshold

A queue containing embeddings only **one optimizer step old** already reduced
mean bidirectional dev R@1 by 6.46pp at batch 1024 and 5.60pp at batch 512
relative to the respective queue-free controls. Under the pre-registered
>1pp harm definition, the harm threshold is therefore age 1 for both batches.
There is no empirically safe non-zero queue age in this projector-only regime.

## Primary pre-registered verdict: passes at the boundary

Exactly **6 of 8** matched-age contrasts fell inside the ±1pp equivalence
margin—the minimum required for `age_dominates`. This is a boundary result,
not clean equivalence. Age 0 failed by −1.422pp and age 64 failed by −1.022pp;
in both cases batch 512 was lower.

| age | b512_minus_b1024_pp | within_margin | direction | net_of_age0_pp |
|---|---|---|---|---|
| 0 | -1.422 | no | batch 512 lower | +0.000 |
| 1 | -0.567 | yes | batch 512 lower | +0.854 |
| 2 | -0.201 | yes | batch 512 lower | +1.221 |
| 4 | -0.015 | yes | batch 512 lower | +1.407 |
| 8 | +0.812 | yes | batch 512 higher | +2.234 |
| 16 | +0.069 | yes | batch 512 higher | +1.491 |
| 32 | -0.430 | yes | batch 512 lower | +0.992 |
| 64 | -1.022 | no | batch 512 lower | +0.399 |

Age 0 is not a queue condition. It is the no-queue control, so its −1.422pp
gap measures the anticipated pinned-LR batch offset rather than queue damage.
The primary 6/8 verdict above remains authoritative because it was
pre-registered without this exclusion.

**Secondary descriptive analysis:** among the seven rows containing an actual
queue, 6/7 lie inside the margin; only age 64 fails. This does not replace the
primary result.

**Exploratory baseline-offset analysis:** subtracting the age-0 batch offset
makes every queue-containing contrast positive or within +2.234pp; age 64
changes from −1.022pp to +0.399pp. This analysis was not pre-registered and
does not alter the primary verdict.

## Modality crossover: the prediction was a miss

| target_age_steps | queue_mode | mean_percent | sd_pp |
|---|---|---|---|
| 4 | image_only | 25.664 | 0.376 |
| 4 | text_only | 32.385 | 0.385 |
| 16 | image_only | 24.243 | 0.229 |
| 16 | text_only | 23.604 | 0.426 |
| 64 | image_only | 23.668 | 0.207 |
| 64 | text_only | 19.503 | 0.933 |

The prediction that image-only queues would be more harmful was **missed**.
Text-only was less harmful at nominal age 4, but more harmful at ages 16 and
64. The reversal is the result; no partial credit is assigned.

## Pre-registered drift-at-eviction regression

Cumulative drift at eviction was estimated by summing measured
previous-probe cosine distances over each entry's measured eviction-residence
window, prorating the ten-step measurement interval at window boundaries.
This is measured path length, not nominal age multiplied by one fixed rate.

The two modality curves **do not collapse onto one relationship**:

- Pooled age-only R²: 0.087.
- Pooled cumulative-drift-only R²: 0.124.
- Image-only drift R²: 0.675.
- Text-only drift R²: 0.765.
- A post-hoc modality-interaction model reaches R²
  0.765, but
  it is descriptive and based on only six three-seed conditions.

The clearest counterexample to a shared curve is image-only age 16 versus
text-only age 64. Their measured cumulative drifts are nearly identical
(0.0861 versus
0.0861), yet degradation is
13.08pp versus
17.82pp—a
4.74pp gap.

Therefore, cumulative drift improves the mechanistic description but is not
a sufficient quantitative model. A modality-specific factor—such as
gradient sensitivity, representation geometry, or false-negative
structure—remains necessary. No new mechanism is claimed from these data.

### Nearest-neighbour sensitivity analysis

Observed cumulative drift spans
0.002035 to
0.319813, a range
of 0.317778.
The four matching tolerances therefore span approximately 1.57% to 15.73% of
the observed range.

| tolerance | tolerance_pct_range | image_age | image_drift | text_age | text_drift | drift_difference | degradation_difference_pp | exceeds_1pp |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 1.57 | 4 | 0.019706 | 16 | 0.021857 | 0.002151 | 2.060 | yes |
| 0.005 | 1.57 | 16 | 0.086131 | 64 | 0.086064 | 0.000068 | 4.740 | yes |
| 0.010 | 3.15 | 4 | 0.019706 | 16 | 0.021857 | 0.002151 | 2.060 | yes |
| 0.010 | 3.15 | 16 | 0.086131 | 64 | 0.086064 | 0.000068 | 4.740 | yes |
| 0.020 | 6.29 | 4 | 0.019706 | 4 | 0.002035 | 0.017671 | 6.721 | yes |
| 0.020 | 6.29 | 4 | 0.019706 | 16 | 0.021857 | 0.002151 | 2.060 | yes |
| 0.020 | 6.29 | 16 | 0.086131 | 64 | 0.086064 | 0.000068 | 4.740 | yes |
| 0.050 | 15.73 | 4 | 0.019706 | 4 | 0.002035 | 0.017671 | 6.721 | yes |
| 0.050 | 15.73 | 4 | 0.019706 | 16 | 0.021857 | 0.002151 | 2.060 | yes |
| 0.050 | 15.73 | 16 | 0.086131 | 64 | 0.086064 | 0.000068 | 4.740 | yes |

| tolerance | tolerance_pct_range | pairs | pairs_over_1pp | conclusion |
|---|---|---|---|---|
| 0.005 | 1.57 | 2 | 2 | Every matched pair shows modality-dependent damage above 1pp. |
| 0.010 | 3.15 | 2 | 2 | Every matched pair shows modality-dependent damage above 1pp. |
| 0.020 | 6.29 | 3 | 3 | Every matched pair shows modality-dependent damage above 1pp. |
| 0.050 | 15.73 | 3 | 3 | Every matched pair shows modality-dependent damage above 1pp. |

The falsification **does generalise within this factorial**. At the strictest
0.005 tolerance, two independent matched-drift pairs are available and both
exceed the 1pp equivalence margin; the same two persist at 0.01. A third pair
enters at 0.02 and remains at 0.05, and all three exceed 1pp. Thus the result
does not rest only on the original 0.0861 crossing or on a permissive
post-hoc tolerance.

The comparisons also falsify either measured variable as a sufficient
univariate explanation in complementary ways. Image-only age 4 versus
text-only age 4 is matched on nominal age but differs in cumulative drift by
0.0177 and in degradation by 6.72pp. Conversely, image-only age 16 versus
text-only age 64 is effectively matched on cumulative drift (difference
0.000068) despite unequal residence ages, yet differs in degradation by
4.74pp. A large modality-dependent gap therefore remains both when age is
held constant and when cumulative drift is held approximately constant.
Neither age nor cumulative projector drift alone predicts the outcome.

The correct mechanism framing is: **projector drift is necessary but not
sufficient**. The controlled matched-drift comparisons falsify drift as a
complete explanation. Representation geometry, gradient sensitivity and
false-negative structure remain candidates, but this design cannot separate
them and no preference among them is claimed.

## Drift saturation

The step-0 distance crossed 0.95 for both modalities at optimizer step 160
(epoch 2), so absolute distance from initialization did saturate early.
The design-relevant movement from the previous measurement did not remain
high:

- Image final/first-epoch ratio:
  0.00001459 (0.145014
  to 0.00000212).
- Text final/first-epoch ratio:
  0.00000526 (0.146512
  to 0.00000077).

Both are far below the independently applied 0.20 threshold. Thus the
pre-registered drift-saturation risk did **not** materialise: continuing
projector motion decayed enough for age to remain discriminating.

## Efficiency cost–benefit

| evidence | comparison | performance_effect_pp | throughput_images_per_second | peak_gpu_memory_gib | available_gpu_memory_gib | interpretation |
|---|---|---|---|---|---|---|
| Original Wave-0 queue configuration | queue 16,384 vs no queue | -18.99 | — | — | — | Approximately 19pp lower R@1 |
| Factorial maximum-capacity smoke | batch 1024, queue 65,536 | — | 1676.2 | 8.88 | 95.01 | No required memory saving; 86.13 GiB headroom |
| Factorial maximum-capacity smoke | batch 512, queue 32,768 | — | 1866.2 | 2.80 | 95.01 | No post-fill throughput degradation |

The measurements come from two explicitly different controls: the ~19pp
performance effect is the accepted Wave-0 dose-response at the originally
used configuration, while throughput and memory stress come from the
factorial's maximum-capacity smoke. The queue offered no favourable trade in
this regime: it sharply reduced retrieval quality, did not save wall-clock
time, and memory capacity was not constraining. Post-fill throughput was
1,676 images/s at batch 1024 and 1,866 images/s at batch 512; the largest
smoke case used 8.88GiB of 95.01GiB.

## Frozen prediction scorecard

| Prediction | Score | Interpretation |
|---|---|---|
| `age_vs_count` | PASS AT BOUNDARY | Primary rule passes exactly 6/8; this is qualified evidence. |
| `modality_asymmetry` | MISS | The observed age-dependent crossover contradicts the directional prediction. |
| `harm_threshold` | NO PRIOR | The observed threshold is age 1 for both batches. |
| `seed_replication` | HIT | Ordering was stable whenever means differed by more than 1pp. |

The frozen predictions file was read but not modified.

This scorecard is also a methods result: there are two hits
(`age_vs_count`, at the boundary, and `seed_replication`), one miss
(`modality_asymmetry`) and one declared no-prior (`harm_threshold`). The miss
produced the chapter's most substantive mechanism result—the modality
crossover. Pre-registration exposed that contradiction; a post-hoc narrative
could instead have absorbed the crossover as apparent confirmation.

## Dissertation claim supported

For frozen DINOv3 ViT-S/16 plus frozen MiniLM with a trainable projector,
cross-batch memory built from post-projection embeddings is unsafe even at
one-step residence. Damage is primarily organised by age under the
pre-registered boundary rule, but the modality crossover is not explained by
cumulative projector drift alone. This supports a scoped claim about
projector-only frozen alignment—not a universal claim that memory queues are
harmful in all contrastive learning.
