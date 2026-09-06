# Preregistration: Guarded Hard-Negative Fine-Ranking Study

**Status:** `FROZEN`  
**Freeze rule:** this file and `threshold_calibration.json` are immutable after
the SHA-256 digests are written to the freeze receipt. Mining and training must
abort if either digest differs.

## 1. Motivation

**Primary.** Existing three-seed Flickr30k-validation results show a large
R@1-to-R@5 gap in both directions: t2i R@1 0.464 versus R@5 0.750, and i2t
R@1 0.613 versus R@5 0.845. Correct items are frequently in the local
neighbourhood but are not ranked first. This is the claim-bearing evidence
for a fine-ranking deficit.

**Supporting.** The audit of 50 systematic t2i failures found fine-grained
attribute errors to be the largest category and also found semantically valid
retrieval alternatives. The sample is conditioned on all three seeds missing
the top 10 and is not a prevalence estimate. It corroborates, but does not
establish, the motivation.

## 2. Hypotheses

- **H1:** guarded hard-negative fine-tuning improves three-seed Flickr30k-
  validation mean bidirectional R@1 over a matched-update control.
- **H2:** the guard is necessary: Guarded HN exceeds Naive HN because
  unfiltered neighbours contain semantically valid alternatives.

## 3. Arms and fixed replication

Three arms at seeds 42, 43 and 44: nine training runs total.

| Arm | Batch-neighbour source |
|---|---|
| Control | Deterministic random different-image neighbours |
| Naive HN | Unfiltered top-64 neighbours mined by the frozen source ensemble |
| Guarded HN | Top-64 neighbours remaining after every rule in Section 6 |

All arms use the same paired-batch construction and the same number of
updates. Therefore the intervention is neighbour selection, not extra forward
passes, a new loss weight, or a different sample count.

## 4. Frozen mining source

- Architecture: M_T1 (224 px C4 vision aggregator plus MiniLM learned-query
  text aggregation and MobileCLIP2-S0 distillation).
- Checkpoints: seed 42 epoch 22, seed 43 epoch 22, seed 44 epoch 20, with the
  identities already recorded by the paired repeated-latency diagnostic.
- Mining pool: the COCO training split used by M_T1; Flickr validation and test
  are prohibited.
- Mining is bidirectional (i2t and t2i), static and performed exactly once.
- The sampler operates on image owners because the locked recipe batches
  images with all of their captions. For an ordered pair of different images
  A and B, its mining score is the mean of: (a) the maximum ensemble-mean
  cosine from A's image embedding to any caption owned by B, and (b) the
  maximum ensemble-mean cosine from any caption owned by A to B's image
  embedding. The top 64 distinct candidate image owners are ranked by this
  symmetric bidirectional score.
- Each checkpoint produces L2-normalised cosine similarities. The mining score
  is the arithmetic mean of the three seed-level similarities. Similarities,
  rather than embeddings, are averaged because independently trained joint
  spaces need not share a coordinate basis.
- Ties are broken by frozen source-row index ascending.
- **Candidate depth: 64.** This matches the repository's established retrieval
  candidate depth and is large relative to the eight retained neighbours.
- **Retained negatives per positive: 8.** This matches the existing hard-
  negative mining constant and provides a fixed pool from which the paired
  sampler rotates neighbours.
- No refresh or seed-specific re-mining is permitted.

## 5. Paired-batch semantics

The batch-size unit remains **images**, exactly as in M_T1. Every 1,024-image
training batch is built as 512 anchor images plus one distinct different-image
neighbour for each anchor. Every selected image contributes all of its owned
captions through the existing grouped collate function; consequently a batch
contains 1,024 unique images and a variable number of caption rows (about five
per image), not 1,024 image-caption pairs. `captions_per_image` remains `null`.
This all-caption treatment and the number of image owners are identical in all
three arms.

A **collision** means either (a) a proposed neighbour is already one of the
512 anchors in that batch, or (b) the same image owner is proposed for more
than one anchor. Anchors are the next 512 unique IDs in the seed-specific
shuffle. Candidate lists are cyclically rotated by a deterministic hash of
training seed, epoch and anchor occurrence. A deterministic bipartite matching
then processes anchors in batch-position order and candidate edges in that
rotated order, augmenting earlier matches when required; it must return 512
distinct neighbours that are disjoint from the anchor set. There is no
epoch-level prohibition on reusing a neighbour in a later batch. If a perfect
matching cannot be formed, the run fails loudly and the study terminates; it
does not backfill from outside the frozen candidate graph. The sampler records
the consumed image and caption keys plus the chosen anchor-neighbour edge.

The Control arm uses the identical matching mechanism, but each anchor's
candidate ordering is a deterministic permutation of all different-image
owners rather than the mined top 64. Naive and Guarded rotate through their
retained pools of eight. This changes data order across seeds, as in the rest
of the project, while the frozen candidate graphs remain identical.

The existing multi-positive InfoNCE and MobileCLIP2 distillation objectives are
unchanged. No additional margin loss or hard-negative loss coefficient is
introduced. Hard negatives act only through controlled batch composition.

## 6. Guarded-arm exclusions

Rules are applied in order, and a candidate excluded by any rule remains
excluded.

1. **Positive ownership:** remove the positive image and every caption owned
   by that image.
2. **Teacher cross-modal guard:** exclude when frozen MobileCLIP2-S0 image-text
   cosine similarity is at least **0.222935**. For i2t evidence this is the
   query-image to candidate-caption similarity, taking the maximum over all
   captions owned by the candidate image. For t2i evidence this is the
   query-caption to candidate-image similarity, taking the maximum over all
   captions owned by the query image. A candidate image is excluded if either
   directional maximum reaches the threshold.
3. **Caption-semantic guard:** exclude when frozen MobileCLIP2-S0 text-text
   cosine similarity is at least **0.765541**. For i2t evidence this is the
   maximum similarity between a candidate caption and any positive caption
   owned by the query image. For t2i evidence this is the maximum similarity
   between a query caption and any caption owned by the candidate image. A
   candidate image is excluded if either directional maximum reaches the
   threshold. This is deliberately conservative about plausible alternatives.

Both thresholds are the preregistered 99th percentiles of 1,000,000
deterministically sampled different-owner COCO-training pairs (calibration
seed 20260801). They were computed from the already frozen MobileCLIP2 cache,
before hard-negative mining, and were not selected using validation outcomes
or mined-neighbour exclusion rates.

Exclusion counts by rule, retained counts and query coverage are diagnostics
only and cannot alter thresholds. Mining is feasible only if every query has
eight valid candidates within its frozen top 64. Otherwise the entire study
is abandoned and reported as `INFEASIBLE_UNDER_PREREGISTERED_GUARD`; there is
no backfill, threshold adjustment or candidate-depth expansion.

## 7. Training budget and continuation

- **Update budget: 330 optimizer steps**, exactly 12.5% of the original
  24-epoch/2,640-step schedule and equivalent to three original COCO epochs.
- Each seed starts from its own frozen M_T1 selected checkpoint listed in
  Section 4.
- Model weights are loaded, but optimizer and scheduler state are deliberately
  reinitialised for the bounded continuation in every arm.
- Fresh AdamW and a single 330-step cosine schedule are used, retaining the
  M_T1 5% warm-up fraction and all other M_T1 hyperparameters. This reset is
  applied identically to all arms and must be disclosed; the resulting runs
  are bounded fine-tuning continuations, not resumes of the original schedule.
- Frozen encoders, 224 px input, C4 image aggregation, M_T1 text aggregation,
  distillation, BF16, batch size and all other recipe fields remain unchanged.
- No early stopping. All 330 steps complete. Recovery checkpoints do not alter
  the schedule or selection.

## 8. Metrics and decision rules

- **Primary selection metric:** Flickr30k-validation mean bidirectional R@1.
- **Required non-selecting diagnostics:** i2t and t2i R@1/R@5/R@10 and MRR.
- Pooled SD for arms a and b is `sqrt((SD_a^2 + SD_b^2) / 2)`.
- Guarded replaces M_T1 only if its three-seed primary mean exceeds M_T1 by
  more than one pooled SD and the unchanged latency/parameter contract is
  verified.
- H2 holds only if Guarded exceeds Naive by more than one pooled SD.
- Directional regressions are reported even if bidirectional mean improves.
- Exclusion rates and intermediate checkpoints cannot select or tune a model.

The one-pooled-SD rule is a preregistered practical-effect heuristic inherited
from the project's earlier selection protocol; it is **not** a formal
significance test. With three seeds per arm it has limited power and nontrivial
false-positive and false-negative risk. Results are therefore reported with
all seed values, means, SDs and ranges, and claims remain descriptive.

## 9. Named outcomes

| Outcome | Interpretation |
|---|---|
| Neither HN arm beats Control | Fine-ranking errors were not recoverable through static batch-level negative selection at this budget; M_T1 stands. |
| Both beat Control; Guarded does not clear Naive | Hard negatives help; filtering is unproven. |
| Guarded clears Control and Naive | H1 and H2 supported; promotion still requires the M_T1 pooled-SD rule. |
| Naive beats Guarded | The guard removed useful signal; report without retuning. |
| Guard cannot retain eight candidates for every query | Study abandoned as preregistered infeasible. |
| Any training batch lacks a perfect 512-anchor/512-neighbour matching | `INFEASIBLE_BATCH_MATCHING`: terminate the entire study immediately, including all other arms and seeds; retain and report completed partial artifacts, but do not replace, backfill or rerun any job. |

## 10. Absolute stopping rule

**After these nine runs complete, no further training occurs regardless of
outcome.** This includes no threshold retuning, new arms, added seeds,
follow-up variants, loss-weight tuning, candidate-depth changes or re-mining.
If the feasibility condition fails before training, the experimental phase
also terminates without a replacement training study. If deterministic
bipartite matching fails during any training run, the whole study terminates
under the named `INFEASIBLE_BATCH_MATCHING` outcome; it is not scoped to only
the affected run, and no remaining or replacement training is permitted.

## 11. Sealed evaluation

Flickr30k test remains sealed until the final configuration is frozen after
this study. All mining, training decisions and promotion use only COCO training
and Flickr30k validation as specified above.
