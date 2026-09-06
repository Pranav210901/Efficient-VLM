# Efficient frozen VLM alignment: research landscape and recommendation

## Executive decision

Do not make 224-pixel retraining, generic token pruning, early exits,
Matryoshka embeddings, quantization, compilation or cascades the contribution.
They are controls.

The strongest dissertation-sized mechanism is:

> **Cross-modal-neighbourhood-preserving conditional token routing for
> heterogeneous frozen encoders.**

A small image-only router operates inside frozen DINOv3. It chooses a fixed
hardware-friendly token budget per image. It is trained using all captions for
the image to preserve the full-token model's bidirectional cross-modal
neighbourhood, while DINOv3 and the sentence encoder remain frozen. At
inference the router sees only the image, so gallery embeddings remain
query-independent and precomputable.

Novelty confidence is medium, not guaranteed. Token pruning itself is crowded.
The defensible distinction is the combination of:

1. independently pretrained heterogeneous frozen experts;
2. a single global dual-encoder retrieval space;
3. query-independent internal vision-token reduction;
4. cross-caption, bidirectional neighbourhood preservation rather than
   classifier attention or a jointly pretrained CLIP signal;
5. fewer than 5M trainable parameters and measured latency as the budget.

## Why this direction

The measured MiniLM student latency is 9.65 ms image-side and 8.09 ms
text-side at batch 64. Thus neither modality alone explains the 17.35 ms total.
Visual token routing can improve at most the image portion. Dynamic text
padding/context and compilation must be measured as deployment controls, while
the router supplies the research mechanism.

The clean Flickr30k result is 42.40% mean bidirectional R@1, versus 68.22% for
OpenCLIP, 78.24% for MobileCLIP2 and 80.58% for SigLIP2. Compression cannot
plausibly recover this 26–38pp gap. Accuracy recovery and efficiency therefore
need two coordinated tracks:

- corrected-recipe distillation/data work for performance;
- retrieval-aware conditional token computation for efficiency.

Their eventual combination is the candidate final model.

## What is already occupied

| Mechanism | Status | Dissertation use |
|---|---|---|
| 224-pixel retraining | established scaling control | required baseline |
| Generic pruning/merging | crowded: DynamicViT, ToMe, ATS, DiffRate | required baseline |
| CLIP/VLM-guided pruning | directly occupied: MADTP, Patch Ranking, Attentive Mask CLIP | do not claim |
| Early exits | occupied, including adaptive image-text retrieval | optional baseline |
| Bi-encoder cascade | directly occupied | systems baseline only |
| Structured CLIP pruning | occupied by MoPE-CLIP and others | avoid |
| Matryoshka dimensions | mature retrieval/storage method | optional storage baseline |
| Quantization | mature; kernel-dependent | deployment baseline |
| Distillation | mature | accuracy mechanism, not novelty |
| Compilation/fused attention | engineering | deployment baseline |

Particularly close work includes MADTP for alignment-guided pruning, ICAR for
adaptive early-exit image-text retrieval, and the July 2026 SaMer preprint for
frozen-encoder token compression in retrieval. Claims must be narrower than
all three.

## Proposed mechanism

### Full-token teacher

Use the locked no-queue student as a full-token teacher. It supplies:

- its normalized image embedding;
- similarities to all five captions of an image;
- local cross-modal neighbour rankings within the batch.

The teacher is frozen and cached where valid.

### Image-only router

Insert a small scorer/merger at one or two DINOv3 blocks. It receives image
tokens only and selects among fixed token-count buckets. Fixed buckets matter:
arbitrary per-sample sparsity often lowers FLOPs without lowering batched GPU
latency.

The router/projector objective combines:

1. the locked no-queue contrastive retrieval loss;
2. full-token image-embedding preservation;
3. bidirectional cross-modal similarity/ranking preservation;
4. cross-caption consensus across all captions belonging to the image;
5. an explicit measured-compute or token-budget penalty.

Only the router and projectors train. The combined trainable count remains
below 5M.

### Why cross-caption consensus matters

Caption-conditioned inference would destroy gallery precomputation. Training
with several captions can nevertheless teach an image-only router which visual
evidence is consistently useful across descriptions. That deployment
constraint distinguishes this proposal from query-conditioned pruning in
fused/generative VLMs.

## Experimental sequence

### Stage A — controls and feasibility

1. Retrain the locked student at 224 pixels, three seeds.
2. Measure realistic dynamic text padding in addition to the frozen
   native-context frontier; never overwrite the pre-registered frontier.
3. Profile eager, compiled and fused-attention execution.
4. Apply training-free ToMe/ATS and random token removal at matched realised
   latency.
5. Establish fixed token-budget curves before training a dynamic router.

Gate: continue only if reducing tokens produces a real measured latency change
and at least one reduced budget retains a scientifically useful fraction of
the full-token retrieval result. The exact threshold must be pre-registered
before results are viewed.

### Stage B — learned retrieval-aware routing

Train one seed across a small set of fixed token budgets. Compare:

- random removal;
- CLS-attention or energy ranking;
- ToMe/ATS;
- embedding-only preservation;
- cross-modal-neighbourhood preservation;
- neighbourhood preservation plus cross-caption consensus.

This isolates whether the proposed training signal contributes beyond an
existing pruning primitive.

### Stage C — adaptive budget and replication

If Stage B succeeds, let an image-only gate choose among the validated fixed
budgets. Confirm the selected method with seeds 43 and 44. Report:

- Flickr30k R@1;
- contaminated COCO validation separately;
- realised token distribution;
- actual image, text and full-stack latency;
- FLOPs, parameters, trainable parameters and memory;
- calibration and per-complexity failure cases;
- fixed-budget oracle versus learned router.

### Stage D — performance recovery

Re-run the corrected-baseline distillation question: does either teacher add
performance above the 42.40% Flickr / corrected-recipe baseline? Use fresh
matched baselines and the same clean evaluation discipline. If distillation
helps, apply it to the efficient configuration and test whether gains and
efficiency compose.

## Falsification criteria

The mechanism fails scientifically if:

- existing ToMe/ATS matches it at equal measured latency and R@1;
- dynamic routing does not beat the convex hull of fixed budgets;
- routing overhead cancels encoder savings at batch 64;
- cross-caption/neighbourhood losses do not outperform embedding preservation;
- the result exists only on contaminated COCO and not Flickr30k;
- a newly found paper already covers the exact frozen heterogeneous
  global-retrieval setting.

A negative result remains useful if it shows that FLOP/token reduction does not
translate into batched latency for frozen SSL encoders.

## Hypothesis assessment

1. Static resolution alone is not novel: supported.
2. Retrieval-aware conditional computation is the best available gap:
   provisionally supported, medium confidence.
3. A small router can retain accuracy while reducing latency: undetermined and
   experimentally falsifiable.
4. Parameter count is an incomplete efficiency proxy: supported by the current
   frontier, where the 45.78M student is slower than 151.28M OpenCLIP.

## Adversarial assessment

The strongest objection is that this is “MADTP/Patch Ranking on DINO.” The
work is defensible only if the cross-caption neighbourhood objective,
query-independent gallery semantics, heterogeneous frozen encoders and
sub-5M constraint are central and ablated. If the method reduces to another
token score, it is not a sufficient contribution.

The second objection is that efficiency work distracts from a large accuracy
gap. This is valid. The dissertation must not imply compression closes the
gap; distillation/data work addresses performance, while routing tests whether
the frozen-expert approach can move to a better measured Pareto point.
