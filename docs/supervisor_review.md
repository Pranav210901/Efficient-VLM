# Supervisor Review

> **Historical-branch notice (8 August 2026).** This review accurately reports
> the original multi-expert Phase 1–2 branch, but it is not the current whole
> dissertation. The integrated frozen-alignment study and all later follow-ups
> are reported in [`dissertation.md`](dissertation.md). Phase 2 is confirmed
> computationally complete and registered as `A01`.

**Project:** Adaptive alignment of lightweight pretrained vision and text encoders  
**Research question:** Can multiple lightweight pretrained vision and text encoders be adaptively combined across multiple vision-language tasks to achieve a better performance–efficiency trade-off than a single fixed encoder pair?
**Status:** Phases 1, 1.5 and 2 completed  
**Prepared:** 23 July 2026  
**Main notebook:** [01_experiment_workflow.ipynb](../notebooks/01_experiment_workflow.ipynb)

## Executive summary

Phase 1 screened lightweight frozen vision and text encoders on COCO retrieval, including baseline and bidirectional latent fusion (BLF) variants. DINOv2 + MiniLM was the strongest retrieval family, but BLF gains were small and not reliable across seeds. Phase 1.5 then measured multi-task performance, error complementarity and efficiency, and selected four diverse expert paths using leakage-safe development evidence. Phase 2 trained a lightweight cross-attention reranker over each frozen path and an adaptive router over all four paths.

The main Phase 2 result is mixed: cross-attention greatly improved some zero-shot classification-development scores, but reduced COCO retrieval performance. Smaller candidate sets worked best, and random-negative training unexpectedly outperformed hard-negative training. The adaptive router learned different path preferences but has not yet demonstrated a reliable end-to-end improvement over the best single bridge. Phase 2 is computationally complete; the next work should correct the dense-teacher comparison metric and improve retrieval training before final held-out testing.

## Phase 1 — encoder and BLF screening

Five vision encoders and four text encoders were combined into 20 baseline pairs; local, global and local+global BLF produced an 80-configuration COCO retrieval grid. The encoders were frozen and small trainable projection/fusion heads aligned their representations.

| Best grid result | Model | Mean R@1 |
|---|---|---:|
| Baseline | DINOv2 + MiniLM | 14.69% |
| Global BLF | DINOv2 + MiniLM | 14.85% |
| Local BLF | DINOv2 + MiniLM | 15.28% |
| Local+global BLF | DINOv2 + MiniLM | **15.38%** |

On the later full COCO development evaluation, DINOv2 + MiniLM local reached **19.52% i2t R@1** and **16.05% t2i R@1**. However, five-seed matched comparisons were inconclusive:

| BLF comparison | Mean change | Wins/losses | 95% interval | Conclusion |
|---|---:|---:|---:|---|
| DINOv2 + MiniLM local | +0.55 pp | 4/1 | −0.30 to +1.39 pp | Inconclusive |
| ConvNeXtV2 + MiniLM local+global | +0.48 pp | 3/2 | −0.87 to +1.83 pp | Inconclusive |

**Interpretation:** BLF can add useful local/global features, but the gains are small relative to seed variation. BLF also improved retrieval more consistently than classification, so it was retained as an ablation rather than treated as a universally better architecture.

### Phase 1 loss and convergence

Phase 1 used the symmetric CLIP contrastive loss:

\[
\mathcal{L}_{CLIP}=\tfrac{1}{2}\left[
CE(S,\mathrm{diag})+CE(S^\top,\mathrm{diag})
\right],
\]

where \(S\) is the batch image–text similarity matrix. The first term retrieves the matching text for each image and the second retrieves the matching image for each text.

Training loss fell strongly, but validation loss normally reached its minimum around epochs 2–3 and then increased. For DINOv2 + MiniLM baseline, training loss fell from 0.885 to 0.038, while validation loss improved from 0.634 to **0.570 at epoch 3** and then worsened to 0.643. Retrieval metrics followed the validation loss rather than the final training loss.

**Convergence assessment:** optimization converged, but the later epochs overfit. The saved best-validation checkpoints, rather than final-epoch weights, are therefore the valid Phase 1 models.

## Phase 1.5 — multi-task evidence and expert selection

The 23 shortlisted baseline/BLF configurations were evaluated on COCO retrieval and development splits derived from CIFAR-100 train, Pets trainval and EuroSAT train. Phase 1.5 added prediction-level complementarity, oracle upper bounds, cross-task specialisation, seed reliability and latency/memory profiling.

| Development task | Best configuration | Result |
|---|---|---:|
| COCO i2t R@1 | DINOv2 + MiniLM local | **19.52%** |
| COCO t2i R@1 | DINOv2 + MiniLM local | **16.05%** |
| CIFAR-100 Top-1 | ConvNeXtV2 + MiniLM baseline | **29.79%** |
| Pets Top-1 | DINOv2 + E5 baseline | **7.34%** |
| EuroSAT Top-1 | DINOv2 + BGE baseline | **22.27%** |

Individual zero-shot accuracy was modest, but error diversity was large. An ideal, non-deployable oracle over the available experts reached approximately **31.2% COCO i2t R@1, 69.1% CIFAR, 43.5% Pets and 73.6% EuroSAT**. This gap motivated adaptive expert combination: different models solved different samples even when their average scores were similar.

Four primary paths were selected by performance, unique wins, oracle contribution, architecture diversity and efficiency:

| Selected Phase 2 path | Selection score | Purpose |
|---|---:|---|
| DINOv2 + MiniLM | **0.732** | General semantic anchor |
| DINOv2 + BGE | 0.725 | Strong complementary semantic path |
| ConvNeXt + MiniLM | 0.645 | Faster hierarchical/local path |
| EfficientNet + BGE | 0.498 | Low-cost complementary path |

Training negatives were mined separately from COCO train2017, CIFAR train, Pets trainval and EuroSAT train. Held-out benchmark errors were marked diagnostic-only. The audit found **zero protected-test overlap** and Phase 1.5 passed **23/23 readiness checks**.

Phase 1.5 did not train a new predictive model, so it has no optimization loss to assess.

## Phase 2 — cross-attention reranking and adaptive routing

For each selected path, the dual encoder first retrieved the top \(K\) candidates. A trainable bridge then reranked only those candidates:

```text
frozen image tokens ─┐
                     ├─ projections → text-to-vision cross-attention
frozen text tokens ──┘               → pooling → pair score
```

The canonical bridge used one 256-dimensional Transformer cross-attention layer, four heads and masked-mean pooling. The pretrained encoders remained frozen.

### Bridge loss and convergence

The bridge optimized:

\[
\mathcal{L}_{bridge}
=\mathcal{L}_{ITM}
+\mathcal{L}_{rank}
+0.05\mathcal{L}_{contrastive}.
\]

- **Image–text matching loss:** binary cross-entropy; matching pairs should score high and negatives low.
- **Ranking loss:** margin loss with margin 0.2; a positive should score at least 0.2 above a negative.
- **Projection contrastive loss:** a small regularizer encouraging positive interaction features to separate from negative features.

All four bridge training losses decreased smoothly over 12 epochs:

| Bridge | Initial recorded loss | Final/best loss | Convergence |
|---|---:|---:|---|
| ConvNeXt + MiniLM | 0.340 | 0.0148 | Stable decrease |
| DINOv2 + MiniLM | 0.369 | 0.0144 | Stable decrease |
| DINOv2 + BGE | 0.363 | 0.0124 | Stable decrease |
| EfficientNet + BGE | 0.291 | **0.0066** | Stable decrease |

**Convergence assessment:** the training objectives converged. However, low training loss did not guarantee retrieval generalisation; EfficientNet had the lowest loss but was not the best retrieval bridge.

### Bridge evaluation

At candidate depth 16:

| Bridge | COCO i2t R@1 | COCO t2i R@1 | CIFAR | Pets | EuroSAT |
|---|---:|---:|---:|---:|---:|
| ConvNeXt + MiniLM | **10.42%** | **4.43%** | 40.52% | 28.80% | 83.80% |
| DINOv2 + MiniLM | 10.38% | 4.33% | 42.90% | 16.98% | **88.79%** |
| DINOv2 + BGE | 9.66% | 4.24% | **42.92%** | 16.03% | 88.02% |
| EfficientNet + BGE | 6.44% | 3.13% | 27.51% | **34.24%** | 82.35% |

The bridges learned useful class discrimination, but retrieval was worse than the Phase 1 DINOv2 + MiniLM dual encoder (17.92% i2t and 15.44% t2i R@1). Increasing depth raised the chance that the correct item was present but made reranking harder:

| Candidate depth | COCO coverage | Mean directional R@1 | CIFAR Top-1 |
|---:|---:|---:|---:|
| 16 | 66.7% | **7.36%** | **42.90%** |
| 32 | 80.1% | 5.09% | 36.64% |
| 64 | 90.7% | 2.84% | 24.54% |
| 128 | 96.2% | 1.43% | 18.51% |

**Interpretation:** more candidates improved coverage but introduced distractors that the bridge could not reliably order. EuroSAT stayed constant because all 10 labels were already present at depth 16.

### Most important ablation

At depth 64, random-negative training beat hard-negative training on every headline task:

| Negative source | Mean COCO R@1 | CIFAR | Pets | EuroSAT |
|---|---:|---:|---:|---:|
| Hard negatives | 2.80% | 19.84% | 5.71% | 88.32% |
| Random negatives | **8.02%** | **37.16%** | **8.97%** | **92.42%** |

The mined hard negatives were likely too noisy/difficult or encouraged over-specialisation. Negative sampling affected results more than modest changes to heads, layers, bridge width or pooling.

### Dense adaptive teacher

A dense router combined the four bridge scores using static, uniform, task-conditioned, input-conditioned and task-attention strategies. Its loss was binary cross-entropy on the combined score plus a path-balancing term and a small entropy reward to discourage collapse onto one path. Most learned strategies reached a plateau by roughly epochs 13–26; for example task-and-input fell from 0.105 to a minimum of 0.0148 and ended at 0.0161. Task-attention plateaued at a much higher loss (minimum 1.046).

Task-attention produced the largest stored one-run mean score (0.224), but the five-seed canonical task-and-input run did not establish an advantage over the best bridge. There is also a reporting limitation: dense “utility” is currently a mean raw score, whereas best-bridge utility uses inverse retrieval rank and classification correctness. These are not directly comparable, so the reported dense-versus-bridge difference should not be used as a final statistical claim until the metric is unified.

## Overall conclusion and immediate next step

The work supports three conclusions:

1. **Experts are complementary:** Phase 1.5 oracle gaps and task-specific winners justify exploring adaptive selection.
2. **The current bridge is task-dependent:** it improved classification-development performance but harmed COCO retrieval.
3. **Training design matters more than small architecture changes:** random negatives and small candidate sets were substantially better than the intended hard-negative/large-candidate setup.

The immediate next step is to recompute dense-teacher evaluation with one common end-to-end utility, then retrain retrieval bridges using random or mixed curriculum negatives and validate the chosen setting across seeds. Only after freezing that decision should the reserved final test splits be evaluated.

**Authoritative outputs:** [Phase 1.5 selection](../results/phase15/expert_selection/selected_experts.yaml), [Phase 1.5 readiness](../results/phase15/phase2_readiness/readiness_report.md), and [Phase 2 report](../results/phase2/phase2_report.md).
