# Prior art for the Alignment project

_Systematic literature map completed 2026-07-26_

## Executive conclusion

The project’s broad architecture is established prior art: Freeze-Align, SAIL,
ShareLock and STRUCTURE already align pretrained frozen unimodal encoders using
small trainable mappings. Sigmoid loss, multiple positive captions, detached
memories, momentum queues and vision-language distillation are also established
components.

The strongest defensible gap found in this review is narrower:

> A controlled study of modality-specific historical-feature age in
> projector-only alignment between independently frozen vision and text
> encoders.

Cross-Batch Memory already identifies feature drift and ablates memory ratio
and batch size. Adaptive Cross Batch Normalization already sweeps batch size
and memory size, reports cases where ordinary XBM is worse than no memory, and
corrects stale accumulated embeddings. The potential contribution is therefore
not “discovering staleness,” a generic queue dose response, or a generic
capacity/batch interaction. It is determining how drift behaves when two
separately pretrained modalities have distinct age distributions and are
connected through rapidly changing lightweight projectors.

No directly matching cross-modal, dual-frozen, projector-only,
modality-separated queue-age study was found in this corpus. This is a
qualified search result, not proof that none exists.

## Critical correction to the proposed directional hypothesis

The implementation computes:

```text
i2t loss: current image anchors × current/historical text gallery
t2i loss: current text anchors  × current/historical image gallery
```

With all captions, the text queue refreshes approximately five times faster
than the image queue at the same numerical capacity. The older image gallery
therefore directly perturbs the **t2i loss branch**, while a text-only queue
directly perturbs the **i2t loss branch**.

This does not guarantee the same ordering in final i2t/t2i R@1 because both
projectors receive gradients from the bidirectional loss. The confirmatory
predictions should therefore be:

1. Image-only history changes the t2i branch loss/gradient more than the i2t
   branch.
2. Text-only history changes the i2t branch loss/gradient more than the t2i
   branch.
3. A secondary, low-confidence prediction is that full-queue t2i R@1 degrades
   more than i2t if modality-specific age dominates the coupled optimisation.

The secondary sign prediction is already contradicted by the observed
zero/16,384 endpoints:

| Batch | i2t R@1 drop | t2i R@1 drop |
|---:|---:|---:|
| 512 | 24.94pp | 16.31pp |
| 1,024 | 23.48pp | 14.50pp |

Thus, the current exploratory evidence shows **greater i2t degradation**, not
greater t2i degradation. This may reflect coupling, different branch
sensitivities, baseline headroom or another mechanism. It cannot be repackaged
as preregistered evidence. For new experiments, the image-only/text-only
branch-level hypotheses are primary; any full-queue downstream sign is
secondary and low-confidence.

## Verification levels

The source index explicitly distinguishes:

- `full_text_and_supplement`;
- `full_text`;
- `selected_fulltext_sections`;
- `primary_abstract`.

SAIL, XBM and their supplements were inspected in full. Adaptive Cross Batch
Normalization’s full source was inspected. Other entries retain the narrower
verification label shown in [sources.csv](sources.csv); inclusion in this map
does not imply every paper was read end-to-end.

## Direct competitors

| Paper | What overlaps | What remains different |
|---|---|---|
| [Freeze-Align (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Maniparambil_Harnessing_Frozen_Unimodal_Encoders_for_Flexible_Multimodal_Alignment_CVPR_2025_paper.html) | Frozen independent towers, lightweight MLP projectors, pair selection, retrieval | Does not provide the present modality-specific queue-age study |
| [SAIL (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.html) | Frozen towers, alignment layers, sigmoid loss, multiple captions | Uses pre-encoded base features and current batches up to 32,768; no queue ablation found |
| [ShareLock (TMLR 2026)](https://openreview.net/forum?id=wqBHJNqeQJ) | Frozen pretrained models, lightweight alignment, reusable features | No matching queue-age mechanism study found |
| [STRUCTURE (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/dee8f820d86aca28ab0328a9243020f9-Paper-Conference.pdf) | Frozen unimodal models, small mappings, paired-data efficiency | Focuses on representation geometry |
| [LiT (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_LiT_Zero-Shot_Transfer_With_Locked-Image_Text_Tuning_CVPR_2022_paper.html) | Locked image tower and contrastive image-text training | Text side is trained rather than independently frozen |
| [ASIF (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html) | Connects independently trained modalities | Training-free anchor method rather than learned projectors |
| [dino.txt (2024)](https://arxiv.org/abs/2412.16334) | Frozen DINO-family representation aligned with text | Different alignment modules and no matching queue experiment |

### Claim consequence

Do not claim a new frozen-alignment architecture, lightweight projector, or
DINO-to-text alignment paradigm. Treat these papers as required direct
baselines or related work.

## Representation compatibility

| Paper | Relevance |
|---|---|
| [The Platonic Representation Hypothesis (ICML 2024)](https://proceedings.mlr.press/v235/huh24a.html) | Theoretical/empirical context for convergence between separately learned representations |
| [Do Vision and Language Encoders Represent the World Similarly? (2024)](https://arxiv.org/abs/2401.05224) | Direct diagnostics of structural similarity across vision and language encoders |

These works support the project’s motivation, but they also mean that latent
cross-modal compatibility is not a new premise.

## Losses, captions and negative count

| Paper | Component already studied | Consequence |
|---|---|---|
| [SigLIP (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) | Pairwise sigmoid image-text loss; negative/positive ratio; batch scaling | Sigmoid itself is not novel |
| [FFF (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Bulat_FFF_Fixing_Flawed_Foundations_in_Contrastive_Pre-Training_Results_in_Very_CVPR_2024_paper.html) | Multiple pseudo-captions, true positives and sigmoid loss | All-caption sigmoid is not novel |
| [UniCL (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Unified_Contrastive_Learning_in_Image-Text-Label_Space_CVPR_2022_paper.html) | Multi-positive visual-semantic contrastive targets | Multi-positive InfoNCE has direct precedent |
| [Supervised Contrastive Learning (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | General many-positive contrastive objective | Mathematical component is established |
| [CLIP-Lite (AISTATS 2023)](https://proceedings.mlr.press/v206/shrivastava23a.html) | One negative per positive and small-batch efficiency | More negatives are not universally required |

These sources make the age/count separation essential. A decreasing
capacity-performance curve alone cannot establish that stale age, rather than
the changing negative distribution, caused the result.

## Queues, feature drift and historical negatives

| Paper | What it establishes | Relation to this project |
|---|---|---|
| [Cross-Batch Memory (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html) | Historical embeddings help if features drift slowly; memory-ratio and batch-size ablations | Strongest prior art; generic capacity and batch studies are not new |
| [Momentum Contrast (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) | Queue consistency through a momentum key encoder | Project queue has no momentum encoder |
| [Adaptive Cross Batch Normalization (2023)](https://arxiv.org/abs/2303.17127) | Batch sweep at fixed memory, memory sweep at fixed batch, XBM instability and moment correction | Leaves a modality-asymmetric cross-modal gap, not a generic age/capacity gap |
| [Hard Negative Mixing (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/f7cade80b7cc92b991cf4d2806d6bd78-Abstract.html) | Hard-negative quality and diminishing returns from large memories | Supports measuring quality/count separately from age |

### Current project evidence

At batch 1,024 and one seed, the existing training-time best-dev curve is:

| Queue capacity | Mean bidirectional R@1 |
|---:|---:|
| 0 | 35.76% |
| 1,024 | 30.06% |
| 4,096 | 24.78% |
| 8,192 | 21.21% |
| 16,384 | 16.77% |

This is strong monotonic evidence but remains observational because capacity
changes both historical-negative count and age. It is also not, by itself, a
new class of ablation relative to XBM and Adaptive Cross Batch Normalization.

## Distillation

| Paper | Overlap |
|---|---|
| [TinyCLIP (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html) | Cross-modal affinity mimicking |
| [CLIP-KD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html) | Feature, relation and contrastive distillation |
| [DIME-FM (2023)](https://arxiv.org/abs/2303.18232) | Efficient multimodal-teacher distillation |
| [MobileCLIP (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html) | Compact VLMs, CLIP/caption-teacher reinforced training |
| [MobileCLIP2 (2025)](https://arxiv.org/abs/2508.20691) | Stronger teacher ensembles, captions and temperature controls |
| [MCAD (NAACL 2024)](https://aclanthology.org/2024.findings-naacl.96/) | Feature/distribution distillation for efficient dual-encoder image-text retrieval |

Teacher caches, cosine feature matching, KL/similarity transfer and compact
paired-VLM teachers are not individually novel. The project’s distillation
result is best treated as a controlled diagnostic of how much teacher geometry
can be captured under a small adaptation budget.

## Efficient retrieval and adaptive experts

| Paper | Overlap | Claim constraint |
|---|---|---|
| [Bi-Encoder Cascades (ICCVW 2023)](https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/Honig_Bi-Encoder_Cascades_for_Efficient_Image_Search_ICCVW_2023_paper.html) | Cheap/expensive text-image retrieval cascade | Efficient bi-encoder cascades already exist |
| [MoVA (2024)](https://arxiv.org/abs/2404.13046) | Context-aware routing/fusion of vision experts | Multiple-vision-expert routing is established |
| [MOVE (2025)](https://arxiv.org/abs/2502.15381) | Automatic selection among specialised vision encoders | Cannot claim first adaptive encoder selection |
| [MoDE CLIP (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_MoDE_CLIP_Data_Experts_via_Clustering_CVPR_2024_paper.html) | Task/query-metadata weighting of CLIP experts | Direct prior art for task-aware expert routing |
| [SCOPE (OpenReview 2025)](https://openreview.net/forum?id=GxEyklHpWB) | One encoder selected per image-text pair | Very close instance-level routing prior art; status must be refreshed |

An adaptive-routing contribution would need to be narrower: heterogeneous,
independently pretrained frozen image/text pairs; inference-time, label-free
signals; and an explicit measured retrieval cost budget. The historical
label-derived Phase 2 routing cannot establish this.

## Reproducibility and pre-registration

| Paper | Relevance |
|---|---|
| [Improving Reproducibility in ML Research (JMLR 2021)](https://www.jmlr.org/papers/v22/20-303.html) | Reproducibility checklists, code policy and reliable workflows |
| [Sources of Irreproducibility in ML (2022)](https://arxiv.org/abs/2204.07610) | Connects design/implementation choices to false experimental conclusions |
| [Preregistering NLP Research (NAACL 2021)](https://aclanthology.org/2021.naacl-main.51/) | Negative results, deviations and confirmatory/exploratory separation |
| [Underspecification (JMLR 2022)](https://www.jmlr.org/papers/v23/20-1335.html) | Warns against generalising beyond what an experimental specification establishes |

The audit-trail contribution is defensible if framed as:

> Pre-registration correctly constrained the decision and prevented
> retrospective gate adjustment, but did not make the tested recipe an
> identifying experiment for architectural capability.

The original v3 report did not claim an architectural ceiling. The
over-generalisation risk arose in later human narration, not in the registered
gate outcome.

## Required confirmatory design

The core design should include:

1. Queue age levels matched across batch 512 and 1,024.
2. Three seeds per condition.
3. Queue-free controls.
4. Mandatory image-only and text-only queues.
5. Actual insertion and consumption age logged separately by modality.
6. Fixed optimiser-update budget, scheduler and evaluation split.
7. i2t and t2i branch losses, gradient norms and similarity distributions.
8. Projector-drift measurement by recomputing stored examples under the current
   projector.
9. Explicit acknowledgement that batch size also changes in-batch negative
   count.

The single-modality experiments are required because they isolate the direct
loss-branch effect better than a shared numerical capacity.

## Adversarial review

### Could this still be “just XBM behaving badly”?

Yes, unless the study measures actual drift and demonstrates a repeatable
cross-modal, modality-specific relationship with age. XBM and Adaptive Cross
Batch Normalization already establish the general phenomenon. A performance
curve alone is insufficient.

### Does matching age identify causality completely?

No. The completed factorial supports age over count only at the boundary of
the pre-registered rule (6/8 matched-age contrasts), and the single-modality
result falsifies projector drift as a sufficient explanation. Cross-modality
conditions with matched cumulative drift retain materially different damage.
The correct claim is that drift is necessary but not sufficient and that a
modality-specific factor remains unidentified. Representation geometry,
gradient sensitivity and false-negative structure remain candidates, but the
factorial cannot separate them.

### Can the existing curve be called preregistered confirmation?

No. It has already been observed. New seeds and new matched-age/modality cells
can be confirmatory; the original curve is hypothesis-generating evidence.

### Is the architecture new?

No. Freeze-Align and SAIL are close enough that an architecture-first novelty
claim would be vulnerable.

### Is the methods/audit contribution merely a bug report?

Not if it derives a general, measured lesson: a correct registered decision
rule can reject a confounded recipe without identifying the capability of the
architecture under study.

## Defensible dissertation contribution

Before the new controls:

> We observe a strong monotonic association between detached memory capacity
> and degraded retrieval in dual-frozen projector alignment, and hypothesise
> that asymmetric modality-specific historical-feature age contributes to the
> mechanism.

After the completed matched-age and single-modality factorial:

> We show that post-projection cross-batch memory has no safe non-zero age in
> dual-frozen projector alignment, with harm already measurable at one
> optimizer step. Matched-age controls support age over entry count at the
> boundary of the pre-registered rule, while replicated cross-modality
> matched-drift counterexamples demonstrate that projector drift is necessary
> but not sufficient. A modality-specific factor remains unidentified under
> the controlled performance–efficiency protocol.

## Research-artifact index

- [Search plan](plan.md)
- [Source index](sources.csv)
- [Atomic findings](findings/)
- [Per-paper evidence notes](sources/)
- [Refresh protocol](refresh_targets.md)
