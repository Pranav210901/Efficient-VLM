# Efficient Alignment of Frozen Vision and Language Encoders Under Tight Adaptation and Latency Budgets

## Dissertation manuscript

**Status:** integrated final-evidence manuscript  
**Date:** 14 August 2026  
**Repository:** `alignment_vlm`  
**Final test status:** Flickr30k test run and reported (three seeds); see §6.1

> This manuscript contains the complete scientific argument supported by the
> repository. It still requires the university's title-page, declaration,
> acknowledgements, word-count, citation-style, and PDF template before formal
> submission. Results labelled “validation” are development evidence and are
> not represented as final test results.

The 25-area literature disposition, comparison hierarchy and irreducible-gap
record are available in
[`literature_gap_closure_20260810.md`](literature_gap_closure_20260810.md).

## Abstract

Contrastive vision–language models acquire their capability by jointly training two large
towers on web-scale paired data, at costs incompatible with many research and deployment
settings. This dissertation asks whether independently pretrained encoders can instead be
aligned while frozen, using fewer than five million trainable inference parameters within a
measured latency budget, against MobileCLIP2-S0 as the compact jointly pretrained reference
and OpenCLIP ViT-B/32 as a locally reconstructed field anchor.

Early results were weak for an unexpected reason, and the principal contribution is
consequently diagnostic. Decomposing the training recipe isolated the cross-batch memory
queue: although the encoders were frozen, queued embeddings sat downstream of a rapidly
changing projector and fell out of correspondence with the current model. Removing it
recovered roughly 19 percentage points of retrieval R@1, and a preregistered factorial
(22 conditions, 66 runs, three seeds) found harm beginning after a single optimizer step with
no safe non-zero age. Conditions matched on measured drift differed in damage by up to 4.74
percentage points, and the text tower degraded approximately 7.2 times more steeply than the
image tower. Drift is associated with damage but does not determine it: a boundary condition
on the slow-drift justification for memory queues, reported without a claim to the complete
causal mechanism.

With the recipe repaired, learned-query aggregation over frozen token sequences produced the
strongest strictly frozen model at 54.487% validation mean bidirectional R@1 with 2.896M
trainable inference parameters, and a preregistered bounded low-rank relaxation of the final
encoder blocks reached 63.416% with 4.862M, inside the latency budget. On the sealed one-shot
Flickr30k test these reached 52.90 ± 0.87% and 62.20 ± 0.45% across three seeds, a measured
9.30-point cost of strict freezing, retaining approximately 91.2% of the field anchor but only
79.5% of the compact reference. Zero-shot classification did not transfer: on Oxford-IIIT Pets
the aligned models reach 8–10% against MobileCLIP2-S0's 89.2%. This work therefore does not
beat CLIP or reach compact-reference parity. It quantifies how far frozen alignment reaches
under a stated budget, and surfaces a training failure that is not specific to it.

## 1. Introduction

### 1.1 Motivation

Contrastive vision-language models map images and text into a shared embedding
space, and from that single construction they inherit retrieval, zero-shot
classification and a great deal else. The construction is also expensive. Model
quality is coupled to large paired datasets, long training schedules and two
towers optimised together, which places the method out of reach of most
research settings and many deployment ones.

There is an obvious alternative. Strong vision encoders and strong sentence
encoders already exist, trained separately and at someone else's expense; if
the representations they have learned are compatible enough, aligning them
should require only a small trainable module rather than a second round of
web-scale pretraining. The expensive part has already happened. What remains is
a bridge.

Whether that bridge can be built cheaply is not obvious, and this dissertation
takes the question seriously enough to let it fail. Encoders trained
independently need not expose compatible geometry at all. A training recipe
that is dependable when both towers learn can behave quite differently when
most parameters are held fixed, because the assumptions that justified it no
longer hold. Efficiency claimed from parameter counts need not survive
end-to-end measurement on real hardware. And model selection quietly breaks
when baselines are imported from other checkpoints, other splits or other
profiling sessions, which is common enough in this literature to be worth
guarding against explicitly. Comparability and measurement are therefore
treated here as part of the research problem rather than as reporting
overhead.

### 1.2 Research question

> Can modern frozen self-supervised vision encoders, combined with
> compatibility-based pair selection and sub-five-million-parameter adaptation,
> approach compact jointly pretrained vision-language models under a fixed
> inference budget?

The question is evaluated through a hierarchy rather than by relabelling one
favourable baseline. MobileCLIP2-S0 is the primary compact jointly pretrained
reference and also the distillation teacher. OpenCLIP ViT-B/32 is a widely used
field anchor with a split-matched local validation score and same-allocation
latency control; it is not described as compact. SigLIP2 ViT-B/32 is an
alternative-objective field reference. “Approach” means materially reducing a
split-matched local accuracy gap while keeping locally trainable inference
parameters below five million and satisfying a paired latency comparison. The
dissertation reports a strictly frozen-backbone endpoint and a bounded LoRA
adaptation endpoint. Both endpoints have since been evaluated once on the sealed
Flickr30k test, so compact-reference retention is reported on matched splits
rather than withheld as a cross-split comparison.

### 1.3 Contributions

The contributions are empirical and methodological rather than architectural.

1. A controlled reconstruction of the frozen-alignment problem that replaced
   mismatched historical baselines with local checkpoint- and protocol-matched
   references.
2. A diagnosis showing that a cross-batch memory queue, rather than the loss
   family, caused the dominant early performance collapse.
3. A batch, age, and modality factorial showing immediate stale-negative harm
   and demonstrating that drift magnitude alone does not determine the damage.
4. A constrained alignment frontier showing that learned selection over frozen
   patch/token sequences contributes substantially more than access to those
   sequences through mean pooling alone.
5. A bounded LoRA upper bound and a declared same-allocation timing amendment
   that separate model improvement from measurement-session variation.
6. A reproducible negative-result record: failed compatibility screening,
   harmful memory, weak web-data transfer under the tested budget, non-monotonic
   teacher quality, unsuccessful LR retuning, and hard-negative mining stopped
   by its frozen guard.
7. A closed speed--accuracy test showing that parameter-free internal token
   merging can improve measured latency while substantially damaging retrieval,
   so reduced token count is not by itself an efficiency-frontier improvement.

### 1.4 Thesis claim

The evidence supports the following bounded answer. Modern frozen unimodal
encoders can recover a substantial portion of the split-matched OpenCLIP field
anchor's retrieval performance with fewer than five million locally trainable
inference parameters and comparable measured latency, but compact-reference
parity is not demonstrated. Learned token aggregation is the most effective
fully frozen-backbone intervention. Limited encoder adaptation closes more of
the field-anchor gap, while training recipe and measurement validity matter
enough to reverse model-selection conclusions.

## 2. Related work

### 2.1 Locked and frozen vision-language alignment

[LiT](https://arxiv.org/abs/2111.07991) established that a pretrained image
encoder can be locked while contrastive language-side training teaches a text
model to read out its representation. More directly, Freeze-Align studies
[alignment between independently pretrained frozen unimodal encoders](https://openaccess.thecvf.com/content/CVPR2025/html/Maniparambil_Harnessing_Frozen_Unimodal_Encoders_for_Flexible_Multimodal_Alignment_CVPR_2025_paper.html),
and ShareLock similarly investigates
[lightweight alignment around frozen pretrained models](https://openreview.net/forum?id=wqBHJNqeQJ).
[SAIL](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.html)
uses frozen towers, trainable alignment layers, a refined sigmoid objective and
large pre-encoded batches. [STRUCTURE](https://papers.nips.cc/paper_files/paper/2025/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html)
adds geometry preservation for limited paired data. These are direct scientific
peers, not background examples.
[SOTAlign](https://arxiv.org/abs/2602.23353) extends the same broad regime to
semi-supervised alignment with paired and unpaired data; it is a forward-looking
peer rather than a protocol-matched result for this study.
These works mean that “frozen encoders plus projectors” is established prior art.
The present work does not claim that construction as novel. Its contribution is
the controlled constraint regime and the mechanism and measurement findings
that arise within it. Their published Flickr30k figures are tabulated in Section
4.11 with an explicit protocol-mismatch warning.

### 2.2 Historical embeddings and representation drift

[Cross-Batch Memory](https://arxiv.org/abs/1912.06798) increases the available
negative set by storing embeddings from preceding mini-batches. Its motivating
observation is slow feature drift: sufficiently recent embeddings approximate
current ones closely enough to remain useful. The method was demonstrated in
supervised unimodal metric learning. [Adaptive Cross Batch
Normalization](https://arxiv.org/abs/2303.17127) later treated the mismatch
between accumulated and current embeddings as a distribution-alignment problem
and showed that stale memory can sometimes underperform no memory.
[Memory Enhanced Embedding Learning](https://arxiv.org/abs/2103.15686) uses a
momentum encoder for cross-modal video--text retrieval precisely to prevent
rapidly changing historical embeddings. These precedents make generic claims
that “staleness hurts” or “memory can be worse than none” non-novel.

The present queue study occupies a narrower regime: bidirectional image-text
contrast with independently frozen towers and trainable projectors. It does not
claim that historical embedding memory is generally harmful. It asks whether
slow drift is an adequate safety argument here, and whether image-side and
text-side queues respond equivalently.

### 2.3 Modality gap

[Mind the Gap](https://arxiv.org/abs/2203.02053) shows that image and text
representations can occupy separated regions of a shared contrastive space,
with initialisation and contrastive optimisation both contributing. This
creates an important rival explanation: perhaps equal cosine drift has unequal
effects simply because the two modality regions have different geometry.
[Yaras et al.](https://proceedings.mlr.press/v280/yaras25a.html) later identify
mismatched pairs and learnable temperature as contributors and show that closing
the gap can improve retrieval.

The dissertation therefore avoids claiming a new causal mechanism. Its
descriptive result is a difference in sensitivity, not merely a fixed offset:
the text tower has a lower fitted intercept but a much steeper degradation
slope. Static modality separation could be part of why the sensitivities differ,
but drift magnitude by itself is still insufficient to predict queue safety.

### 2.4 Learned aggregation over frozen tokens

Pooling by learned queries is established. The [Set
Transformer](https://arxiv.org/abs/1810.00825) introduced attention-based
permutation-invariant set processing and pooling by learned seed vectors.
[Attentive probing](https://arxiv.org/abs/2506.10178) studies efficient
cross-attention over patch-level features from frozen image encoders and
explicitly frames the problem as an accuracy–parameter trade-off. Related
vision-language systems also use attention poolers:
[CoCa](https://openreview.net/forum?id=Ee277P3AYC) uses attentional poolers,
[SigLIP2](https://arxiv.org/abs/2502.14786) uses MAP pooling, and
[BLIP-2](https://proceedings.mlr.press/v202/li23q.html) uses a Q-Former to read
frozen features.

Consequently, neither the learned-query text pooler nor the vision-token
transformer is presented as a new primitive. The result of interest is
comparative: within one frozen alignment system and one budget, mean pooling
gave a small gain while learned aggregation gave a much larger one. Access to
patch information was insufficient; learning which information to retain was
what paid.

### 2.5 Token reduction and measured efficiency

[Token Merging](https://arxiv.org/abs/2210.09461) combines similar tokens to
increase transformer throughput without requiring retraining. More elaborate
[PuMer](https://aclanthology.org/2023.acl-long.721/) directly prunes and merges
tokens in vision-language systems, reporting throughput and memory gains with a
small accuracy trade-off.
[DynamicViT](https://proceedings.nips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html),
[Token Fusion](https://arxiv.org/abs/2312.01026),
[Dyna-ViT](https://openaccess.thecvf.com/content/CVPR2026F/html/Rubab_Dyna-ViT_Parameter-Free_Pre-Encoder_Token_Pruning_for_Efficient_Vision_Transformers_CVPRF_2026_paper.html),
and [saliency-driven merging](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_Saliency-Driven_Token_Merging_for_Vision_Transformers_CVPR_2026_paper.html)
further show that adaptive token reduction is a crowded field. TokenShift is
deliberately simpler: fixed non-overlapping 2×2 spatial averaging, inserted
after a chosen DINOv3 block, with CLS and register tokens preserved. It is not
content-adaptive ToMe and is not claimed as a novel pruning algorithm. Its role
is to establish whether reducing the internal patch sequence changes measured
full-stack latency in this exact system.

### 2.6 Compact models, distillation, and scaling

[TinyCLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html),
[CLIP-KD](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html),
[MobileCLIP](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html)
and [MobileCLIP2](https://machinelearning.apple.com/research/mobileclip2)
establish that compact VLM quality depends on joint data design, reinforced
training and distillation, not parameter count alone. MobileCLIP2-S0 is therefore
the primary compact reference and, because it supplies this project's teacher
features, an upper bound on transferable teacher information. The project's
teacher comparison and equal-weight mixture are ablations within established
distillation practice, not new distillation algorithms.

[Reproducible scaling laws for contrastive language-image
learning](https://arxiv.org/abs/2212.07143) show power-law behaviour across data,
model size, and compute, while emphasising that training distribution changes
the scaling relationship. This literature is essential for interpreting the
remaining OpenCLIP gap. A small COCO-trained alignment head is not expected to
reproduce all capability acquired through large-scale paired pretraining. The
gap is therefore not clean evidence of an architectural failure, and the
present experiments do not isolate architecture from pretraining scale.
[DataComp](https://arxiv.org/abs/2304.14108) further demonstrates that controlled
data curation can change model quality at fixed training procedures and compute.
[DataComp-VLM](https://arxiv.org/abs/2606.28551) extends the data-centric warning
to broader VLM mixtures and task suites; its scale is contextual, not directly
comparable to this small-data alignment study.

### 2.7 Parameter-efficient adaptation and negative selection

Adapters and low-rank updates are established in vision-language models.
[VL-Adapter](https://openaccess.thecvf.com/content/CVPR2022/html/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.html)
and low-rank CLIP adaptation precede FreezeShift. More directly,
[HALoRA](https://ojs.aaai.org/index.php/AAAI/article/view/40056) allocates a
hierarchical LoRA budget across dual encoders while accounting for
vision--language asymmetry. FreezeShift is therefore a bounded empirical upper
bound and tower ablation, not a new PEFT method.
[B-HFA](https://doi.org/10.1145/3805622.3810607) supplies a recent retrieval-side
adapter and hierarchical-aggregation comparison, but adapts a pretrained VLM
rather than aligning independently pretrained unimodal towers.

Hard-negative selection also carries false-negative risk because image--caption
relevance is many-to-many and benchmark annotations are incomplete.
[PCME](https://openaccess.thecvf.com/content/CVPR2021/html/Chun_Probabilistic_Embeddings_for_Cross-Modal_Retrieval_CVPR_2021_paper.html)
documents non-exhaustive COCO annotations, while
[FALCON](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FALCON_False-Negative_Aware_Learning_of_Contrastive_Negatives_in_Vision-Language_Alignment_CVPR_2026_paper.html)
explicitly balances hard and false negatives. The present guarded mining branch
stopped before training and is not evidence against hard-negative methods.

## 3. Methodology

### 3.1 Research design

The study uses staged decision gates. Each stage either locks a choice for the
next stage, preserves a negative outcome, or stops a branch. This reduces the
degree to which later observations silently change earlier success criteria.
The canonical register assigns stable IDs to all 29 experiment families. All
planned computation is closed: one branch is incomplete-archived, one is
superseded, one is a valid stopped-by-gate result, and the remaining 26 are
complete.

The main progression is:

1. reconstruct locally measured paired and frozen baselines;
2. screen encoder compatibility;
3. diagnose the failed recipe rather than relax its gate;
4. isolate queue capacity, age, batch, and modality;
5. test data coverage and transfer under matched budgets;
6. correct latency and FLOP accounting;
7. test distillation, resolution, teacher, and aggregation choices;
8. freeze the strongest fully frozen-backbone model;
9. run a bounded encoder-adaptation upper bound;
10. repair an invalid cross-session latency comparison in one declared
    same-allocation measurement;
11. profile fixed internal token merging; and
12. run the frozen three-seed accuracy closure and retain the no-merge models
    when every merge arm loses accuracy.

### 3.2 Models

The final fully frozen-backbone model uses a DINOv3 ViT-S/16 vision encoder at
224 pixels and all-MiniLM-L6-v2 as the text encoder. Vision patch tokens are
aggregated by the C4 configuration: width 256 with two transformer blocks. Text
tokens are pooled by a 128-dimensional, four-head learned query. Residual
projections map both towers into a normalised 384-dimensional shared space.

#### 3.2.1 Formulation

Write $f_V$ and $f_T$ for the vision and text encoders. Both are held at their
pretrained values, so for an image $x$ and a caption $y$ the token sequences

\[
H_V = f_V(x) \in \mathbb{R}^{N_V \times d_V},
\qquad
H_T = f_T(y) \in \mathbb{R}^{N_T \times d_T}
\]

are fixed functions of the input: no gradient reaches $f_V$ or $f_T$. Everything
the alignment can learn therefore has to live in what reads those sequences.

That reader is a learned-query attention block. Given $k$ learned queries
$Q \in \mathbb{R}^{k \times d}$ and a frozen token sequence $H$, the aggregator
computes

\[
A(H) = \mathrm{softmax}\!\left(
  \frac{(QW_q)(HW_k)^{\top}}{\sqrt{d_h}}
\right) HW_v ,
\]

with $W_q, W_k, W_v$ trainable. The queries, not the tokens, decide what is read
out. Mean pooling is the special case in which the attention weights are held
uniform and nothing is selected, which is why the comparison between the two in
Section 4.6 isolates selection rather than access.

Each aggregated representation is mapped into the shared space by a residual
projection $\pi$ and $\ell_2$-normalised,

\[
z_V = \frac{\pi_V(A_V(H_V))}{\lVert \pi_V(A_V(H_V)) \rVert_2},
\qquad
z_T = \frac{\pi_T(A_T(H_T))}{\lVert \pi_T(A_T(H_T)) \rVert_2},
\qquad
z_V, z_T \in \mathbb{R}^{384},
\]

so that the similarity of an image and a caption is the inner product
$s_{ij} = z_{V,i}^{\top} z_{T,j}$.

Training minimises a symmetric InfoNCE objective over a batch $B$ with learned
temperature $\tau$. Because COCO images carry several captions, the positive set
$P(i)$ for image $i$ contains every in-batch caption sharing its image
identifier rather than a single diagonal entry:

\[
\mathcal{L}_{V \rightarrow T}
= -\frac{1}{|B|} \sum_{i \in B} \log
\frac{\sum_{p \in P(i)} \exp(s_{ip} / \tau)}
     {\sum_{j \in B} \exp(s_{ij} / \tau)},
\qquad
\mathcal{L} = \tfrac{1}{2}\left(
  \mathcal{L}_{V \rightarrow T} + \mathcal{L}_{T \rightarrow V}
\right).
\]

A cross-batch memory queue changes exactly one thing in this expression: it
extends the denominator with embeddings $\tilde{z}$ that were produced by
*earlier* versions of the projector and stored,

\[
\sum_{j \in B} \exp(s_{ij} / \tau)
\;\longrightarrow\;
\sum_{j \in B} \exp(s_{ij} / \tau)
+ \sum_{m \in M} \exp\!\left(z_{V,i}^{\top} \tilde{z}_{T,m} / \tau\right),
\]

where $M$ is the stored memory. The encoders being frozen does not make
$\tilde{z}$ safe, because $\tilde{z}$ is a function of the projector, and the
projector is still moving. Chapter 4 is largely an account of what that
substitution costs.

FreezeShift relaxes the frozen constraint by the smallest amount that could
still be called an adaptation. For a base weight matrix $W$ it learns a rank-$r$
update

\[
W' = W + \frac{\alpha}{r} BA,
\qquad
B \in \mathbb{R}^{d \times r},
\quad
A \in \mathbb{R}^{r \times k},
\quad
r = 128,
\]

with $W$ fixed. Since $W'$ is formed once and merged before deployment, the
adapted model has the same inference graph as the frozen one, which is what
makes the latency comparison in Section 4.8 a like-for-like measurement.

MobileCLIP2-S0 supplies a training-only distillation signal. It is absent at
inference. Training uses queue-free InfoNCE, all available COCO captions, batch
1024, and a 24-epoch cosine schedule. M_T1 contains 2,896,389 trainable
inference parameters in a 47,196,549-parameter deployed stack; the underlying
encoder parameters remain frozen.

FreezeShift adds rank-128 LoRA to the query/key/value/output attention
projections in the final four DINOv3 blocks, the query/value projections in the
final four MiniLM blocks, or both. The base weights remain fixed and LoRA is
merged before latency profiling. Vision-, text-, and dual-tower totals are
4,076,037, 3,682,821, and 4,862,469 trainable inference parameters.
The corresponding full deployed parameter counts are 48,376,197, 47,982,981,
and 49,162,629. “Trainable” and “total” are reported separately throughout.

### 3.3 Data and split policy

COCO training data supplies paired alignment examples. COCO val2017 is treated
as development evidence because it was involved in earlier model selection.
Flickr30k Karpathy validation supplies the primary external development metric
for the final alignment line. Its test split was held sealed throughout model
development and opened once, for the confirmatory evaluation reported later in
this dissertation.
The 224px distilled configuration and checkpoint epochs were both chosen with
exposure to Flickr validation. The measured difference between COCO-dev-selected
and Flickr-selected epochs is 0.128pp; this quantifies one part of, but does not
remove, the optimistic selection exposure.

Classification development analyses use deterministic partitions of CIFAR-100
train, Oxford-IIIT Pets trainval, and a stored stratified EuroSAT split. Official
classification tests are reserved. Historical evaluations that touched official
tests are retained as exploratory evidence and excluded from confirmatory
claims. Split roles and leakage rules are machine-readable in
`configs/evaluation_protocol.yaml`.

### 3.4 Metrics and selection

The primary accuracy metric is mean bidirectional Recall@1:

\[
R@1_{mean}=\frac{R@1_{image\rightarrow text}+R@1_{text\rightarrow image}}{2}.
\]

Final candidate scores are three-seed means, with standard deviations reported.
FreezeShift promotion required the best parameter- and latency-eligible arm to
improve over M_T1 by more than one pooled standard deviation. This is a practical
heuristic, not a null-hypothesis significance test.

Latency is full-stack Q3 time at batch 64 in native bfloat16 on an NVIDIA RTX
PRO 6000 Blackwell. The corrective experiment used ten independently warmed
repetitions, 100 timed iterations per repetition, and deterministic randomised
order within each repetition. Candidate-minus-OpenCLIP paired differences are
the primary comparison; bootstrap intervals classify faster, equivalent, or
slower behaviour.

### 3.5 Queue factorial

The queue programme separates no-queue, capacity, nominal age, realised
eviction age, batch size, and image-only versus text-only memory. Representation
drift at eviction is measured rather than inferred from the nominal setting.

Drift is instrumented on a fixed probe of 512 held-out development pairs whose
frozen encoder features are cached once, so that any movement observed is
movement of the projector and nothing else. Every ten optimizer steps the probe
is re-projected and normalised, and the step distance

\[
\delta(t) = \frac{1}{|S|} \sum_{s \in S}
\left( 1 - z_s^{(t)\top} z_s^{(t - \Delta)} \right),
\qquad \Delta = 10,
\]

is recorded for each modality. The quantity reported at eviction is not the
displacement between two endpoints but the *path length* accumulated while an
entry actually sat in the queue: for a measured residence of $R$ optimizer
steps,

\[
D(t; R) = \sum_{\text{intervals } \subseteq (t - R,\, t]} \delta(\cdot),
\]

with boundary intervals prorated by overlap. Path length is the conservative
choice, because a representation can wander a long way and return, and an
endpoint measure would score that as no drift at all.

Damage is expressed against the matched queue-free control trained under
otherwise identical settings,

\[
\mathrm{degradation} = 100 \times
\left( \mathrm{R@1}^{\text{no queue}}_{\text{mean}}
     - \mathrm{R@1}^{\text{queue}}_{\text{mean}} \right)
\ \text{pp},
\]

so every degradation figure in Chapter 4 is a within-condition contrast rather
than a comparison against a distant baseline.
An age-zero row identifies the residual batch/in-batch-negative-count change.
Predictions and gates were recorded before outcome analysis; the preregistered
modality-asymmetry prediction was scored as a miss when its nominal-axis form
failed.

### 3.6 Reproducibility and integrity

Configs, compact reports, fingerprints, manifests, and integrity receipts are
retained in the submission. Heavyweight checkpoints, caches, datasets, and raw
Slurm logs remain recoverable in a locally manifested quarantine rather than in
GitHub. Negative and stopped branches remain in the register. The environment
is represented by direct dependency specifications and a verified version lock.
The codebase includes unit, integration, schema, protocol, and artifact tests.

## 4. Results

This chapter reports the experiments in the order the decision gates were
passed, because several later choices only make sense once an earlier
measurement has been corrected. The sequence has a shape. It opens with a
failure that looked like a ceiling on frozen alignment and turned out to be a
fault in the recipe (Sections 4.1 to 4.2); it then pursues the component
responsible far enough to say something general about it (Section 4.3); and it
spends the remainder establishing how much performance the repaired recipe can
actually reach under the stated budget, through data, aggregation, bounded
adaptation and token reduction (Sections 4.4 to 4.11).

Three of those later results are negative, and they are reported at the same
length as the positive ones. A branch that was stopped by its own preregistered
gate, a data source that failed to converge inside its compute ceiling, and a
latency optimisation that worked and was rejected anyway are all part of the
evidence rather than omissions from it.

### 4.1 Foundations and the completed adaptive branch

Early Phase 1 experiments screened five vision and four text encoders across
baseline and bidirectional latent fusion variants, 23 configurations in total.
The strongest grid result was DINOv2 ViT-S/14 with MiniLM under the local BLF
variant, at 17.78% mean COCO R@1 (19.52% image-to-text, 16.05% text-to-image),
against 6.39% for the weakest pair. Matched-seed BLF gains were not
statistically stable: across the three matched baseline-versus-BLF comparisons,
fusion won 7 of 15 metric contests and lost 8.

The more consequential result was that no pairing won everywhere. Scored per
task the leaderboard reshuffles completely, and even the best *text* encoder
changes with the task:

| Task | Best pairing | Score |
|---|---|---:|
| COCO image-to-text R@1 | DINOv2 + MiniLM (local) | 19.52% |
| COCO text-to-image R@1 | DINOv2 + MiniLM (local) | 16.05% |
| CIFAR-100 zero-shot | ConvNeXtV2-T + MiniLM | 29.51% |
| EuroSAT zero-shot | DINOv2 + BGE | 22.29% |
| Oxford-IIIT Pets zero-shot | DINOv2 + BGE | 6.87% |

Phase 1.5 then took the same 23 configurations to sample level, measuring
cross-task rankings, pairwise complementarity over 759 configuration pairs and
37,191 per-class rows, a deliberately non-deployable best-of-all-models oracle
across 6,075 subsets, and per-encoder efficiency profiles. Because no single
pairing dominated, four diverse paths were carried forward rather than one. Two
planned compositional evaluations, Winoground and SugarCrepe, were never run:
the datasets were not installed, and they are recorded as missing rather than
dropped, which is why compositional ability remains untested in Section 6.2.
The Pets figure is also worth noting in hindsight, because the weakness it shows
at 6.87% is the same one the sealed evaluation finds in the finished models six
weeks later.

Phase 2 trained cross-attention rerankers and a task-conditioned router. The
branch completed 1,229 long-format result rows, 40 seed rows, nine statistical
summaries, and four Pareto entries. Cross-attention improved some
classification-development results while reducing COCO retrieval. Smaller
candidate sets performed best, random negatives unexpectedly beat hard
negatives, and the dense router's mean sample utility was 0.01436 below the best
bridge with zero wins across five seeds. This mixed outcome is retained as a
completed secondary branch, not folded into the final dual-encoder headline.

### 4.2 Recipe diagnosis

The initial compatibility screen did not clear its frozen gate. The natural
reading of that failure was the discouraging one: that independently pretrained
encoders simply do not expose compatible enough geometry, and that the ceiling
had been found. The natural response would have been to lower the gate.

Neither was correct. Rather than move the threshold, Wave 0 decomposed the
recipe and tested its components separately, and the result relocated the
problem entirely. Removing the memory queue accounted for roughly 19 percentage
points of improvement, whereas loss-family differences were worth about one.
The ceiling had not been a property of the encoders at all. It had been a
property of the training recipe, and specifically of a component included to
help. Queue-free InfoNCE with all captions, batch 1024, and LR scale 3 was
therefore locked.

Under this corrected recipe, the text-encoder ranking inverted: E5 exceeded BGE,
which exceeded MiniLM. DINOv3 ViT-S/16 remained the preferred vision encoder.
MiniLM was retained for continuity and efficiency; the ranking inversion itself
is a result and constrains claims that encoder compatibility is intrinsic and
recipe-independent.

### 4.3 Cross-batch memory damage

The capacity sweep showed monotonic harm:

| Queue condition | Mean R@1 |
|---|---:|
| No queue | 35.76% |
| Increasing stale-memory capacity | monotonically decreasing |
| Capacity 16,384 | 16.77% |

The queue supplied no favourable accuracy, memory, or throughput trade in this
regime. The factorial located harm at measured age one: relative to matched
queue-free controls, the loss was 6.46pp at batch 1024 and 5.60pp at batch 512.
There was no observed safe non-zero age.

Drift was not a complete explanation. Four matched-drift comparisons remained
within the declared tolerance while their degradation differed by as much as
4.74pp. Separate descriptive regressions produced an image slope of 5.553 and a
text slope of 131.442 percentage points per unit drift. Because modality ranges
differed, the conservative comparison restricted both to their overlapping
drift interval; the resulting slopes were 21.4 and 153.3, a 7.2-fold ratio.

This does not identify a mechanism, and the dissertation does not claim one.
False-negative structure, local geometry, and gradient sensitivity all remain
possible contributors, and separating them would require instrumentation this
study does not have. What the factorial does establish is a boundary that
matters for practice: small cosine movement cannot by itself certify that a
stored cross-modal negative is harmless, because equal movement demonstrably
has modality-dependent consequences. The slow-drift argument is not refuted in
general. It is shown to be insufficient here, which is enough to make any queue
justified by drift alone an unsafe default in this regime.

One false-negative subtype can be excluded from the existing instrumentation.
The loss builds its multi-positive mask after queued entries are appended, so a
queued caption/image sharing the current `image_id` is treated as positive, not
negative. Non-zero queue-positive fractions at epoch boundaries confirm that
this path was exercised. Unannotated semantic matches across different image
IDs remain possible and are the residual false-negative hypothesis.

### 4.4 Data coverage and transfer

At matched update count, full COCO exceeded a 25% subset by 15.03pp on Flickr
validation, showing that unique in-distribution coverage mattered. A qualified
CC3M mirror then underperformed the COCO control by 8.13pp at matched compute
and by 5.89pp after four passes. The extended arm reached its compute ceiling
without convergence, so this is not a universal claim about CC3M.

Mixing COCO back into CC3M training recovered much of the transfer loss:
53.133% Flickr-validation mean R@1 versus 47.209% for CC3M-only. Under this
budget, source and caption distribution mattered more than raw pair count.

### 4.5 Measurement corrections, resolution, and distillation

The first efficiency frontier exposed three conclusion-changing artifacts.
COCO could not be the primary final metric because selection had touched it;
dynamic rather than fixed text padding was required for deployed timing; and
teacher fusion and operator-level FLOP accounting had to match the actual
inference graph.

Distillation had in fact been scoped before this point. A bounded capture probe
had run MobileCLIP2-S0 and SigLIP2-B/32 as separate matched teacher branches
over two seeds, with a preregistered rule that read the absolute student score
rather than the capture fraction, and it returned a limited but measurable
signal. That outcome is why the programme pursued efficiency-normalised
evidence rather than chasing absolute retrieval through distillation alone.

At 224 pixels, DINOv3 closed the latency gap without learned-position
interpolation mismatch. A 24-epoch trajectory quantified the difference between
COCO-dev and Flickr-validation epoch selection. MobileCLIP2 distillation
continued to help after the recipe was corrected, showing that its benefit was
not merely compensation for the queue. Teacher benchmark strength was not
monotonic with student transfer, and an equal two-teacher mixture did not beat
MobileCLIP2 alone. Tuned mixtures were not tested.

### 4.6 Token aggregation

The controlled pooling comparison was decisive:

| Vision-token intervention | Gain over matched baseline |
|---|---:|
| Mean pooling | about +1.33pp |
| Learned-query aggregation | about +6.08pp |
| Transformer aggregation | about +6.91pp |

Thus patch access alone explained little of the gain. A learned selection or
interaction mechanism was required. The 2×2 width/depth study selected C4
(width 256, two blocks). Width and depth both helped but composed
sub-additively, and further capacity was not justified below the five-million
budget.

Text-side learned-query aggregation produced M_T1 at 54.487% mean R@1. Final LR
factor 1.5 lost 0.763pp and failed its promotion rule, retaining factor 1.0.
Guarded hard-negative mining was stopped before training because its frozen
safety rule declared the candidate set infeasible.

### 4.7 Fully frozen-backbone endpoint

M_T1 is the strongest fully frozen-backbone result:

| Property | M_T1 |
|---|---:|
| Flickr30k-validation mean bidirectional R@1 | 54.487% |
| Three-seed SD | 0.328pp |
| Trainable inference parameters | 2,896,389 |
| Same-allocation mean Q3 | 9.148 ms |
| Paired Q3 difference vs OpenCLIP | -0.119 ms |
| 95% bootstrap interval | [-0.139, -0.080] ms |

It is therefore faster than the local OpenCLIP reference under the corrected
same-allocation protocol. Its accuracy remains 15.454pp lower on validation.

### 4.8 FreezeShift upper bound

Every LoRA arm materially improved validation accuracy:

| Arm | Validation R@1 | Gain vs M_T1 | Parameters |
|---|---:|---:|---:|
| Vision | 58.047% | +3.560pp | 4.076M |
| Text | 59.724% | +5.237pp | 3.683M |
| Dual | **63.416%** | **+8.928pp** | **4.862M** |

The first reporter rejected all arms against a 9.480-ms ceiling taken from a
different session. In that allocation the unchanged M_T1 control itself
measured 9.497 ms and failed, demonstrating that the absolute gate was
contaminated by session variation. The negative verdict was preserved and a
measurement amendment was declared before corrective profiling.

In the corrective allocation, OpenCLIP measured 9.267 ms. M_T1, vision, text,
and dual measured 9.148, 9.168, 9.173, and 9.166 ms respectively. Every paired
candidate-minus-reference interval lay below zero. The dual arm therefore
satisfies the unchanged parameter and improvement rules under the repaired
comparison and is the final development adaptation candidate.

Dual retains 90.67% of OpenCLIP validation R@1 and closes 57.77% of the
M_T1-to-OpenCLIP gap. It does not beat OpenCLIP and is not a fully frozen
encoder result.

### 4.9 TokenShift profile

TokenShift reduced 196 patch tokens to 49 after either block 8 or block 6 while
preserving CLS and four register tokens:

| Arm | Mean Q3 | Paired change vs no merge |
|---|---:|---:|
| No merge | 9.122 ms | — |
| Merge after block 8 | 8.138 ms | -0.984 ms |
| Merge after block 6 | 7.515 ms | -1.606 ms |

Both paired intervals were wholly below zero. Because no accuracy evaluation
was part of this profiling stage, the result established only that internal
token count was latency-relevant and motivated the frozen accuracy closure.

### 4.10 TokenShift accuracy closure

The follow-up crossed the two merge depths with the strictly frozen M_T1 and
dual-LoRA parents, using three seeds per arm. Every merge remained faster, but
none preserved its matched parent's retrieval accuracy:

| Parent and merge | Validation R@1 | Change vs parent | Mean Q3 |
|---|---:|---:|---:|
| Frozen, block 8 | 41.907% | -12.581pp | 8.204 ms |
| Frozen, block 6 | 38.537% | -15.950pp | 7.573 ms |
| Dual LoRA, block 8 | 57.725% | -5.690pp | 8.203 ms |
| Dual LoRA, block 6 | 52.554% | -10.861pp | 7.556 ms |

The later merge was consistently less damaging, and LoRA recovered part of the
loss, but even the best trade lost 5.690pp for about 0.934 ms. Because the
package prohibited automatic promotion, the final decision was recorded after
all arms closed: retain frozen M_T1 and dual FreezeShift without token merging.
The result converts the earlier profile into a negative frontier finding—fewer
internal tokens were faster, but not more efficient once accuracy was included.

### 4.11 Reference and peer comparison

Reference results must be read in protocol blocks. The final-candidate rows below
use Flickr30k validation; the compact and SigLIP2 rows retained by the historical
frontier use Flickr30k test. They are shown together to document coverage, not to
create a cross-split ranking.

| Model | Role | Total parameters | Locally trainable parameters | Flickr evidence |
|---|---|---:|---:|---|
| M_T1 | strictly frozen endpoint | 47.197M | 2.896M | 54.487% validation mean R@1 |
| Dual FreezeShift | bounded PEFT endpoint | 49.163M | 4.862M | 63.416% validation mean R@1 |
| MobileCLIP2-S0 | primary compact reference and teacher | 74.835M | not applicable | 78.240% test mean R@1 |
| OpenCLIP ViT-B/32 | field anchor | 151.277M | not applicable | 69.941% validation; 68.220% test |
| SigLIP2 ViT-B/32 | alternative-objective field reference | 376.856M | not applicable | 80.580% test mean R@1 |

Dual FreezeShift is therefore 32.5% of OpenCLIP's total size and reaches 90.67%
of its split-matched validation score. Against MobileCLIP2-S0 it is 65.7% of the
total size, not one third. A performance-retention ratio is not taken from this
table, because doing so would divide validation by test; the retention figures
against both references are reported from the sealed test results instead.

The direct frozen-alignment peer group is also necessary. Freeze-Align reports
87.5/74.1 I2T/T2I R@1 for its large final configuration; the SAIL supplement's
common DINOv2-B/CC12M comparison reports 74.2/60.6 for SAIL-B and 68.1/49.3 for
ShareLock. STRUCTURE evaluates geometry-preserving alignment under limited data
but no directly commensurate final Flickr configuration was identified. These
published values use different encoders, data volumes, objectives and evaluation
choices. They show that the present scores are not state of the art; they do not
invalidate the queue mechanism study or the sub-five-million local-adaptation
frontier.

### 4.12 Where the residual gap lies

Knowing that a gap to OpenCLIP remained said nothing about its character. A
frozen-gallery ranking probe decomposed it, running the three C4 seeds and a
matched OpenCLIP control over the identical Flickr30k validation gallery under
one tie-break rule, with a determinism receipt confirming that repeated runs
reproduced the same ranks. It is a diagnostic: the sealed test was not touched.

The decomposition splits every query into three outcomes—correct at rank 1,
correct but ranked second to tenth, and not retrieved in the top ten at all.
The distinction matters because the second class is a precision failure on
material the model has already found, whereas the third is a failure to find it.

Both models fail in the same way, and the alignment head simply fails more
often. Image-to-text, 29.8% of queries land in ranks 2--10 against OpenCLIP's
18.2%, while only 9.0% fall outside the top ten against OpenCLIP's 1.7%.
Text-to-image, the split is 36.9% against 30.1% and 16.7% against 10.1%.
Roughly three quarters of the image-to-text shortfall is therefore
mis-ordering rather than missing evidence.

This locates the deficit without explaining it. The probe records that Flickr's
five-caption structure inflates image-to-text relative to text-to-image for any
dual encoder, so only the excess over the matched control carries a
model-specific reading, and its ambiguity bins estimate an
ambiguity-associated ceiling rather than an absolute one. What survives those
caveats is a direction for future work: the remaining gap is mostly a ranking
problem among plausible candidates, which is the regime where reranking and
harder negatives act, rather than a representational failure to retrieve.

### 4.13 Sealed final evaluation

The final evaluation was run only after the student roster, selected epochs,
checkpoint paths, prompts and dataset roles had been frozen. Its package exposed
no optimiser or parameter-update path. Flickr30k Karpathy test was the primary
confirmatory retrieval evaluation; CIFAR-100, Oxford-IIIT Pets and EuroSAT were
supplementary zero-shot diagnostics with prior project exposure and are not
presented as pristine confirmatory tests.

| Model | Flickr30k test mean R@1 | Seeds | Full-stack parameters |
|---|---:|---:|---:|
| M_T1, strictly frozen | 52.90 ± 0.87% | 3 | 47.197M |
| Dual FreezeShift | 62.20 ± 0.45% | 3 | 49.163M |
| OpenCLIP ViT-B/32 | 68.22% | fixed reference | 151.277M |
| MobileCLIP2-S0 | 78.25% | fixed reference | 74.835M |
| SigLIP2 ViT-B/32 | 80.46% | fixed reference | 376.856M |

The 9.30pp difference between the two student endpoints is the directly measured
test-set cost of strict freezing in this system. Validation-to-test declines
were modest—1.59pp for M_T1 and 1.22pp for FreezeShift—so the principal model
ordering survived the sealed evaluation. Dual FreezeShift retained 91.18% of
the OpenCLIP field anchor with 32.5% of its total parameters, but only 79.49% of
MobileCLIP2-S0 with 65.7% of its parameters. It therefore approached the field
anchor but remained well short of the compact jointly pretrained reference.

The classification diagnostics reinforced that boundary. The students were
weak relative to all three references, especially on Oxford-IIIT Pets, and LoRA
did not improve every diagnostic. COCO-trained retrieval alignment therefore
did not reproduce broad CLIP-like semantic generalisation.

## 5. Discussion

Two kinds of result came out of the programme, and they deserve different
weight. One is an engineering answer: how far frozen encoders can be pushed
under a five-million-parameter budget, which is useful but bounded to this
setting. The other is a finding about a widely used training component that was
never specific to frozen alignment at all, and which the frozen setting merely
made visible. This chapter argues that the second is the more durable
contribution, sets out what neither result licenses, and treats the measurement
corrections as findings rather than as housekeeping.

### 5.1 Answer to the research question

The answer is split by reference tier. Against the split-matched OpenCLIP field
anchor, “yes, in a bounded sense”: M_T1 preserved 77.90% of the validation score
while meeting the parameter and latency constraints, and bounded dual-tower LoRA
preserved 90.67% with 4.862 million locally trainable inference parameters while
measuring faster in the same allocation. The sealed Flickr30k test has since
been opened, and it confirms the ordering: M_T1 reaches 52.90 ± 0.87% and dual
FreezeShift 62.20 ± 0.45% mean bidirectional R@1 across three seeds, retaining
91.18% of the OpenCLIP field anchor but only 79.49% of MobileCLIP2-S0. Against
the actual compact reference, therefore, an accuracy “approach” claim is still
not demonstrated: neither endpoint establishes parity, and the 9.30-point
difference between them is the measured cost of strict freezing.

The progression also shows that architecture alone is an incomplete account.
Queue removal produced a larger change than early module choices; data source
and coverage mattered; teacher strength did not order student performance; and
an invalid timing comparison initially reversed eligibility. In constrained
systems, the recipe and measurement protocol are part of the model.

### 5.2 Why the queue result matters

XBM's slow-drift argument is plausible in the regime it studied, and nothing
here contradicts it there. The claim under test is narrower and more practical:
that slow movement is sufficient grounds for treating a stored negative as
usable. It is not. The boundary appears once historical negatives cross
independently parameterised modalities whose projectors continue to move. The observation
that harm starts at age one makes “use only a small queue” unsupported here.
The matched-drift dissociation further shows why monitoring one scalar movement
threshold is insufficient.

The strongest objection is the known modality gap. Different cones may make a
unit cosine change more consequential in one modality. But that does not erase
the operational finding; it explains why drift-only safety rules fail. The
descriptive data are also slope-dominated rather than a simple constant offset.
The mechanism remains open, and the dissertation does not claim otherwise.

### 5.3 What token aggregation demonstrates

The token result should be read as evidence about frozen representations, not
as architecture novelty. Global summaries hide useful information. Simply
averaging patches recovers little, while learned interaction recovers much more.
This is consistent with attentive-probing literature: frozen transformer
features distribute task-relevant information across tokens, and a small
learned reader can expose it efficiently.

### 5.4 Adaptation versus freezing

FreezeShift narrows the reference gap substantially. Its tower ablation is also
informative: text-only adaptation gains more than vision-only, and dual
adaptation gains most, so the residual mismatch is not isolated to one tower.
At the same time, allowing LoRA changes the scientific category. The result is
best interpreted as an upper bound on recoverable alignment under the same
parameter budget, while M_T1 remains the answer to the stricter frozen-backbone
question.

This result must also be placed behind HALoRA and related adapter/LoRA work in
the novelty hierarchy. The useful evidence is the controlled vision/text/dual
tower comparison under one local budget, not the invention of low-rank
vision--language adaptation.

### 5.5 Measurement as a research result

The latency amendment is not cosmetic. A gate that rejects its unchanged
control cannot support a model comparison. Preserving the original verdict,
declaring the defect after observation, freezing the corrective protocol before
measurement, and reporting paired uncertainty provides a transparent repair.
It is weaker than a perfectly preregistered first attempt but stronger than
silently replacing the threshold.

### 5.6 Why TokenShift did not move the frontier

TokenShift tests the final inference-side implication of the aggregation work.
Earlier experiments showed that patch detail is valuable only when a learned
reader can select it; TokenShift then removed three quarters of the patch tokens
partway through the frozen vision tower. Its speed gain confirms that those
tokens carry real computational cost, while its accuracy loss confirms that
they still carry information used by the downstream reader. The milder loss
after block 8 suggests later representations are more compressible, and the
smaller loss under dual LoRA suggests adaptation can absorb some compression,
but neither observation supports deployment of the tested fixed merge.

## 6. Limitations and threats to validity

### 6.1 Validation selection and the sealed test

Development headline accuracy values use Flickr30k validation. Multiple model
choices touched that split, and the small M_T1 gain in particular is vulnerable
to selection pressure. The larger FreezeShift improvement is harder to explain away on
magnitude alone, but it remains development evidence on that split.

The confirmatory one-shot result is reported in Section 4.13. It reduces but
does not remove the threat from validation selection: only the two student
endpoints were evaluated across seeds, while each reference remains a single
fixed checkpoint without an accuracy-variance estimate. The sealed test confirms
the model ordering and quantifies the strict-freezing cost, but it does not turn
one dataset into evidence of broad semantic generalisation.

Three seeds quantify local run variability
but provide weak tail uncertainty, and the deterministic reference evaluations
do not estimate accuracy variance. The pooled-SD promotion rule is a decision
heuristic, not a hypothesis test.

### 6.2 Scale and generality

The study uses a bounded set of encoders, COCO-centred training, one qualified
CC3M mirror, and one principal hardware class. Conclusions about queues, data,
and teachers are conditional on this setting. CC3M-only training did not
converge before its compute ceiling, so no universal statement about web data is
available. The final M_T1 and FreezeShift endpoints are established for
short-caption retrieval; Phase 2's mixed classification results do not establish
their compositional, long-text, multilingual or domain-shift generalisation.

### 6.3 Post-observation analyses

The modality-specific drift regressions and latency repair were initiated after
anomalies were observed. They are labelled accordingly. The queue regressions
are descriptive and small-sample; the latency repair is stronger because its
corrective protocol was frozen before a new allocation was measured.

### 6.4 Mechanism identification

The queue factorial separates several conditions but not false-negative
structure, representation curvature, and gradient sensitivity. The dissertation
establishes dissociation, not causation. Direct geometry-normalised drift and
instrumented-gradient studies remain future work. FALCON and PCME make the
false-negative alternative especially important: COCO and Flickr relevance is
many-to-many. Known same-image repetitions were multi-positive masked; unlabelled
cross-image semantic matches were not.

### 6.5 Latency portability

Absolute milliseconds depend on GPU, software, clocking, padding, batch, and
input policy. The defensible quantity is the paired same-allocation comparison,
not universal latency. TokenShift was accuracy-qualified on the declared
validation protocol and rejected, but its exact trade-off remains specific to
the tested fixed 2×2 merge, depths, parents, and hardware.

### 6.6 Literature coverage

The novelty assessment is bounded to the screened corpus. The 10 August 2026
refresh adds recent direct or near-direct work including SOTAlign, HALoRA,
FALCON, Dyna-ViT and SAD-TM. Rapid work on frozen alignment and efficient probing
still means the Tier-1 search should be repeated at formal submission. Absence
from the corpus is not proof that no related result exists.

### 6.7 Efficiency and governance coverage

The repository measures parameters, FLOPs and latency, but it did not uniformly
instrument peak training memory, energy, carbon or financial cost. Retrospective
carbon estimates would introduce unrecorded assumptions and are not supplied.
Likewise, the responsible-reporting section identifies inherited bias and
licensing duties but is not a demographic fairness audit or legal opinion.

## 7. Reproducibility, ethics, and responsible reporting

The GitHub repository retains machine-readable experiment status, frozen
configs, hashes, compact generated reports, and test coverage. Heavyweight
datasets, checkpoints, caches, tensors, and Slurm logs are excluded from public
submission but were moved into a locally recoverable, manifested quarantine.
Failed and stopped studies remain represented by their configs and reports. A
dependency lock records the verified environment specification.

The work uses established vision-language datasets and pretrained models. Its
primary ethical risks arise from inherited dataset and model biases, especially
in open-web captions and zero-shot classification. This dissertation does not
audit demographic fairness and should not imply that retrieval performance
establishes safe deployment. Dataset licences and pretrained-model licences must
be checked for any redistribution or commercial use. The public submission does
not redistribute Flickr30k/COCO images or third-party model weights; it retains
identifiers, manifests, configuration and derived aggregate results. Software
licences do not automatically grant rights in training images, captions, model
weights or downstream uses.

Responsible reporting requires the following distinctions:

- validation versus test;
- frozen backbones versus LoRA-adapted backbones;
- trainable parameters versus total model parameters;
- same-allocation relative latency versus portable absolute latency;
- parameter-free token reduction versus a genuine accuracy--latency frontier
  improvement; and
- an empirical boundary condition versus a proven mechanism.

## 8. Conclusion

This dissertation set out to determine whether independently pretrained vision
and text encoders could approach a compact jointly pretrained VLM under tight
adaptation and inference constraints. The answer is qualified: the work
approaches the split-matched OpenCLIP field anchor under its local budget, but it
does not demonstrate parity with MobileCLIP2-S0, the primary compact reference.
That distinction is a result, not a relabelling opportunity. The answer emerged
through correction as much as invention. Weak early results were not an
immutable compatibility ceiling; a stale memory queue had caused most of the
collapse. Historical
embeddings were harmful immediately, and their damage could not be certified by
drift magnitude alone. In-distribution coverage, recipe, padding, teacher choice,
and measured rather than nominal efficiency all changed conclusions.

Within the fully frozen-backbone regime, learned token aggregation produced
M_T1 at 54.487% validation R@1 with 2.896 million trainable parameters and
faster same-allocation latency than OpenCLIP. Bounded LoRA adaptation raised the
score to 63.416% with 4.862 million trainable parameters, still faster than the
reference under the paired measurement. This closes much of the gap but not all
of it. The outcome is therefore neither parity nor failure: it is a quantified
frontier, with an explicit cost for strict freezing and a measured benefit from
limited adaptation.

The most durable contribution, though, is probably not the frontier. It is the
queue result, and it is not confined to frozen alignment. Cross-batch memory is
used well beyond this setting, and it is routinely justified by the argument
that features move slowly enough for stored embeddings to remain usable. That
argument turns out to be insufficient. Harm here began after a single optimizer
step; conditions matched on measured drift differed in damage by up to 4.74
percentage points; and the text tower degraded roughly 7.2 times more steeply
than the image tower over the same drift range. Drift and damage are associated,
but drift does not determine damage, so a scalar drift threshold cannot certify
that a stored negative is safe. The frozen encoders did not cause this. They
made it measurable, by holding everything else still.

That result was only reachable because the programme was willing to be wrong in
public. Queues, data scale, aggregation, adaptation and timing were each allowed
to produce negative or inconvenient outcomes: a mining branch stopped by its own
gate, a web-scale data source that failed to converge inside its compute
ceiling, a latency optimisation that worked and was rejected because it cost too
much accuracy, and a timing comparison that had to be declared invalid after it
had already produced a verdict. The authorised one-shot final-test evaluation
has since been carried out and closes the confirmatory sequence: on the sealed
Flickr30k test the strictly frozen endpoint reaches 52.90 ± 0.87% and dual
FreezeShift 62.20 ± 0.45% mean bidirectional R@1. Multi-seed reference
evaluation, broader task generalisation and mechanism instrumentation remain
legitimate future work rather than evidence silently assumed complete.

The honest summary is a narrow one. This work does not beat CLIP, and it does
not reach compact-reference parity. What it establishes is how far efficient
alignment of independently pretrained encoders can currently be pushed, what
strict freezing costs when that cost is actually measured rather than assumed,
and that a training component in common use can fail immediately, in
modality-dependent ways, for reasons its standard justification does not
predict.

## References

The submission bibliography is available in [`references.bib`](references.bib).
Core primary sources used in this manuscript are:

- Zhai et al., [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991), 2022.
- Maniparambil et al., [Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](https://openaccess.thecvf.com/content/CVPR2025/html/Maniparambil_Harnessing_Frozen_Unimodal_Encoders_for_Flexible_Multimodal_Alignment_CVPR_2025_paper.html), 2025.
- Zhang et al., [Assessing and Learning Alignment of Unimodal Vision and Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.html), 2025.
- Gröger et al., [With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You](https://papers.nips.cc/paper_files/paper/2025/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html), 2025.
- Ruthardt et al., [Better Language Models Exhibit Higher Visual Alignment (ShareLock)](https://openreview.net/forum?id=wqBHJNqeQJ), 2026.
- Wang et al., [Cross-Batch Memory for Embedding Learning](https://arxiv.org/abs/1912.06798), 2020.
- Ajanthan et al., [Adaptive Cross Batch Normalization for Metric Learning](https://arxiv.org/abs/2303.17127), 2023.
- Liang et al., [Mind the Gap](https://arxiv.org/abs/2203.02053), 2022.
- Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825), 2019.
- Psomas et al., [Attention, Please! Revisiting Attentive Probing](https://arxiv.org/abs/2506.10178), 2025.
- Cao et al., [PuMer: Pruning and Merging Tokens for Efficient Vision Language Models](https://aclanthology.org/2023.acl-long.721/), 2023.
- Bolya et al., [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461), 2022.
- Wu et al., [TinyCLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html), 2023.
- Vasu et al., [MobileCLIP](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html), 2024.
- Faghri et al., [MobileCLIP2](https://machinelearning.apple.com/research/mobileclip2), 2025.
- Kim et al., [FALCON](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FALCON_False-Negative_Aware_Learning_of_Contrastive_Negatives_in_Vision-Language_Alignment_CVPR_2026_paper.html), 2026.
- Cherti et al., [Reproducible Scaling Laws for Contrastive Language-Image Learning](https://arxiv.org/abs/2212.07143), 2023.
- Gadre et al., [DataComp](https://arxiv.org/abs/2304.14108), 2023.

## Appendix A. Experiment inventory

The authoritative inventory is [`experiments/README.md`](../experiments/README.md).
The repository records 29 experiment families spanning foundations,
adaptive reranking, recipe diagnosis, queue mechanism, data and transfer,
efficiency and token architecture, diagnostics, bounded adaptation, and
measurement repair. Phase 2, FreezeShift, TokenShift profiling, the latency
amendment, and the TokenShift accuracy closure are complete. Alignment v3
remains incomplete-archived, the recovery branch is superseded, and guarded
hard-negative mining is a valid stopped-by-gate result.

## Appendix B. Final evidence paths

- Fully frozen M_T1: `results/text_aggregation_study/training/M_T1/report/report.json`
- Final LR decision: `results/final_lr_study/report/report.json`
- FreezeShift: `freezeshift/results/report/report.json`
- Corrective latency: `latency_amendment/results/report/report.json`
- Final model decision: `docs/final_model_freeze.md`
- TokenShift: `tokenshift/results/profile/report.json`
- TokenShift accuracy: `tokenshift_training/results/report/report.json`
- Queue factorial: `results/queue_factorial/report/report.json`
- Drift analysis: `results/queue_factorial/interpretation/drift_regression.json`
- Known-positive audit: `results/queue_factorial/interpretation/known_positive_audit.md`
- Phase 2: `results/phase2/phase2_report.json`
- Experiment registry: `experiments/registry.yaml`
