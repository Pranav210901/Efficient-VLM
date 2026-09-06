# 39 — Data scaling and retrieval diagnostics

Grouped source covering `cc3m_acquisition`, `mixed_data_training`,
`data_scale_pilot` and `probe1_error_decomposition`.

- **Category:** supporting context
- **Verification level:** `search_summary`
- **Credibility:** 4–5 · **Recency:** mixed · **Bias:** 2

## Data scaling

### 39a — Reproducible scaling laws for contrastive language-image learning (arXiv 2212.07143, CVPR 2023)

The canonical citation for CLIP-family scaling behaviour and the reference point
for any data-scale claim the project makes.

> "Performance consistently improves when increasing scale following power law
> trends, with scaling coefficients measured for zero-shot retrieval on MS-COCO
> and Flickr30K."

> "Open image-text datasets like MS-COCO, Visual Genome, YFCC-100M, Conceptual
> Captions CC3M and CC12M not matching the current scale of private data."

> "Scale bottleneck effects have been observed, where OpenCLIP ViT-L/14 shows
> almost no improvement on LAION-400M when increasing the number of samples seen
> from 13B to 34B."

**Relevance:** the project's 26–38pp gap to OpenCLIP/MobileCLIP2/SigLIP2 is
substantially a *data-scale* gap, and this paper quantifies that. It is the
strongest available defence against a reviewer who reads the gap as an
architectural failure.

### 39b — An Inverse Scaling Law for CLIP Training (arXiv 2305.07017)

Counterweight: larger encoders tolerate shorter token sequences. Relevant to the
resolution / token-count arms.

### 39c — Dataset Growth (arXiv 2405.18347), VeCLIP (2310.07699), Filter & Align (2312.06726), Differential-informed Sample Selection (2507.12998)

Data curation and sample-efficiency literature. Relevant to the CC3M mixing
decision — these argue curation beats raw volume:

> "Curation can substantially outperform baselines on retrieval evaluations and
> reach raw baseline performance in 2.3% of the time."

**Implication for the project:** a mixed COCO+CC3M arm that adds raw CC3M
without curation is testing the weaker of the two known levers. Worth
pre-registering an explicit statement about which lever is being tested.

## Retrieval diagnostics

### 39d — Benchmark Granularity and Model Robustness for Image-Text Retrieval (arXiv 2407.15239)

Closest published analogue to `probe1_error_decomposition`.

> "Studies compare common image-text retrieval benchmarks like MS-COCO and
> Flickr30k with their fine-grained augmented versions (MS-COCO-FG and
> Flickr30k-FG) using specified sets of linguistic features capturing concept
> granularity. Fine-grained variants with richer captions consistently enhance
> retrieval performance, especially in text-to-image tasks with average
> improvements of 16.23%."

> "Researchers have introduced taxonomies of perturbations and evaluation
> frameworks to investigate the performance of vision-language models on coarse
> and fine-grained datasets."

**Relevance:** the project's `FAILURE_CATEGORIES` taxonomy
(fine_grained_attribute, counting_quantity, spatial_relation, action_interaction,
…) overlaps this work's taxonomy. Cite it, and check whether the categories can
be aligned to an existing published taxonomy rather than defined ad hoc — a
reviewer will ask why these nine categories.

### 39e — Assessing Brittleness of Image-Text Retrieval Benchmarks (arXiv 2407.15239 companion / related)

Same programme; supports the annotation-ambiguity category in the project's
rubric.

### 39f — A Comprehensive Survey on Composed Image Retrieval (arXiv 2502.18495)

Survey — use only for framing, low direct relevance.
