# Literature-gap closure matrix

**Cut-off:** 10 August 2026  
**Question:** How does this dissertation compare with frozen-alignment and
efficient vision-language research, and which remaining differences are claims,
protocol limitations, or genuinely missing evidence?  
**Search record:** 20 decomposed academic-index searches returned 322 logged
records; 15 primary records were cited in the closure guide. The targeted search
export contained 273 unique records after global deduplication. These were
triaged against the existing 54-entry project bibliography and its 288-work
citation screen, then high-risk claims were checked against primary paper pages.
The free academic indexes were noisy, so a search hit was not treated as evidence
until its primary abstract, paper, or supplement was inspected.

This matrix closes a *writing gap* when the relevant literature and claim boundary
are now explicit. It does not call an empirical gap closed merely because a paper
has been cited. No unrun baseline, energy figure, or licence permission is
inferred. The final-test material below was added after the original cut-off and
is tied to the repository's source-of-record result tables.

## Bottom line

The dissertation is complete as a bounded development-evidence study, but it is
not a state-of-the-art claim and it does not demonstrate parity with the compact
reference named by the research question. Its strongest original result is the
queue decomposition: equal measured representation drift can produce materially
different damage, and the two towers have sharply different drift sensitivity.
Its strongest model result is dual FreezeShift, which reaches 63.416% Flickr30k
validation mean bidirectional R@1 with 4.862M locally trainable inference
parameters, and **62.20 ± 0.45% on the sealed Flickr30k test** (see the
sealed-test section below, added after this document's original cut-off). It is 32.5% of the *total* size of the OpenCLIP ViT-B/32 field anchor
and retains 90.67% of that anchor's locally measured validation R@1. It is not one
third the size of MobileCLIP2-S0: it is 65.7% of its total size, and the repository
contains a split-matched final-test comparison to MobileCLIP2-S0: 62.20% versus
78.25%, or 79.49% retention. This is evidence of a remaining gap, not parity.

### Submission-date refresh: 20 August 2026

A standard-depth refresh used 11 free-lane academic-index searches including
reconnaissance, logging 182 unique-paper receipts and retaining five primary
papers for claim checking. One over-constrained multi-name query returned no
results; decomposed searches recovered the relevant records. Freeze-Align,
SAIL, STRUCTURE, and SOTAlign remain the closest methodological comparisons.
Hyperdimensional cross-modal alignment of frozen image and language models is a
new adjacent thread, but its principal task is efficient image captioning rather
than Flickr-style dual-encoder retrieval, so it is not inserted into the numeric
peer table. No refreshed source displaces the dissertation's novelty hierarchy:
frozen alignment and lightweight connectors are prior art; the strongest claim
remains the measured modality-asymmetric staleness result and the dissociation
between embedding drift and retrieval damage.

## The 25 gaps, ranked and resolved

| Rank | Gap | Resolution in this submission | Closure class | Residual boundary |
|---:|---|---|---|---|
| 1 | Frozen-alignment peers | Freeze-Align, SAIL, ShareLock and STRUCTURE are now treated as direct peers and tabulated below. | **Closed in writing** | Published results use different encoders, data, objectives and splits. |
| 2 | MobileCLIP2 as primary compact reference | MobileCLIP2-S0 is the Tier-1 compact jointly pretrained reference and also the distillation teacher. | **Closed in framing** | The final split-matched test comparison shows 79.49% retention, not parity. |
| 3 | OpenCLIP and SigLIP2 roles | OpenCLIP B/32 is a field anchor and same-allocation latency control; SigLIP2 B/32 is a different-objective field reference. Neither is called compact. | **Closed** | Final-test comparisons are split-matched; only OpenCLIP also has a split-matched local development comparison and same-allocation latency measurement. |
| 4 | Protocol comparability | Tables separate local validation, historical local test, and published external results. | **Closed** | Cross-table ranking is prohibited. |
| 5 | Queue novelty boundary | XBM, ACBN, MoCo, MEEL, CODER and CSMCIR establish memory and staleness prior art. Novelty is restricted to drift--damage dissociation and modality-specific sensitivity. | **Closed** | Absence is bounded to the screened corpus; causation is not established. |
| 6 | Parameter accounting | Total inference, locally trainable inference, and query-side counts are separated. | **Closed** | Published peers do not all expose equivalent counts. |
| 7 | Validation-selection exposure | Configuration and checkpoint selection on Flickr validation, the 0.128pp epoch-selection gain, and the frozen-roster one-shot test are stated. | **Closed** | The test was run once after roster freeze; further test-led selection is prohibited. |
| 8 | Learned aggregation prior art | Set Transformer PMA, CoCa poolers, SigLIP2 MAP, BLIP-2 Q-Former and attentive probing establish the primitive. The thesis claims an internal ablation, not architecture novelty. | **Closed** | The exact near-no-op gate was not separately ablated. |
| 9 | Parameter-efficient adaptation | VL-Adapter, low-rank CLIP adaptation, B-HFA and 2026 HALoRA now bound FreezeShift. | **Closed in writing** | HALoRA is a direct conceptual comparator but not protocol matched or retrained here. |
| 10 | Compact/mobile VLM landscape | TinyCLIP, MobileCLIP and MobileCLIP2 contextualise compact joint training and distillation. | **Closed** | The project does not claim a mobile deployment benchmark. |
| 11 | Distillation and teacher selection | TinyCLIP, CLIP-KD and MobileCLIP2 establish distillation; this project contributes only a bounded teacher/mixture result. MobileCLIP2 is explicitly both teacher and upper bound. | **Closed** | Tuned mixtures and teacher ensembles were not searched experimentally. |
| 12 | Token pruning and merging | DynamicViT, ToMe, PuMer, Token Fusion, Dyna-ViT and SAD-TM contextualise TokenShift. PuMer is the closest multimodal efficiency comparison. | **Closed** | TokenShift tests fixed 2x2 averaging only, not adaptive methods. |
| 13 | Latency methodology | Hardware, precision, batch, warm-up, timed repetitions, randomised order, padding and paired bootstrap intervals are stated. | **Closed** | Absolute latency is not portable beyond the measured software/hardware stack. |
| 14 | Data scale and distribution | DataComp, CLIP scaling and compute-aware curation make clear that the remaining reference gap is entangled with paired-data scale and distribution. | **Closed** | The non-converged CC3M arm cannot support a universal dataset claim. |
| 15 | Hard and false negatives | The stopped hard-negative branch is not called a negative-mining result. Existing loss diagnostics show that queued same-image recurrences were multi-positive masked, ruling out that narrow error. PCME and 2026 FALCON motivate residual unlabelled cross-image false negatives. | **Partially closed by reanalysis** | Semantic cross-image relevance or FALCON-style training was not measured. |
| 16 | Modality gap | The discussion incorporates established modality-gap explanations and uses the conservative overlap-restricted slope comparison. | **Closed** | Geometry-normalised drift remains future work. |
| 17 | Encoder compatibility | Freeze-Align's CKA result is compared with the project's recipe-dependent rank inversion. Compatibility is treated as predictive but not intrinsic. | **Closed** | The encoder grid is finite and dataset dependent. |
| 18 | Loss and batch effects | InfoNCE, SigLIP and SAIL establish loss/batch context; Wave 0 is reported as a controlled negative result in which queue removal dominated loss choice. | **Closed** | The project does not generalise beyond the tested batches/objectives. |
| 19 | Resolution and positional effects | The 224px result is tied to DINOv3's positional mechanism and the exact measured preprocessing policy. | **Closed** | Other resolutions and encoders were not exhaustively tested. |
| 20 | Retrieval statistics | Three-seed means/SDs, paired latency intervals and direction-specific R@1 are reported. The selection heuristic is not described as a significance test. | **Substantially closed** | Three seeds give weak tail uncertainty; deterministic references lack run-to-run accuracy variance. |
| 21 | Task generalisation | Phase 2 classification/retrieval results are retained as a secondary branch; final M_T1/FreezeShift claims are retrieval specific. | **Closed as a limitation** | Evaluating final endpoints on compositionality, classification and domain shift would be new evidence. |
| 22 | Compute, memory and energy | Parameter counts, FLOPs, latency, hardware and training schedules are available. Energy/carbon was not instrumented and is not estimated retrospectively. | **Partially closed** | Energy, peak training memory and end-to-end training cost are not uniformly available. |
| 23 | Reproducible negative results | Failed gates, stopped mining, queue harm, CC3M limits, teacher mixtures, LR retuning and TokenShift accuracy loss remain registered with evidence paths. | **Closed** | Heavy raw assets are quarantined locally rather than distributed on GitHub. |
| 24 | Bias, data governance and licensing | The manuscript now distinguishes performance from safe deployment, notes inherited bias, and records that datasets/checkpoints are not redistributed. | **Partially closed** | This is not a demographic fairness audit or legal licence opinion. |
| 25 | 2025--2026 recency | The refresh adds SOTAlign, HALoRA, B-HFA, FALCON, Dyna-ViT, SAD-TM and DataComp-VLM. | **Closed to cut-off** | Literature recency decays; repeat the Tier-1 search at final submission. |

## Comparison hierarchy

### Tier 1: compact jointly pretrained reference

MobileCLIP2-S0 is the reference that matches the wording of the research
question. It is also the distillation teacher, so it represents both an external
compact model and an upper bound on what the chosen teacher could transfer. The
project's final endpoints are smaller in total parameters, but not by threefold:
M_T1 is 63.1% and dual FreezeShift 65.7% of MobileCLIP2-S0's size.

### Tier 2: field anchor and matched development control

OpenCLIP ViT-B/32 is retained because it is a widely recognisable CLIP anchor and
because it has a split-matched local development score and a same-allocation
latency measurement. It is not a compact-model peer. The defensible headline is:

> Dual FreezeShift uses 49.163M total inference parameters, 32.5% of the local
> OpenCLIP ViT-B/32 stack, and reaches 90.67% of its Flickr30k-validation mean
> bidirectional R@1 while measuring 0.101ms faster in the paired allocation.

This is a development comparison, not a final-test or state-of-the-art claim.

### Tier 3: alternative jointly pretrained objective

SigLIP2 ViT-B/32 supplies a strong sigmoid-objective reference. Its role is to
show that the field frontier is not specific to CLIP's softmax objective, not to
act as the compact target.

### Tier 4: frozen-alignment peer group

Published Flickr30k figures below are included to answer the direct examiner
question, “how does this compare with other frozen-alignment work?” They are not
pooled with the project's validation scores.

| Work/configuration | Frozen towers | Alignment data | Flickr split reported | I2T R@1 | T2I R@1 | Mean | Comparability note |
|---|---|---:|---|---:|---:|---:|---|
| Freeze-Align, DINOv2 + All-Roberta-Large | yes | combined set, up to 20M pairs | paper reports validation during ablation and test for final evaluation | 87.5 | 74.1 | 80.8 | Much larger encoders/data; published protocol differs. |
| SAIL-B (NV2), CC12M raw | yes | 12M | standard retrieval table | 74.2 | 60.6 | 67.4 | Refined sigmoid loss, multi-positive captions, batch 32,768. |
| ShareLock (NV2), re-evaluated in SAIL supplement | yes | 12M | standard retrieval table | 68.1 | 49.3 | 58.7 | Values come from SAIL's common-backbone comparison, not this repository. |
| STRUCTURE | yes | limited paired data, task dependent | no directly commensurate final configuration identified | -- | -- | -- | Geometry-preserving objective; compare method and data regime, not a fabricated scalar. |
| M_T1 | yes | COCO | **validation** | direction-specific final aggregate not re-tabulated here | -- | **54.487** | Three-seed development result; the later test result is reported separately below. |
| Dual FreezeShift | base weights fixed; LoRA adaptation | COCO | **validation** | direction-specific final aggregate not re-tabulated here | -- | **63.416** | Not a strictly frozen-encoder result. |

Primary records: [Freeze-Align](https://openaccess.thecvf.com/content/CVPR2025/html/Maniparambil_Harnessing_Frozen_Unimodal_Encoders_for_Flexible_Multimodal_Alignment_CVPR_2025_paper.html),
[SAIL](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.html),
[ShareLock](https://openreview.net/forum?id=wqBHJNqeQJ), and
[STRUCTURE](https://papers.nips.cc/paper_files/paper/2025/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html).

## Local model/reference accounting

These counts use full deployed stacks. “Locally trainable” means parameters
optimised by this project and retained at inference; it must not be confused with
total size.

| Model | Scientific role | Total inference params | Locally trainable inference params | Accuracy evidence | Latency evidence |
|---|---|---:|---:|---|---|
| M_T1 | strictly frozen endpoint | 47.197M | 2.896M | 54.487% Flickr validation | 9.148ms Q3, corrected allocation |
| Dual FreezeShift | bounded PEFT upper bound | 49.163M | 4.862M | 63.416% Flickr validation | 9.166ms Q3, corrected allocation |
| MobileCLIP2-S0 | primary compact reference and teacher | 74.835M | not applicable | 78.240% Flickr test in historical frontier | 21.280ms Q3 in historical frontier allocation |
| OpenCLIP ViT-B/32 | field anchor and latency control | 151.277M | not applicable | 69.941% Flickr validation; 68.220% test in historical frontier | 9.267ms Q3 in corrected allocation |
| SigLIP2 ViT-B/32 | alternative-objective field reference | 376.856M | not applicable | 80.580% Flickr test in historical frontier | 11.638ms Q3 in historical frontier allocation |

The historical-frontier latency rows and corrected-allocation latency rows are
different experiments and must not be ranked as if paired. Likewise, validation
and test scores must not be divided to create a performance-retention claim.

## Recent papers that change the discussion

- [HALoRA (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40056)
  makes modality-aware, budgeted low-rank adaptation a direct conceptual peer for
  FreezeShift. FreezeShift remains useful as a small factorial, not a new PEFT
  algorithm.
- [PuMer (ACL 2023)](https://aclanthology.org/2023.acl-long.721/) is a closer
  multimodal token-reduction comparator than ToMe alone; it reports throughput
  and memory benefits with small accuracy loss, whereas fixed TokenShift lost
  5.69--15.95pp.
- [FALCON (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FALCON_False-Negative_Aware_Learning_of_Contrastive_Negatives_in_Vision-Language_Alignment_CVPR_2026_paper.html)
  confirms that hard-negative utility and false-negative risk must be separated.
  This project did not perform that separation.
- [SOTAlign (2026)](https://arxiv.org/abs/2602.23353) extends frozen alignment with
  paired and unpaired data and therefore belongs in future peer updates.
- [Explaining and Mitigating the Modality Gap (2025)](https://proceedings.mlr.press/v280/yaras25a.html)
  supplies a gradient-flow account of the modality gap but does not report the
  queue-age, per-modality drift--damage dissociation measured here.
- [DataComp-VLM (2026)](https://arxiv.org/abs/2606.28551) reinforces that data
  composition and scaling are first-order variables; it does not make this
  project's small-data result directly comparable to web-scale joint training.

## Sealed-test results (added after the 10 August cut-off)

The sealed Flickr30k test has since been run. Source of record:
`artifacts/07_final_evaluation/zero_shot/report/aggregate_results.csv` and
`per_seed_results.csv`; protocol in `configs/final_zero_shot/pipeline.yaml`.
Students are three seeds (42/43/44); references are single deterministic runs.

| Model | Flickr30k **test** mean R@1 | SD | Validation (for contrast) |
|---|---:|---:|---:|
| M_T1 strictly frozen | **52.90%** | 0.87 | 54.487% |
| Dual FreezeShift (dual LoRA) | **62.20%** | 0.45 | 63.416% |
| OpenCLIP ViT-B/32 (field anchor) | 68.22% | — | — |
| MobileCLIP2-S0 (compact reference) | 78.25% | — | — |
| SigLIP2 ViT-B/32 | 80.46% | — | — |

Three consequences. First, the cost of strict freezing is now measured on the
sealed test: **9.30pp** (62.20 − 52.90), against 8.93pp on validation. Second,
test regressed from validation as predicted (−1.59pp frozen, −1.22pp LoRA),
which is evidence the selection protocol behaved as described. Third, retention
is now statable split-matched, and must be reported against both comparators:
**91.18% of the OpenCLIP field anchor** but **79.49% of MobileCLIP2-S0**, the
compact reference named by the research question. The OpenCLIP ratio alone is a
selected comparison. Both reference values are single runs with no SD, so the
ratios inherit uncertainty only from the student's ±0.45.

The headline remains that this is not parity with the compact reference:
MobileCLIP2-S0 (78.25%) and SigLIP2 (80.46%) are both well ahead of dual
FreezeShift (62.20%) on the same split.

## Irreducible items before formal submission

Items 1 and 2 below are now **closed** by the sealed-test run recorded above and
are retained for provenance. Item 3 is **not** closed; see the caution beneath.

1. ~~a one-shot Flickr30k test result~~ — **closed**; artifacts exist.
2. ~~split-matched final-candidate accuracy against MobileCLIP2-S0 and
   SigLIP2~~ — **closed**; both are in the same `flickr30k_test` block above.
3. final-endpoint generalisation beyond short-caption retrieval — **still open.**
   Zero-shot CIFAR-100, Pets and EuroSAT diagnostics were produced in the same
   package, but `configs/final_zero_shot/pipeline.yaml` explicitly discloses that
   "classification benchmarks are external zero-shot diagnostics ... they are not
   described as pristine confirmatory tests." They therefore cannot close this
   item. Pets in particular is 9.90% (frozen) and 7.96% (dual LoRA) against
   OpenCLIP's 83.10% on 37 classes, i.e. roughly 3.7x and 2.9x a 2.70% chance
   baseline: far above chance but far below usable. The three prompt templates
   are defined once globally and so are identical across all five models, which
   rules out template asymmetry as the explanation; references do, by necessity,
   use checkpoint-native tokenisation and preprocessing. The likely reading is a
   genuine limit of COCO caption supervision for fine-grained breed vocabulary,
   but that should be stated as a diagnostic observation, not a closed
   generalisation finding.
4. semantic cross-image false-negative, geometry-normalised and
   gradient-sensitivity mechanism experiments;
5. instrumented training energy/carbon and a formal demographic fairness audit;
6. university-specific front matter, word count, formatting and supervisor sign-off.

These do not prevent submission as a bounded development-evidence dissertation
if labelled as limitations. They do prevent stronger claims of compact-model
parity, final-test confirmation, mechanism identification, deployment readiness,
or state of the art.
