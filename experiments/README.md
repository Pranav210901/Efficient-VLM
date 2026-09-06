# Canonical experiment index

## Dissertation question

> **Can modern frozen SSL vision encoders, combined with compatibility-based
> pair selection and sub-5M-parameter adaptation, approach compact jointly
> pretrained vision-language models under a fixed inference budget?**

This page is the research-facing map of the repository. It gives every
experiment family a stable ID, a headline that says what was tested, and an
outcome that says what was learned. It does **not** rename the physical result
directories. Those paths are part of the experiment provenance and may be read
by resume, reporting, fingerprint, and checkpoint code.

## Status vocabulary

| Status | Meaning |
|---|---|
| `COMPLETE` | The planned computation and report completed. |
| `STOPPED_BY_GATE` | The preregistered gate stopped the branch; this is a valid bounded result. |
| `INCOMPLETE_ARCHIVED` | The original plan did not complete and is retained as historical evidence. |
| `SUPERSEDED` | A later controlled experiment replaced this branch for decision-making. |
| `READY_NOT_RUN` | Code/configuration exists, but no result may be claimed. |
| `IN_PROGRESS` | A frozen study is actively executing; partial outputs are not results. |

## Current headline

The strongest fully frozen-backbone model is **M_T1**:

- frozen DINOv3 ViT-S/16 vision encoder at 224 px;
- two-block, 256-d learned vision-token transformer aggregation (`C4`);
- frozen all-MiniLM-L6-v2 text encoder;
- 128-d, four-head learned-query text aggregation;
- trainable residual projections into a shared normalized 384-d embedding;
- MobileCLIP2-S0 distillation during training only;
- queue-free InfoNCE, all COCO captions, batch 1024;
- 2,896,389 inference-trainable parameters, below the 5M limit;
- 54.487% three-seed Flickr30k-validation mean bidirectional R@1;
- same-allocation mean full-stack Q3 latency 9.148 ms on an RTX PRO 6000
  Blackwell, 0.119 ms faster than the local OpenCLIP reference.

The bounded FreezeShift follow-up found a stronger development-selected model:
dual-tower rank-128 LoRA reached 63.416% validation mean R@1 with 4,862,469
trainable inference parameters. After the declared same-allocation measurement
repair, its mean Q3 latency was 9.166 ms, 0.101 ms faster than OpenCLIP (95%
bootstrap CI for the paired difference: -0.145 to -0.016 ms). This is an
adaptation upper bound, not evidence for a fully frozen encoder claim. The later
no-training sealed evaluation measured 62.20 ± 0.45% test R@1 across three seeds.

The final TokenShift accuracy study closed the remaining profile-only caveat.
All four merge arms were faster, but every arm was materially less accurate
than its matched no-merge parent. The least damaging merge, dual tower after
block 8, still lost 5.690pp; no TokenShift arm was promoted.

## Experiment story

### Track F — Foundations and initial alignment

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `F00` | **Legacy BLF and encoder-matrix foundation** | `COMPLETE` | Early BLF and frozen-pair experiments established the codebase and initial baselines; they are historical context, not the current comparison protocol. | [`results/legacy_summary`](../results/legacy_summary), [`results/phase15`](../results/phase15) |
| `F01` | **Alignment v2: a locally measured strong paired reference** | `COMPLETE` | A checkpoint-matched OpenCLIP reference and three-seed frozen-unimodal baselines replaced weak or mismatched historical comparisons. | [`results/alignment_v2`](../results/alignment_v2), [`docs/alignment_v2_methodology.md`](../docs/alignment_v2_methodology.md) |
| `F02` | **Alignment v3: compatibility screening did not clear its gate** | `INCOMPLETE_ARCHIVED` | The official negative screen was preserved. It motivated controlled recipe diagnosis rather than silent threshold changes. | [`results/alignment_v3`](../results/alignment_v3), [`docs/alignment_v3_methodology.md`](../docs/alignment_v3_methodology.md) |
| `F03` | **Batch-calibration recovery branch** | `SUPERSEDED` | Reference evaluation and recovery manifests were produced, but the branch was overtaken by the more complete Wave-0 recipe and queue diagnosis. | [`results/alignment_v3_recovery`](../results/alignment_v3_recovery), [`docs/alignment_v3_recovery_methodology.md`](../docs/alignment_v3_recovery_methodology.md) |
| `F04` | **Alignment v4: teacher-signal capture was measurable but limited** | `COMPLETE` | Projection-only alignment captured about 23% of the available teacher gap under the then-current queued recipe. This became motivation for testing whether the ceiling was caused by the recipe. | [`results/alignment_v4`](../results/alignment_v4), [`docs/alignment_v4_capture_probe.md`](../docs/alignment_v4_capture_probe.md) |

### Track A — Adaptive reranking and routing

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `A01` | **Phase 2: cross-attention bridges and adaptive routing** | `COMPLETE` | Cross-attention substantially improved some classification-development scores but reduced COCO retrieval. Random negatives beat hard negatives, and the dense task-conditioned router lost to the best single bridge on all five seeds under the reported utility metric. This completed branch is secondary evidence, not the final model line. | [`results/phase2`](../results/phase2), [`docs/supervisor_review.md`](../docs/supervisor_review.md) |

### Track R — Recipe correction and pair selection

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `R01` | **Wave 0: the memory queue, not the loss family, explained the collapse** | `COMPLETE` | Queue-free InfoNCE with all captions, batch 1024, and LR scale 3 was locked. Queue removal accounted for roughly 19pp; loss-family differences were about 1pp. | [`results/alignment_v4_wave0`](../results/alignment_v4_wave0), [`configs/alignment_v4_wave0`](../configs/alignment_v4_wave0) |
| `R02` | **Recipe-corrected pair re-screen: text ranking inverted** | `COMPLETE` | DINOv3 ViT-S/16 remained the better vision encoder, while E5 > BGE > MiniLM under the corrected recipe. MiniLM was retained for continuity; the ranking change was reported as the finding. | [`results/alignment_v4_wave0/wave0-rescreen`](../results/alignment_v4_wave0/wave0-rescreen), [`results/alignment_v4_wave0/selection`](../results/alignment_v4_wave0/selection) |

### Track Q — Queue mechanism

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `Q01` | **Queue dose response: more stale negatives caused monotonic damage** | `COMPLETE` | Mean R@1 fell from 35.76% with no queue to 16.77% at capacity 16,384. The queue had no favourable accuracy, memory, or throughput trade in this regime. | [`results/alignment_v4_wave0/wave0-queue-ablation`](../results/alignment_v4_wave0/wave0-queue-ablation) |
| `Q02` | **Queue factorial: there was no safe stale-negative threshold** | `COMPLETE` | Harm began at age one. The preregistered age-versus-count rule passed exactly at its 6/8 boundary; the queue-free age-0 row was separately identified as a batch/LR control. Modality damage crossed over with age. | [`results/queue_factorial`](../results/queue_factorial), [`configs/queue_factorial`](../configs/queue_factorial) |
| `Q03` | **Drift was necessary but not sufficient to explain queue damage** | `COMPLETE` | Cross-modality conditions with nearly matched cumulative drift still differed by several percentage points. Representation geometry, gradient sensitivity, and false-negative structure remained unseparated candidates. | [`results/queue_factorial/interpretation`](../results/queue_factorial/interpretation) |

### Track D — Data scale and transfer

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `D01` | **COCO scale pilot: unique in-distribution coverage mattered** | `COMPLETE` | At matched updates, full COCO exceeded the 25% subset by 15.03pp on Flickr validation, justifying a bounded broader-data test under the preregistered rule. | [`results/data_scale_pilot`](../results/data_scale_pilot), [`configs/data_scale_pilot`](../configs/data_scale_pilot) |
| `D02` | **CC3M-mirror scaling: volume alone did not transfer within budget** | `COMPLETE` | The qualified pixparse CC3M mirror underperformed the COCO control by 8.13pp at matched compute and 5.89pp after four passes. The extended arm reached its ceiling without convergence, so a universal claim about CC3M was not made. | [`results/cc3m_scale`](../results/cc3m_scale), [`docs/cc3m_acquisition_protocol.md`](../docs/cc3m_acquisition_protocol.md) |
| `D03` | **Mixed COCO + CC3M recovered much of the transfer loss** | `COMPLETE` | The mixed arm reached 53.133% Flickr-validation mean R@1 versus 47.209% for CC3M-only, showing that source/caption distribution mattered more than raw pair count alone under this protocol. | [`results/mixed_data_training`](../results/mixed_data_training), [`configs/mixed_data_training`](../configs/mixed_data_training) |

### Track E — Efficiency, distillation, and architecture

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `E01` | **Initial efficiency frontier: protocol details changed the apparent frontier** | `COMPLETE` | Flickr30k was made primary because COCO had selection leakage. Dynamic padding, deployed MobileCLIP2 fusion, and corrected operator-level FLOPs replaced misleading first-pass measurements. | [`results/efficiency_frontier`](../results/efficiency_frontier), [`docs/efficiency_frontier.md`](../docs/efficiency_frontier.md) |
| `E02` | **Wave 1: distillation still helped after the recipe was repaired** | `COMPLETE` | Distillation was not merely compensating for the broken queue recipe; MobileCLIP2 gave a reproducible gain over its matched corrected baseline at no inference-time teacher cost. | [`results/alignment_wave1`](../results/alignment_wave1), [`configs/alignment_wave1`](../configs/alignment_wave1) |
| `E03` | **Resolution curve: 224 px closed the latency gap without positional mismatch** | `COMPLETE` | 224 px was selected as the lowest-cost setting that cleared the OpenCLIP latency budget; 192 px traded further accuracy for speed with no additional frontier value. DINOv3's dynamic rotary encoding avoided learned-position interpolation artifacts. | [`results/resolution_arm`](../results/resolution_arm), [`results/resolution_distillation_224`](../results/resolution_distillation_224) |
| `E04` | **Long 224-px distillation: training and transfer-epoch selection were measured** | `COMPLETE` | A 24-epoch trajectory established the MobileCLIP2-distilled 224-px baseline and quantified the difference between COCO-dev and Flickr-validation epoch selection. | [`results/resolution_distillation_224_long`](../results/resolution_distillation_224_long) |
| `E05` | **Teacher benchmark strength did not predict student transfer** | `COMPLETE` | Matched single-teacher results were non-monotonic across OpenCLIP, MobileCLIP2, and SigLIP2. Equal 0.5/0.5 multi-teacher distillation did not beat MobileCLIP2 alone; this did not rule out tuned mixtures. | [`results/teacher_extension_224`](../results/teacher_extension_224) |
| `E06` | **Patch access helped; learned patch selection helped much more** | `COMPLETE` | Mean pooling recovered about 1.33pp, whereas learned-query and transformer aggregation recovered about 6.08pp and 6.91pp. Access to patch information alone was insufficient. | [`results/token_projection_training`](../results/token_projection_training), [`results/token_projection_profile`](../results/token_projection_profile) |
| `E07` | **Vision-token scaling saturated below the 5M parameter budget** | `COMPLETE` | The 2x2 width/depth study selected `C4` (256-d, two blocks). Width and depth both helped but composed sub-additively; further capacity was not justified. | [`results/token_aggregator_scale_training`](../results/token_aggregator_scale_training), [`results/token_aggregator_scale_profile`](../results/token_aggregator_scale_profile) |
| `E08` | **Text aggregation gave a small eligible gain; E5 missed the latency gate** | `COMPLETE` | MiniLM learned-query aggregation (`M_T1`) raised Flickr-validation mean R@1 to 54.487%. E5 variants exceeded the independently frozen 9.480-ms ceiling and were not promoted. Repeated paired profiling placed M_T1 pooled Q3 near 8.998 ms. | [`results/text_aggregation_study`](../results/text_aggregation_study), [`configs/text_aggregation_study`](../configs/text_aggregation_study) |
| `E09` | **Final LR retuning did not improve the selected model** | `COMPLETE` | LR factor 1.5 lost 0.763pp to the current factor and failed the promotion rule; factor 1.0 remained final. | [`results/final_lr_study`](../results/final_lr_study), [`configs/final_lr_study`](../configs/final_lr_study) |

### Track X — Diagnostics and bounded follow-ups

| ID | Headline | Status | Finding / role | Physical evidence |
|---|---|---|---|---|
| `X01` | **Probe 1: the dominant deficit was fine ranking, not gross retrieval** | `COMPLETE` | Rank and ambiguity diagnostics showed that correct items were commonly in the neighbourhood but not at rank one. The audit was corroborating evidence, not the load-bearing metric. | [`results/probe1_error_decomposition`](../results/probe1_error_decomposition) |
| `X02` | **Guarded hard-negative mining was infeasible under its frozen safety rules** | `STOPPED_BY_GATE` | The mining feasibility gate returned `INFEASIBLE_UNDER_PREREGISTERED_GUARD`; no training was authorized and thresholds were not relaxed after observation. | [`results/guarded_hard_negative_study`](../results/guarded_hard_negative_study), [`configs/guarded_hard_negative_study`](../configs/guarded_hard_negative_study) |
| `X03` | **FreezeShift: bounded encoder-adaptation upper bound** | `COMPLETE` | All three rank-128 LoRA arms improved over M_T1 by more than one pooled SD. Under the repaired same-allocation latency comparison all were parameter- and latency-eligible; dual-tower adaptation was selected at 63.416% validation mean R@1 and 4.862M trainable parameters. It remained 6.525pp below OpenCLIP and is reported outside the fully frozen claim. | [`freezeshift`](../freezeshift), [`docs/final_model_freeze.md`](../docs/final_model_freeze.md) |
| `X04` | **TokenShift: internal token merging reduced latency** | `COMPLETE` | Fixed 2×2 merging after DINOv3 block 8 reduced mean Q3 by 0.984 ms; merging after block 6 reduced it by 1.606 ms. This profile justified the bounded accuracy study in `X06`; it was not itself a promotion decision. | [`tokenshift`](../tokenshift) |
| `X05` | **Same-allocation latency amendment** | `COMPLETE` | The original 9.480-ms cross-session gate invalidly failed its unchanged M_T1 control. A declared corrective allocation profiled OpenCLIP, M_T1 and all FreezeShift arms together; all candidates were significantly faster than OpenCLIP on paired Q3 latency. The original verdict remains preserved as historical evidence. | [`latency_amendment`](../latency_amendment) |
| `X06` | **TokenShift accuracy study** | `COMPLETE` | All four three-seed arms reduced latency but lost accuracy. Block-8 merging cost 12.581pp on frozen M_T1 and 5.690pp on dual LoRA; block-6 merging cost 15.950pp and 10.861pp. The no-merge parents therefore remained final before the roster was frozen for test evaluation. | [`tokenshift_training`](../tokenshift_training) |

## Canonical narrative in one paragraph

A weak frozen-alignment result was first made comparable to strong locally
evaluated references. Compatibility screening then failed, but the failure was
not repaired by changing its gate. Instead, a recipe factorial isolated a
training-only memory queue as the dominant cause of the collapse. A dedicated
factorial established immediate harm, a dose response, modality crossover, and
the insufficiency of representation drift as a complete mechanism. With the
recipe repaired, data-scale experiments showed that unique coverage helped
within COCO but raw web-scale volume did not transfer within the tested compute
budget; mixing COCO back in recovered performance. The efficiency track then
corrected measurement artifacts, reduced resolution to meet the latency budget,
showed that teacher strength alone did not select the best teacher, and recovered
large gains by learning which frozen image patches to retain. Vision aggregation
eventually saturated, while a small text-side aggregator produced the best fully
frozen-backbone model. Final LR retuning did not improve it, and guarded
hard-negative training was stopped by its preregistered safety gate. A bounded
LoRA follow-up then recovered a further 8.93pp within the 5M parameter budget.
The original cross-session latency gate was shown invalid and repaired with a
declared same-allocation comparison; the dual-tower arm became the final
development-selected adaptation model. TokenShift subsequently demonstrated
that fixed internal token merging can reduce latency, which motivated the final
accuracy experiment. That closure showed the speedup came with a 5.69--15.95pp
accuracy loss, so it strengthened the no-merge decision instead of changing the
selected models.

## Naming rule for all future work

1. Allocate the next stable ID in `experiments/registry.yaml`.
2. Give it a research headline, not a code nickname.
3. Record one primary question and one primary outcome.
4. Keep the existing implementation directory name as `physical_paths`.
5. Never label a built or submitted experiment `COMPLETE` until its result-level
   report is closed.
6. Never reuse an ID after a branch is stopped or superseded.

Validate the index after adding a result directory or registry entry:

```bash
python experiments/check_registry.py
```
