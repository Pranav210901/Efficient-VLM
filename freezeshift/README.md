# FreezeShift

Bounded parameter-efficient adaptation study for the final M_T1 dual encoder.
The completed `alignment_vlm` results are inputs and are never modified.

## Frozen baseline

- DINOv3 ViT-S/16, 224 px, C4 patch-token aggregation
- all-MiniLM-L6-v2 with learned-query text aggregation
- MobileCLIP2-S0 distillation
- queue-free InfoNCE, batch 1024, 24-epoch cosine
- LR `0.002545584412271571`
- 2,896,389 trainable inference parameters
- 54.487% Flickr30k-validation mean bidirectional R@1

## Arms

All arms use rank-128 LoRA on QKV/output attention projections in the final
four DINOv3 blocks and/or query/value projections in the final four MiniLM
blocks. Base encoder weights remain frozen. LoRA is merged before latency
profiling.

| Arm | Added LoRA parameters | Total trainable inference parameters |
|---|---:|---:|
| Vision-only | 1,179,648 | 4,076,037 |
| Text-only | 786,432 | 3,682,821 |
| Dual-tower | 1,966,080 | 4,862,469 |

## Execute

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
source .venv/bin/activate
bash freezeshift/submit.sh --partition teaching --max-total-gpus 8 --resume
```

No Flickr30k test evaluation is included. The original report used validation,
the 5M adaptation budget, and the frozen 9.480 ms Q3 latency ceiling.

## Reporting amendment

After all training and epoch evaluations completed, dual-arm seed 43 produced
an exact Flickr-validation maximum at both epochs 23 and 24. The original
reporter correctly stopped because no tie policy had been declared. A
post-observation deterministic amendment now selects the earliest epoch for an
exact tie and records the event in every arm report. This does not change any
checkpoint or metric and leaves Flickr test sealed.

The arm-report baseline was also corrected from a stale inherited 44.714%
value to the matched frozen M_T1 result: 54.487179% and repeated pooled Q3
latency 8.998041 ms.

Recover reports and profiling without retraining or reevaluating epochs:

```bash
bash freezeshift/resume_reporting.sh --partition teaching
```

## Completed result and measurement repair

All nine training runs completed. Vision-, text-, and dual-tower adaptation
reached 58.047%, 59.724%, and 63.416% validation mean R@1 respectively, versus
54.487% for M_T1; all remained below 5M trainable inference parameters.

The original cross-session 9.480-ms ceiling was subsequently shown invalid
because it rejected the unchanged contemporaneous M_T1 control. The declared
`latency_amendment/` profiled OpenCLIP, M_T1, and all arms in one allocation.
All three arms were faster than OpenCLIP with paired 95% bootstrap intervals
entirely below zero. Applying the original accuracy and parameter rules to that
repaired comparison selects the dual arm as the final development candidate.
The original negative verdict remains preserved in `results/report/report.json`;
the post-amendment decision is recorded in `docs/final_model_freeze.md`.
