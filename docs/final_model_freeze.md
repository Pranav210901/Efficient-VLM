# Final development-model freeze

**Recorded:** 10 August 2026  
**Scope:** historical development-selection freeze; the Flickr30k test was opened later by the no-training final-evaluation package

## Decision

The final bounded-adaptation development candidate is **FreezeShift dual**.
The final fully frozen-backbone candidate is **M_T1**.

FreezeShift dual is selected because it is the highest-accuracy arm satisfying
the study's original accuracy and parameter rules after the declared
same-allocation latency repair:

| Criterion | Frozen rule | Dual result | Decision |
|---|---:|---:|---|
| Flickr30k-validation mean bidirectional R@1 | maximise among eligible arms | 63.416% | best arm |
| Gain over M_T1 | greater than one pooled SD | +8.928pp vs 0.455pp | pass |
| Trainable inference parameters | less than 5,000,000 | 4,862,469 | pass |
| Paired mean Q3 latency vs OpenCLIP | at most 0 ms | -0.101 ms | pass |
| Latency uncertainty class | report, do not hide | 95% CI [-0.145, -0.016] ms | faster |

M_T1 remains the endpoint for the fully frozen-backbone claim: 54.487%
validation mean R@1, 2,896,389 trainable inference parameters, and a paired
mean Q3 difference of -0.119 ms against OpenCLIP.

## TokenShift closure

The completed TokenShift accuracy study does not alter either selection. Fixed
2×2 merging after block 8 or block 6 reduced latency but lost accuracy against
both matched no-merge parents:

| Parent | Block 8 loss | Block 6 loss | Decision |
|---|---:|---:|---|
| Fully frozen M_T1 | -12.581pp | -15.950pp | retain M_T1, no merge |
| FreezeShift dual | -5.690pp | -10.861pp | retain dual, no merge |

The block-8 dual arm was the least damaging and still did not justify trading
5.690pp for 0.934 ms. This is the frozen post-study decision required by the
package's prohibition on automatic promotion.

## Interpretation

The dual model reaches 90.67% of the locally evaluated OpenCLIP validation
score and closes 57.77% of the M_T1-to-OpenCLIP validation gap. It does not beat
OpenCLIP: the remaining difference is 6.525pp. Its result is an upper bound on
what limited encoder adaptation can recover, not evidence that fully frozen
encoders alone attain the same score.

## Amendment provenance

The original FreezeShift reporter compared arms with a 9.480-ms OpenCLIP-derived
ceiling measured in another session. That ceiling rejected the unchanged M_T1
control at 9.497 ms, demonstrating session contamination. The original negative
verdict is preserved in `freezeshift/results/report/report.json`.

The corrective protocol in `latency_amendment/amendment.yaml` was frozen after
the anomaly was observed but before corrective measurement. Slurm job 2297019
then measured OpenCLIP, M_T1, and all three arms in the same allocation with ten
independently warmed repetitions, randomised order, batch 64, native bfloat16,
and 100 timed iterations per repetition. This record applies that declared
replacement comparison; it does not rewrite the historical run.

## Test seal

`data/flickr30k/test.csv` has not been evaluated. Opening it is a one-shot final
evaluation decision and requires explicit author/supervisor approval. No model,
threshold, epoch, or architecture may change after that evaluation without the
test result being reclassified as development evidence.

## Evidence

- `freezeshift/predictions.json`
- `freezeshift/results/report/report.json`
- `latency_amendment/amendment.yaml`
- `latency_amendment/results/report/report.json`
- `results/text_aggregation_study/training/M_T1/report/report.json`
- `tokenshift_training/results/report/report.json`
