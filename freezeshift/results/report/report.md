# FreezeShift result

**Status:** complete  
**Evaluation scope:** Flickr30k validation; Flickr30k test remained sealed

## Original preregistered result

The nine-run study completed, but the original cross-session 9.480-ms gate
classified every LoRA arm as ineligible. The machine-readable original verdict
is preserved in `report.json`; under that verdict the final arm remained M_T1.

| Arm | Validation mean R@1 | Gain vs M_T1 | Trainable parameters | Original Q3 | Original eligible |
|---|---:|---:|---:|---:|---|
| M_T1 | 54.487% | — | 2,896,389 | 9.497 ms | control |
| Vision LoRA | 58.047% | +3.560pp | 4,076,037 | 9.483 ms | no |
| Text LoRA | 59.724% | +5.237pp | 3,682,821 | 9.641 ms | no |
| Dual LoRA | **63.416%** | **+8.928pp** | 4,862,469 | 9.605 ms | no |

All gains exceeded one pooled standard deviation. None beat the locally
evaluated OpenCLIP validation reference of 69.941%.

## Measurement defect and corrective result

The fixed ceiling failed the unchanged contemporaneous M_T1 control, so the
cross-session decision was not a valid hardware comparison. A declared
post-observation measurement amendment profiled OpenCLIP, M_T1, and all three
arms in one allocation. All were faster than OpenCLIP; for dual LoRA the paired
mean Q3 difference was -0.101 ms with a 95% bootstrap interval of
[-0.145, -0.016] ms.

Under the repaired comparison, every arm is latency- and parameter-eligible.
Applying the unchanged accuracy promotion rule selects **dual LoRA** as the
final development adaptation candidate. This decision is recorded separately
in `docs/final_model_freeze.md` so the original report remains auditable.

## Claim boundary

FreezeShift is a bounded encoder-adaptation upper bound. It improves materially
over the fully frozen-backbone M_T1 model but does not establish a fully frozen
encoder result and does not beat OpenCLIP. No rank, layer, alpha, dropout,
learning-rate, or module search followed observation.
