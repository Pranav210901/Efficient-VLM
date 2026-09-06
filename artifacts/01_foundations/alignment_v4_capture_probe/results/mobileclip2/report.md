# Alignment v4 distillation capture probe

Status: **COMPLETE**

## Locally measured results

| status   | source         | kind        | experiment_id        | run_id                        |   seed |   i2t_R@1 |   i2t_R@5 |   i2t_R@10 |   mean_rank_i2t |   median_rank_i2t |   t2i_R@1 |   t2i_R@5 |   t2i_R@10 |   mean_rank_t2i |   median_rank_t2i |   num_image_queries |   num_text_queries |   mean_R@1 |   image_query_latency_ms |   text_query_latency_ms |   bidirectional_pair_latency_ms |   peak_inference_memory_bytes |   params_total_training |   params_total_inference |   params_trainable_training |   params_trainable_inference |   params_training_only | gpu                                          | checkpoint                                                                             | fingerprint_digest                                               |
|:---------|:---------------|:------------|:---------------------|:------------------------------|-------:|----------:|----------:|-----------:|----------------:|------------------:|----------:|----------:|-----------:|----------------:|------------------:|--------------------:|-------------------:|-----------:|-------------------------:|------------------------:|--------------------------------:|------------------------------:|------------------------:|-------------------------:|----------------------------:|-----------------------------:|-----------------------:|:---------------------------------------------|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| COMPLETE | measured_local | sensitivity | distill_strength_1p0 | distill_strength_1p0__seed_42 |     42 |    0.2564 |    0.5248 |     0.6548 |         21.4492 |                 5 |  0.190556 |  0.436648 |   0.569709 |         31.8327 |                 7 |                5000 |              25011 |   0.223478 |                  2.70518 |                 1.21794 |                         3.92312 |                   1.98791e+08 |                46171779 |                 45778563 |                     1871619 |                      1478403 |                 393216 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/mobileclip2/sensitivity/distill_strength_1p0__seed_42/best.pt | 77855d135da12030a8ec273bbe2bc1a396d05b2264d180957fa7205d3a15c9df |
| COMPLETE | measured_local | sensitivity | distill_strength_1p0 | distill_strength_1p0__seed_43 |     43 |    0.258  |    0.5294 |     0.6588 |         21.3766 |                 5 |  0.189357 |  0.438847 |   0.570549 |         31.7687 |                 7 |                5000 |              25011 |   0.223678 |                  2.7159  |                 1.14255 |                         3.85845 |                   1.98791e+08 |                46171779 |                 45778563 |                     1871619 |                      1478403 |                 393216 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/mobileclip2/sensitivity/distill_strength_1p0__seed_43/best.pt | 288d699e51a9c3aa38da8314e5c41edb5a5dc60e7ae949ce36ecddc38ef16952 |
| COMPLETE | measured_local | sensitivity | matched_baseline     | matched_baseline__seed_42     |     42 |    0.1566 |    0.3848 |     0.5146 |         37.6772 |                10 |  0.109832 |  0.305785 |   0.433169 |         48.5407 |                14 |                5000 |              25011 |   0.133216 |                  2.70864 |                 1.13457 |                         3.84321 |                   1.97218e+08 |                45778563 |                 45778563 |                     1478403 |                      1478403 |                      0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/mobileclip2/sensitivity/matched_baseline__seed_42/best.pt     | dc161134a8b53682ea9c16a6c0d8c26a3bedb54a4b234dd0ff4ec990b1749382 |
| COMPLETE | measured_local | sensitivity | matched_baseline     | matched_baseline__seed_43     |     43 |    0.1544 |    0.3822 |     0.5194 |         37.4526 |                10 |  0.111311 |  0.305466 |   0.43149  |         48.717  |                14 |                5000 |              25011 |   0.132856 |                  2.69508 |                 1.12572 |                         3.8208  |                   1.97218e+08 |                45778563 |                 45778563 |                     1478403 |                      1478403 |                      0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/mobileclip2/sensitivity/matched_baseline__seed_43/best.pt     | b2ea6947b6d75da591ee87d2a5676f1ffd5db70ea3b4844292a3508937abb079 |

## Capture probe decision

```json
{
  "status": "COMPLETE",
  "proceed": false,
  "teacher_id": "mobileclip2_s0_dfndr2b",
  "teacher_R@1": 0.5298200398683548,
  "teacher_anchor_split": "sealed_coco_val_5000",
  "probe_split": "coco_dev_5000",
  "baseline_R@1": 0.1330356765538454,
  "baseline_seed_values": [
    0.13321583718061447,
    0.13285551592707634
  ],
  "baseline_seed_std": 0.0002547856017824502,
  "distilled_R@1": 0.22357820346951485,
  "distilled_seed_values": [
    0.2234780713915825,
    0.2236783355474472
  ],
  "distilled_seed_std": 0.00014160814264054008,
  "delta_R@1": 0.09054252691566944,
  "capture_fraction": 0.22819076376732442,
  "distillation_strength": 1.0,
  "viable_threshold_R@1": 0.3,
  "nonviable_threshold_R@1": 0.22,
  "decision": "INCONCLUSIVE_MIDDLE",
  "caveat": "Teacher anchor is measured on sealed COCO-val while the probe is evaluated on COCO-dev-5000."
}
```


## Literature-only context

No literature rows configured.

Small-seed confidence intervals use Student-t critical values and are labelled unstable.
