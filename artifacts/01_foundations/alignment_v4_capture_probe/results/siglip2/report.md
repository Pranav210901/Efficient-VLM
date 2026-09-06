# Alignment v4 distillation capture probe

Status: **COMPLETE**

## Locally measured results

| status   | source         | kind        | experiment_id        | run_id                        |   seed |   i2t_R@1 |   i2t_R@5 |   i2t_R@10 |   mean_rank_i2t |   median_rank_i2t |   t2i_R@1 |   t2i_R@5 |   t2i_R@10 |   mean_rank_t2i |   median_rank_t2i |   num_image_queries |   num_text_queries |   mean_R@1 |   image_query_latency_ms |   text_query_latency_ms |   bidirectional_pair_latency_ms |   peak_inference_memory_bytes |   params_total_training |   params_total_inference |   params_trainable_training |   params_trainable_inference |   params_training_only | gpu                                          | checkpoint                                                                         | fingerprint_digest                                               |
|:---------|:---------------|:------------|:---------------------|:------------------------------|-------:|----------:|----------:|-----------:|----------------:|------------------:|----------:|----------:|-----------:|----------------:|------------------:|--------------------:|-------------------:|-----------:|-------------------------:|------------------------:|--------------------------------:|------------------------------:|------------------------:|-------------------------:|----------------------------:|-----------------------------:|-----------------------:|:---------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| COMPLETE | measured_local | sensitivity | distill_strength_1p0 | distill_strength_1p0__seed_42 |     42 |    0.267  |    0.5474 |     0.6778 |         19.4826 |                 4 |  0.201311 |  0.455879 |   0.59046  |         29.7214 |                 7 |                5000 |              25011 |   0.234156 |                  2.66308 |                 1.10838 |                         3.77146 |                   1.99578e+08 |                46368387 |                 45778563 |                     2068227 |                      1478403 |                 589824 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/siglip2/sensitivity/distill_strength_1p0__seed_42/best.pt | 6a600525d512e70a5a5bdeee0cf4d06ab874521c05e5eca43cfd2216eb028ba6 |
| COMPLETE | measured_local | sensitivity | distill_strength_1p0 | distill_strength_1p0__seed_43 |     43 |    0.262  |    0.545  |     0.6704 |         19.7374 |                 4 |  0.201351 |  0.45552  |   0.589621 |         29.8994 |                 7 |                5000 |              25011 |   0.231676 |                  2.72608 |                 1.1119  |                         3.83798 |                   1.99578e+08 |                46368387 |                 45778563 |                     2068227 |                      1478403 |                 589824 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/siglip2/sensitivity/distill_strength_1p0__seed_43/best.pt | 34d394c8f713c11af8bb687c4a37fb38259ddb352546c9d6587d22e7483160d9 |
| COMPLETE | measured_local | sensitivity | matched_baseline     | matched_baseline__seed_42     |     42 |    0.1566 |    0.3848 |     0.5146 |         37.6772 |                10 |  0.109832 |  0.305785 |   0.433169 |         48.5407 |                14 |                5000 |              25011 |   0.133216 |                  2.67933 |                 1.14618 |                         3.82551 |                   1.97218e+08 |                45778563 |                 45778563 |                     1478403 |                      1478403 |                      0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/siglip2/sensitivity/matched_baseline__seed_42/best.pt     | 68a6fd757575a2ee8154b3cabd2da3e6841b9469741419a7b98c19afb7dddb13 |
| COMPLETE | measured_local | sensitivity | matched_baseline     | matched_baseline__seed_43     |     43 |    0.1544 |    0.3822 |     0.5194 |         37.4526 |                10 |  0.111311 |  0.305466 |   0.43149  |         48.717  |                14 |                5000 |              25011 |   0.132856 |                  2.68681 |                 1.15549 |                         3.84229 |                   1.97218e+08 |                45778563 |                 45778563 |                     1478403 |                      1478403 |                      0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | checkpoints/alignment_v4/siglip2/sensitivity/matched_baseline__seed_43/best.pt     | 723279812231cad20123a1fcf8effae24a474cce59d08e5559f4df8614db6b8e |

## Capture probe decision

```json
{
  "status": "COMPLETE",
  "proceed": false,
  "teacher_id": "siglip2_vit_b32_256_webli",
  "teacher_R@1": 0.5688067078590393,
  "teacher_anchor_split": "sealed_coco_val_5000",
  "probe_split": "coco_dev_5000",
  "baseline_R@1": 0.1330356765538454,
  "baseline_seed_values": [
    0.13321583718061447,
    0.13285551592707634
  ],
  "baseline_seed_std": 0.0002547856017824502,
  "distilled_R@1": 0.2329157032072544,
  "distilled_seed_values": [
    0.23415570706129074,
    0.23167569935321808
  ],
  "distilled_seed_std": 0.0017536302677730873,
  "delta_R@1": 0.099880026653409,
  "capture_fraction": 0.22920299762527732,
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
