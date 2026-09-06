# Alignment v2 results

Status: **COMPLETE**

All retrieval metrics use the same grouped five-caption COCO validation protocol. Latency is measured independently for batch-one image and text queries on the executing GPU.

| kind             | experiment_id                     |   seed |   i2t_R@1 |   t2i_R@1 |   mean_R@1 |   bidirectional_pair_latency_ms |   params_total |   params_trainable |
|:-----------------|:----------------------------------|-------:|----------:|----------:|-----------:|--------------------------------:|---------------:|-------------------:|
| frozen_unimodal  | convnext_minilm_residual_mp       |     42 |    0.1636 | 0.144479  |  0.15404   |                         2.82245 |    5.28169e+07 |        2.28352e+06 |
| frozen_unimodal  | convnext_minilm_residual_mp       |     43 |    0.1694 | 0.146158  |  0.157779  |                         2.89068 |    5.28169e+07 |        2.28352e+06 |
| frozen_unimodal  | convnext_minilm_residual_mp       |     44 |    0.1664 | 0.145958  |  0.156179  |                         2.97163 |    5.28169e+07 |        2.28352e+06 |
| frozen_unimodal  | dinov2_minilm_residual_mp         |     42 |    0.1466 | 0.131846  |  0.139223  |                         4.776   |    4.68567e+07 |        2.08692e+06 |
| frozen_unimodal  | dinov2_minilm_residual_mp         |     43 |    0.1406 | 0.130167  |  0.135384  |                         4.85035 |    4.68567e+07 |        2.08692e+06 |
| frozen_unimodal  | dinov2_minilm_residual_mp         |     44 |    0.1428 | 0.129368  |  0.136084  |                         4.80081 |    4.68567e+07 |        2.08692e+06 |
| frozen_unimodal  | efficientnet_bge_residual_mp      |     42 |    0.086  | 0.080435  |  0.0832175 |                         4.27894 |    3.99132e+07 |        2.54567e+06 |
| frozen_unimodal  | efficientnet_bge_residual_mp      |     43 |    0.0836 | 0.080435  |  0.0820175 |                         4.0883  |    3.99132e+07 |        2.54567e+06 |
| frozen_unimodal  | efficientnet_bge_residual_mp      |     44 |    0.0826 | 0.0796754 |  0.0811377 |                         3.96052 |    3.99132e+07 |        2.54567e+06 |
| paired_reference | openclip_vit_b32_quickgelu_openai |    nan |    0.5002 | 0.30355   |  0.401875  |                         4.16965 |    1.51277e+08 |        0           |
