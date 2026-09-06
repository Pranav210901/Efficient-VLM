# Wave 0 v3-pair re-screen

Status: **RANKING_CHANGED_STOP_FOR_DECISION**

The locked student remains **DINOv3 ViT-S/16 + all-MiniLM-L6-v2**. The ranking change is a finding, but the main-study pair is permanently retained to preserve continuity with the MiniLM queue curve, screen, teacher caches, and 31 claim-bearing runs. Wave 1 remains blocked.

## Structure of the ranking change

- **Vision ranking is stable:** ViT-S/16 exceeds ConvNeXt-Tiny at every text encoder under both queued and corrected recipes.
- **Text ranking inverted:** MiniLM leads within both vision families under the queue; after correction both are E5 > BGE > MiniLM.
- This is modality-specific: text-encoder selection changed while vision-encoder selection remained intact.
- The text side enqueues about 5× as many rows per step. Across these six pairs, queue removal improved i2t by 26.28pp versus 17.26pp for t2i, an extra 9.02pp on i2t. Together with the text-ranking inversion, these are three observations pointing in the same direction.
- The queued band was 14.20–17.02% (2.82pp); corrected is 35.91–38.90% (2.99pp). Both total spreads are about 3pp; the defensible claim is suppression plus ranking flattening/inversion, not a materially smaller max–min range.

| experiment_id                          | vision_encoder       | text_encoder     |   v3_dev_R@1 |   v3_i2t_R@1 |   v3_t2i_R@1 |   v3_rank |   rescreen_dev_R@1 |   rescreen_i2t_R@1 |   rescreen_t2i_R@1 |   rescreen_rank |
|:---------------------------------------|:---------------------|:-----------------|-------------:|-------------:|-------------:|----------:|-------------------:|-------------------:|-------------------:|----------------:|
| dinov3_vits16__e5_small_v2             | dinov3_vits16        | e5_small_v2      |     0.16179  |       0.186  |     0.137579 |         3 |           0.38897  |             0.4614 |           0.316541 |               1 |
| dinov3_vits16__bge_small_en            | dinov3_vits16        | bge_small_en     |     0.169608 |       0.1958 |     0.143417 |         2 |           0.38159  |             0.446  |           0.31718  |               2 |
| dinov3_vits16__all_minilm_l6_v2        | dinov3_vits16        | all_minilm_l6_v2 |     0.170167 |       0.1908 |     0.149534 |         1 |           0.375972 |             0.4414 |           0.310543 |               3 |
| dinov3_convnext_tiny__e5_small_v2      | dinov3_convnext_tiny | e5_small_v2      |     0.142372 |       0.158  |     0.126744 |         5 |           0.374711 |             0.4378 |           0.311623 |               4 |
| dinov3_convnext_tiny__bge_small_en     | dinov3_convnext_tiny | bge_small_en     |     0.141951 |       0.1524 |     0.131502 |         6 |           0.372571 |             0.4312 |           0.313942 |               5 |
| dinov3_convnext_tiny__all_minilm_l6_v2 | dinov3_convnext_tiny | all_minilm_l6_v2 |     0.160867 |       0.1732 |     0.148535 |         4 |           0.359093 |             0.4152 |           0.302987 |               6 |
