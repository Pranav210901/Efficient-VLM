# Retrospective paired-bootstrap uncertainty

Effects are left minus right in percentage points. Intervals quantify dataset/query sampling conditional on the frozen checkpoints; they are not training-population confidence intervals. Student predictions are averaged over seeds before resampling, with paired seed-effect SD shown separately where available.

| comparison           | left_model        | right_model                       | dataset           | metric        |   n_image_queries |   n_text_queries |    effect |    ci_low |   ci_high |   bootstrap_iterations |   paired_seed_effect_sd_pp |     n |
|:---------------------|:------------------|:----------------------------------|:------------------|:--------------|------------------:|-----------------:|----------:|----------:|----------:|-----------------------:|---------------------------:|------:|
| dual_vs_strict       | mt1_dual_lora     | mt1_strict_frozen                 | flickr30k_test    | mean_R@1      |              1000 |             5000 |   9.29667 |   8.15    |  10.49    |                  10000 |                   0.698737 |   nan |
| dual_vs_strict       | mt1_dual_lora     | mt1_strict_frozen                 | cifar100_zeroshot | top1_accuracy |               nan |              nan |   1.55333 |   1       |   2.11667 |                  10000 |                   1.29032  | 10000 |
| dual_vs_strict       | mt1_dual_lora     | mt1_strict_frozen                 | pets_zeroshot     | top1_accuracy |               nan |              nan |  -1.8352  |  -2.59835 |  -1.09022 |                  10000 |                   1.67156  |  3669 |
| dual_vs_strict       | mt1_dual_lora     | mt1_strict_frozen                 | eurosat_zeroshot  | top1_accuracy |               nan |              nan |  -2.28807 |  -3.41564 |  -1.14403 |                  10000 |                   1.3637   |  4050 |
| strict_vs_openclip   | mt1_strict_frozen | openclip_vit_b32_quickgelu_openai | flickr30k_test    | mean_R@1      |              1000 |             5000 | -15.32    | -17.05    | -13.5766  |                  10000 |                 nan        |   nan |
| strict_vs_openclip   | mt1_strict_frozen | openclip_vit_b32_quickgelu_openai | cifar100_zeroshot | top1_accuracy |               nan |              nan | -28.31    | -29.3533  | -27.28    |                  10000 |                 nan        | 10000 |
| strict_vs_openclip   | mt1_strict_frozen | openclip_vit_b32_quickgelu_openai | pets_zeroshot     | top1_accuracy |               nan |              nan | -73.0989  | -74.4617  | -71.7362  |                  10000 |                 nan        |  3669 |
| strict_vs_openclip   | mt1_strict_frozen | openclip_vit_b32_quickgelu_openai | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -10.1975  | -12.1235  |  -8.32901 |                  10000 |                 nan        |  4050 |
| dual_vs_openclip     | mt1_dual_lora     | openclip_vit_b32_quickgelu_openai | flickr30k_test    | mean_R@1      |              1000 |             5000 |  -6.02333 |  -7.62008 |  -4.36667 |                  10000 |                 nan        |   nan |
| dual_vs_openclip     | mt1_dual_lora     | openclip_vit_b32_quickgelu_openai | cifar100_zeroshot | top1_accuracy |               nan |              nan | -26.7567  | -27.8     | -25.7367  |                  10000 |                 nan        | 10000 |
| dual_vs_openclip     | mt1_dual_lora     | openclip_vit_b32_quickgelu_openai | pets_zeroshot     | top1_accuracy |               nan |              nan | -74.9341  | -76.2878  | -73.5532  |                  10000 |                 nan        |  3669 |
| dual_vs_openclip     | mt1_dual_lora     | openclip_vit_b32_quickgelu_openai | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -12.4856  | -14.3045  | -10.6829  |                  10000 |                 nan        |  4050 |
| strict_vs_mobileclip | mt1_strict_frozen | mobileclip2_s0_dfndr2b            | flickr30k_test    | mean_R@1      |              1000 |             5000 | -25.35    | -26.8568  | -23.8033  |                  10000 |                 nan        |   nan |
| strict_vs_mobileclip | mt1_strict_frozen | mobileclip2_s0_dfndr2b            | cifar100_zeroshot | top1_accuracy |               nan |              nan | -41.7     | -42.7133  | -40.6733  |                  10000 |                 nan        | 10000 |
| strict_vs_mobileclip | mt1_strict_frozen | mobileclip2_s0_dfndr2b            | pets_zeroshot     | top1_accuracy |               nan |              nan | -79.3404  | -80.5487  | -78.0958  |                  10000 |                 nan        |  3669 |
| strict_vs_mobileclip | mt1_strict_frozen | mobileclip2_s0_dfndr2b            | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -17.1111  | -18.6093  | -15.5885  |                  10000 |                 nan        |  4050 |
| dual_vs_mobileclip   | mt1_dual_lora     | mobileclip2_s0_dfndr2b            | flickr30k_test    | mean_R@1      |              1000 |             5000 | -16.0533  | -17.5633  | -14.5767  |                  10000 |                 nan        |   nan |
| dual_vs_mobileclip   | mt1_dual_lora     | mobileclip2_s0_dfndr2b            | cifar100_zeroshot | top1_accuracy |               nan |              nan | -40.1467  | -41.13    | -39.1466  |                  10000 |                 nan        | 10000 |
| dual_vs_mobileclip   | mt1_dual_lora     | mobileclip2_s0_dfndr2b            | pets_zeroshot     | top1_accuracy |               nan |              nan | -81.1756  | -82.3749  | -79.9855  |                  10000 |                 nan        |  3669 |
| dual_vs_mobileclip   | mt1_dual_lora     | mobileclip2_s0_dfndr2b            | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -19.3992  | -21.0617  | -17.7284  |                  10000 |                 nan        |  4050 |
| strict_vs_siglip2    | mt1_strict_frozen | siglip2_vit_b32_256_webli         | flickr30k_test    | mean_R@1      |              1000 |             5000 | -27.56    | -29.1668  | -25.9667  |                  10000 |                 nan        |   nan |
| strict_vs_siglip2    | mt1_strict_frozen | siglip2_vit_b32_256_webli         | cifar100_zeroshot | top1_accuracy |               nan |              nan | -39.5     | -40.4801  | -38.5266  |                  10000 |                 nan        | 10000 |
| strict_vs_siglip2    | mt1_strict_frozen | siglip2_vit_b32_256_webli         | pets_zeroshot     | top1_accuracy |               nan |              nan | -82.8564  | -83.9738  | -81.7116  |                  10000 |                 nan        |  3669 |
| strict_vs_siglip2    | mt1_strict_frozen | siglip2_vit_b32_256_webli         | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -17.4074  | -19.1687  | -15.6049  |                  10000 |                 nan        |  4050 |
| dual_vs_siglip2      | mt1_dual_lora     | siglip2_vit_b32_256_webli         | flickr30k_test    | mean_R@1      |              1000 |             5000 | -18.2633  | -19.79    | -16.6933  |                  10000 |                 nan        |   nan |
| dual_vs_siglip2      | mt1_dual_lora     | siglip2_vit_b32_256_webli         | cifar100_zeroshot | top1_accuracy |               nan |              nan | -37.9467  | -38.9533  | -36.94    |                  10000 |                 nan        | 10000 |
| dual_vs_siglip2      | mt1_dual_lora     | siglip2_vit_b32_256_webli         | pets_zeroshot     | top1_accuracy |               nan |              nan | -84.6916  | -85.7636  | -83.6195  |                  10000 |                 nan        |  3669 |
| dual_vs_siglip2      | mt1_dual_lora     | siglip2_vit_b32_256_webli         | eurosat_zeroshot  | top1_accuracy |               nan |              nan | -19.6955  | -21.3745  | -18.0329  |                  10000 |                 nan        |  4050 |

## Queue staleness share

```json
{
  "n": 6,
  "effect": 0.3179076679900849,
  "ci_low": 0.31446749560541676,
  "ci_high": 0.32195061337353925,
  "bootstrap_iterations": 10000,
  "effect_percent": 31.79076679900849,
  "ci_low_percent": 31.446749560541676,
  "ci_high_percent": 32.19506133735393,
  "unit": "six age-by-seed cells; descriptive cell bootstrap"
}
```
