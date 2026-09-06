# Residual fresh-queue mechanism

Status: **COMPLETE**. Primary metric is Flickr30k-validation mean bidirectional R@1.

## Fresh-capacity curve

|   queue_size |   n |     mean |          sd |   minimum |   maximum |
|-------------:|----:|---------:|------------:|----------:|----------:|
|            0 |   3 | 0.373212 | 0.00267427  |  0.370632 |  0.375972 |
|         1024 |   3 | 0.308865 | 0.00346988  |  0.305419 |  0.312359 |
|         4096 |   3 | 0.252456 | 0.00126244  |  0.25103  |  0.25343  |
|        16384 |   3 | 0.208999 | 0.0010887   |  0.207759 |  0.209798 |
|        65536 |   3 | 0.193356 | 0.000960426 |  0.192262 |  0.194062 |

## Fixed-capacity mechanisms

| arm                      |   n |     mean |         sd |   minimum |   maximum |
|:-------------------------|----:|---------:|-----------:|----------:|----------:|
| mass_normalized          |   3 | 0.271102 | 0.00249648 |  0.269408 |  0.273969 |
| matched_random_filter    |   3 | 0.209272 | 0.0012586  |  0.207819 |  0.210018 |
| semantic_filter          |   3 | 0.261981 | 0.00270719 |  0.260008 |  0.265068 |
| unweighted_fresh_control |   3 | 0.208999 | 0.0010887  |  0.207759 |  0.209798 |

## Preregistered paired effects

```json
{
  "semantic_vs_matched_random": {
    "n": 3,
    "mean_pp": 5.270961175362269,
    "sd_pp": 0.21667747599958784,
    "clears_practical_threshold": true
  },
  "mass_normalized_vs_control": {
    "n": 3,
    "mean_pp": 6.210317959388097,
    "sd_pp": 0.3583728310427993,
    "clears_practical_threshold": true
  },
  "semantic_vs_control": {
    "n": 3,
    "mean_pp": 5.298282454411189,
    "sd_pp": 0.2021938957425653,
    "clears_practical_threshold": true
  }
}
```

All new arms use the fresh reprojected queue. The semantic threshold and teacher cache were frozen before these outcomes; matched-random filtering removes the same per-query count.
