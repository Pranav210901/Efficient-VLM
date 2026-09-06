# E5 baseline component-level latency diagnostic

Diagnostic only: no training, checkpoint mutation, re-gating, or ceiling change.

## 1. Component latency breakdown

Isolated component timings use cached module boundaries and are not additive. Percentages use each cell's full-stack neural Q3 denominator.

| Stage | MiniLM M-T0 Q3 (ms) | M-T0 % | E5 E-T0 Q3 (ms) | E-T0 % |
|---|---:|---:|---:|---:|
| cpu_tokenization | 1.6948 | 19.07% | 1.7418 | 17.24% |
| vision_encoder | 6.9103 | 77.76% | 6.9776 | 69.05% |
| image_aggregation | 0.6121 | 6.89% | 0.5885 | 5.82% |
| image_projection | 0.0669 | 0.75% | 0.0661 | 0.65% |
| text_encoder | 1.2750 | 14.35% | 2.4111 | 23.86% |
| text_aggregation | 0.0291 | 0.33% | 0.0292 | 0.29% |
| text_projection | 0.0627 | 0.71% | 0.0621 | 0.61% |
| **Full-stack neural** | **8.8870** | **100%** | **10.1055** | **100%** |

## 2. Padding and sequence-length audit

| Item | MiniLM M-T0 | E5 E-T0 |
|---|---:|---:|
| Configured maximum | 512 | 512 |
| Padded batch length | 32 | 34 |
| Real-token mean | 15.546875 | 17.546875 |
| Real-token median | 15.0 | 17.0 |
| Real-token maximum | 32 | 34 |
| Padding policy | dynamic_batch_max | dynamic_batch_max |

## 3. Precision-path audit

### M_T0

| Boundary | Input dtype(s) | Output dtype(s) | BF16 output |
|---|---|---|---:|
| embeddings | non-floating | float32 | False |
| encoder.layer.0 | float32 | float32 | False |
| encoder.layer.1 | float32 | float32 | False |
| encoder.layer.2 | float32 | float32 | False |
| encoder.layer.3 | float32 | float32 | False |
| encoder.layer.4 | float32 | float32 | False |
| encoder.layer.5 | float32 | float32 | False |
| text_encoder_output | non-floating | float32 | False |
| masked_mean_pool | float32 | float32 | False |
| text_projection | float32 | bfloat16 | True |

### E_T0

| Boundary | Input dtype(s) | Output dtype(s) | BF16 output |
|---|---|---|---:|
| embeddings | non-floating | float32 | False |
| encoder.layer.0 | float32 | float32 | False |
| encoder.layer.1 | float32 | float32 | False |
| encoder.layer.2 | float32 | float32 | False |
| encoder.layer.3 | float32 | float32 | False |
| encoder.layer.4 | float32 | float32 | False |
| encoder.layer.5 | float32 | float32 | False |
| encoder.layer.6 | float32 | float32 | False |
| encoder.layer.7 | float32 | float32 | False |
| encoder.layer.8 | float32 | float32 | False |
| encoder.layer.9 | float32 | float32 | False |
| encoder.layer.10 | float32 | float32 | False |
| encoder.layer.11 | float32 | float32 | False |
| text_encoder_output | non-floating | float32 | False |
| masked_mean_pool | float32 | float32 | False |
| text_projection | float32 | bfloat16 | True |

## 4. Side-by-side summary

Frozen ceiling: **9.480 ms Q3**.

