# Paired repeated-measurement latency diagnostic: M_T1 vs M_T0

Diagnostic only. No retraining, checkpoint/epoch reselection, automatic gate decision, or ceiling modification was performed.

- Slurm job ID: `2288035`
- Node: `aisurrey38.surrey.ac.uk`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Frozen ceiling (context only): `9.480 ms Q3`
- Protocol: batch 64, 20 warm-up iterations and 100 timed iterations per independent profile
- Repetitions: 10 per seed per arm; 60 total profiles
- Ordering: deterministically randomized within every seed/repetition pair
- CI: 10,000-replicate percentile bootstrap CI for the mean Q3

## Summary

| Comparison | Seed | n | Median Q3 (ms) | Mean Q3 (ms) | SD (ms) | Mean Q3 95% CI (ms) |
|---|---:|---:|---:|---:|---:|---:|
| M_T0 | 42 | 10 | 8.746039 | 8.780511 | 0.110923 | [8.744532, 8.851063] |
| M_T0 | 43 | 10 | 8.748921 | 8.748624 | 0.001770 | [8.747552, 8.749644] |
| M_T0 | 44 | 10 | 8.758653 | 8.758731 | 0.001232 | [8.758033, 8.759482] |
| M_T0 | pooled | 30 | 8.749435 | 8.762622 | 0.063269 | [8.749576, 8.786770] |
| M_T1 | 42 | 10 | 8.995463 | 9.011452 | 0.054312 | [8.992872, 9.046453] |
| M_T1 | 43 | 10 | 9.000660 | 9.000295 | 0.002221 | [8.998962, 9.001597] |
| M_T1 | 44 | 10 | 8.998147 | 8.997939 | 0.001964 | [8.996767, 8.999088] |
| M_T1 | pooled | 30 | 8.998041 | 9.003229 | 0.030889 | [8.996712, 9.015173] |
| M_T1_minus_M_T0_paired | 42 | 10 | 0.249561 | 0.230942 | 0.056741 | [0.194546, 0.250041] |
| M_T1_minus_M_T0_paired | 43 | 10 | 0.251530 | 0.251671 | 0.002368 | [0.250216, 0.253016] |
| M_T1_minus_M_T0_paired | 44 | 10 | 0.239124 | 0.239208 | 0.002489 | [0.237704, 0.240608] |
| M_T1_minus_M_T0_paired | pooled | 30 | 0.248450 | 0.240607 | 0.032832 | [0.227679, 0.248148] |

## Frozen checkpoint identities

- M_T0 seed 42: epoch 14; `checkpoints/token_aggregator_scale_training/C4_d256_h8_b2_ff512/sensitivity/distill_strength_1p0__seed_42/epoch_14.pt`; SHA-256 `e65a7ed575f09582b95f26691692e59a04a840198dcd4c39d108289569282235`
- M_T0 seed 43: epoch 22; `checkpoints/token_aggregator_scale_training/C4_d256_h8_b2_ff512/sensitivity/distill_strength_1p0__seed_43/epoch_22.pt`; SHA-256 `508cbce6e2393c034580127a6a4cd21f390b8fc2af40086b99229c3aad244187`
- M_T0 seed 44: epoch 20; `checkpoints/token_aggregator_scale_training/C4_d256_h8_b2_ff512/sensitivity/distill_strength_1p0__seed_44/epoch_20.pt`; SHA-256 `42b449db1497d55c6dca3c154c86cb64ce615093f556347379de21a463d2f19a`
- M_T1 seed 42: epoch 22; `checkpoints/text_aggregation_study/M_T1/sensitivity/distill_strength_1p0__seed_42/epoch_22.pt`; SHA-256 `15edf5155a034f474cf3be44c797e578a15af3d53463df96b0f19d96a689d4be`
- M_T1 seed 43: epoch 22; `checkpoints/text_aggregation_study/M_T1/sensitivity/distill_strength_1p0__seed_43/epoch_22.pt`; SHA-256 `a6b273972fb58e310dcd38ed24a648c7d0de568503090e480b9750190bd7bed2`
- M_T1 seed 44: epoch 20; `checkpoints/text_aggregation_study/M_T1/sensitivity/distill_strength_1p0__seed_44/epoch_20.pt`; SHA-256 `b8872c31deaae3b0ecf71c0553f25d3cfbc29f894c30dbeb60e1c9c246a0fb0e`

## Artifacts

- Raw profiles: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/raw_profiles.csv`
- Summary statistics: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/summary_statistics.csv`
- Paired values: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/paired_differences.csv`
- Machine-readable report: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/report.json`
- This summary: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/text_aggregation_study/diagnostics/m_t1_repeated_latency/report.md`

No PASS/FAIL verdict is emitted by this diagnostic.
