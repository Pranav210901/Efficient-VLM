# Phase 2 Report

Status: **COMPLETE**

All required result sections are present.

## Multi-seed statistics

| method                          | task      | metric                    |   count |          mean |           std |   standard_error |      ci95_low |     ci95_high |   mean_difference |   wins |   losses |   ties |
|:--------------------------------|:----------|:--------------------------|--------:|--------------:|--------------:|-----------------:|--------------:|--------------:|------------------:|-------:|---------:|-------:|
| best_bridge                     | multitask | latency_ms                |       5 |   3.54972     |   1.95497     |      0.874289    |   1.83611     |   5.26333     |       nan         |    nan |      nan |    nan |
| best_bridge                     | multitask | mean_sample_utility       |       5 |   0.181472    |   0.00462988  |      0.00207055  |   0.177414    |   0.185531    |       nan         |    nan |      nan |    nan |
| best_bridge                     | multitask | peak_memory_bytes         |       5 |   8.09461e+08 |   0           |      0           |   8.09461e+08 |   8.09461e+08 |       nan         |    nan |      nan |    nan |
| dense_teacher                   | multitask | gating_entropy            |       5 |   1.29952     |   0.030934    |      0.0138341   |   1.2724      |   1.32663     |       nan         |    nan |      nan |    nan |
| dense_teacher                   | multitask | latency_ms                |       5 |   0.000821942 |   0.000743358 |      0.00033244  |   0.000170359 |   0.00147352  |       nan         |    nan |      nan |    nan |
| dense_teacher                   | multitask | mean_sample_utility       |       5 |   0.167114    |   0.00149209  |      0.000667282 |   0.165806    |   0.168422    |       nan         |    nan |      nan |    nan |
| dense_teacher                   | multitask | path_utilisation_variance |       5 |   0.00887809  |   0.00296206  |      0.00132467  |   0.00628173  |   0.0114744   |       nan         |    nan |      nan |    nan |
| dense_teacher                   | multitask | peak_memory_bytes         |       5 |   0           |   0           |      0           |   0           |   0           |       nan         |    nan |      nan |    nan |
| dense_teacher_minus_best_bridge | multitask | mean_sample_utility       |     nan | nan           | nan           |    nan           | nan           | nan           |        -0.0143584 |      0 |        5 |      0 |

## Pareto frontier

|   direction |   candidate_depth | metric                |   latency_ms | task             | method                                              | split_role   |   performance |   peak_memory_bytes |   throughput_per_second | offline_index_cost               | online_query_cost        | pareto_optimal   |
|------------:|------------------:|:----------------------|-------------:|:-----------------|:----------------------------------------------------|:-------------|--------------:|--------------------:|------------------------:|:---------------------------------|:-------------------------|:-----------------|
|         nan |                64 | additional_latency_ms |    0.0850325 | eurosat_zeroshot | optional_path__convnext_tiny__e5_small_v2__baseline | development  |      0.874815 |          1435940352 |                11760.2  | reused_phase1_dual_encoder_cache | 64 cross-attention pairs | True             |
|         nan |                64 | additional_latency_ms |    0.114774  | eurosat_zeroshot | optional_path__swin_tiny__e5_small_v2__baseline     | development  |      0.891111 |          1557580288 |                 8712.74 | reused_phase1_dual_encoder_cache | 64 cross-attention pairs | True             |
|         nan |                64 | additional_latency_ms |    0.299223  | eurosat_zeroshot | layers_1                                            | development  |      0.911358 |           809460736 |                 3341.99 | reused_phase1_dual_encoder_cache | 64 cross-attention pairs | True             |
|         nan |                64 | additional_latency_ms |    0.325095  | eurosat_zeroshot | random_negatives                                    | development  |      0.924198 |           809460736 |                 3076.02 | reused_phase1_dual_encoder_cache | 64 cross-attention pairs | True             |
