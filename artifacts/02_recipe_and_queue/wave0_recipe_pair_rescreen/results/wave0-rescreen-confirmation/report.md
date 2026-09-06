# Wave 0 top-three pair confirmation

DINOv3 ViT-S/16 + MiniLM remains permanently locked for the main study. E5 is not adopted. The decision preserves continuity with the MiniLM queue curve, screen, teacher caches, and all 31 claim-bearing runs; Wave 1 remains blocked.

E5 38.883% (SD 0.012pp), BGE 38.262% (0.094pp), and MiniLM 37.324% (0.282pp) replicate the ranking without observed overlap. The queue-selected MiniLM pair is last by 1.56pp. MiniLM is also the most variable pair; this is reported as an observation, not a mechanism claim.

E5 demonstrates that the correction generalises beyond MiniLM. One E5 best-available-configuration point is deferred to the end-of-project efficiency frontier and is outside the main study.

| experiment_id                   | text_encoder     |   mean_3seed |   std_3seed |   minimum |   maximum |   rank |
|:--------------------------------|:-----------------|-------------:|------------:|----------:|----------:|-------:|
| dinov3_vits16__e5_small_v2      | e5_small_v2      |     0.38883  | 0.000121953 |  0.38875  |  0.38897  |      1 |
| dinov3_vits16__bge_small_en     | bge_small_en     |     0.382617 | 0.000944918 |  0.38159  |  0.38345  |      2 |
| dinov3_vits16__all_minilm_l6_v2 | all_minilm_l6_v2 |     0.373238 | 0.00282365  |  0.370332 |  0.375972 |      3 |
