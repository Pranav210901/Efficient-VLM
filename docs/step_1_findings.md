# Step 1 Findings: Completed Encoder and BLF Alignment Matrix

Updated: 2026-07-11

## Status and authoritative sources

This document records the completed first experimental stage of the dissertation: dataset preparation, baseline encoder comparison, BLF ablation, the extended encoder matrix, result correction, and the matched ConvNeXt finalist check.

The authoritative result table is:

```text
results/alignment_matrix_best_clean.csv
```

It contains 80 unique configurations selected consistently by best validation image-to-text Recall@1. The older `results/alignment_matrix_results.csv` must not be used for final reporting because it mixed best-epoch and final-epoch recovery rows.

| Work item | Status |
| --- | --- |
| COCO one-caption training and validation CSVs | Completed and checked |
| Initial baseline and BLF experiments | Completed |
| Extended vision/text encoder matrix | Completed |
| Local, global, and local+global BLF ablations | Completed |
| Best-epoch result reconstruction and alias removal | Completed |
| ConvNeXt 15-epoch finalist comparison | Completed |
| All-caption COCO validation CSV and grouped-positive metric support | Implemented, not yet run across finalists |
| Five-seed DINOv2 and ConvNeXt reliability sweep | Prepared, not yet run |
| Final matched-seed notebook analysis | Prepared, awaiting sweep results |

## Research question

The experiment tests whether lightweight Branch-Level Fusion (BLF) visual branches improve image-text retrieval alignment when added to frozen pretrained vision and text encoders.

Four model setups were compared for each encoder pair:

1. `baseline`: main pretrained vision encoder only.
2. `local`: baseline plus a trainable local-detail BLF branch.
3. `global`: baseline plus a trainable global-context BLF branch.
4. `local_global`: baseline plus both BLF branches.

Only the final three are BLF variants; the baseline is the comparison condition.

## Model architecture

Each image is processed by a frozen pretrained vision encoder. Depending on the ablation, the raw image is also processed by a trainable local branch, global branch, or both. The resulting visual features are combined using `concat_mlp` fusion and projected into a 256-dimensional shared embedding space.

Each caption is processed by a frozen pretrained text encoder and projected into the same shared space. Image and text embeddings are L2-normalised and trained with a symmetric CLIP-style contrastive loss. A learned temperature/logit-scale parameter controls similarity sharpness.

The principal selection metric is validation image-to-text Recall@1 (`i2t_R@1`). Supporting metrics are text-to-image Recall@1, Recall@5/10, mean rank, median rank, and validation contrastive loss.

## Dataset preparation and integrity

The completed matrix used COCO Captions 2017 with one caption selected per image.

| Split | Rows | Unique images | Unique caption strings | Exact duplicate rows | Null cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training (`data/train.csv`) | 118,287 | 118,287 | 113,948 | 0 | 0 |
| Validation (`data/val.csv`) | 5,000 | 5,000 | 4,959 | 0 | 0 |

All referenced training and validation image paths exist, and there is no image overlap between the two splits.

An additional standard all-caption validation file has now been prepared for the next stage:

| File | Rows | Unique images | Unique caption strings | Exact duplicate annotation rows |
| --- | ---: | ---: | ---: | ---: |
| `data/val_all_captions.csv` | 25,014 | 5,000 | 24,794 | 6 |

The six duplicate annotation rows are retained from the source annotations as separate text queries. The new grouped-positive evaluation treats every caption belonging to an image as correct. These five-caption metrics are not part of the completed 80-row matrix and will be reported separately after the matched-seed run.

## Experimental configuration

| Setting | Value |
| --- | --- |
| Image size | 224 × 224 |
| Shared embedding dimension | 256 |
| Fusion hidden dimension | 512 |
| Local branch output | 128 |
| Global branch output | 128 |
| Projection dropout | 0.1 |
| Vision encoder | Pretrained and frozen |
| Text encoder | Pretrained and frozen |
| Optimiser | AdamW |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Maximum epochs in matrix | 5 |
| Mixed precision | Enabled on CUDA |
| Fusion | `concat_mlp` |
| Selection criterion | Highest validation `i2t_R@1` |
| Original random seed | 42 |

### Batch-size history

The four original core baselines were trained before the RTX PRO configuration change with batch size 32. Subsequent extended and BLF runs used batch size 64 with eight data-loader workers. Legacy checkpoints did not save standalone configuration files, so the clean CSV correctly marks seed and batch size as `unknown` instead of inventing metadata.

This batch-size history is a confound for direct claims involving the original core baseline rows. The pending matched-seed experiment uses batch size 64 for every baseline and BLF candidate, removing this confound for the two headline comparisons.

## Completed experiment coverage

Five distinct vision encoders were evaluated:

- `efficientnet_b0`
- `convnext_tiny`
- `convnextv2_tiny`
- `dinov2_vits14`
- `swin_tiny`

Four distinct text encoders were evaluated:

- `all_minilm_l6_v2`
- `bge_small_en`
- `e5_small_v2`
- `distilbert`

The labels `minilm_l6` and `all_minilm_l6_v2` both resolve to `sentence-transformers/all-MiniLM-L6-v2`. They were originally executed as separate labels but are not distinct models. The clean table keeps only the canonical `all_minilm_l6_v2` label.

The scientifically unique matrix therefore contains:

```text
5 vision encoders × 4 text encoders × 4 model setups = 80 configurations
```

This comprises 20 baselines and 60 BLF configurations. On disk there are 100 original labelled matrix executions because the MiniLM alias added 20 redundant executions, plus two separate ConvNeXt finalist executions. Thus, there are 102 completed training executions but 80 unique matrix configurations.

### Why the Section 3.2 output reported 63 jobs

The original extended BLF request contained 75 labelled jobs:

```text
5 vision encoders × 5 text labels × 3 BLF setups = 75
```

Twelve core BLF jobs had already completed in the earlier notebook section:

```text
2 vision encoders × 2 text encoders × 3 BLF setups = 12
```

Because the launcher used `--resume`, those jobs were skipped and the controller reported `75 − 12 = 63` remaining jobs. The number 63 describes work pending in that launch; it is not the number of unique final configurations.

## Result-integrity correction

The original aggregate CSV contained inconsistent checkpoint selection:

- Normal job completion wrote metrics returned from `best.pt`.
- Resume recovery loaded metrics from `final.pt`.
- Retrieval performance commonly declined after the first few epochs, so final-epoch rows could be substantially worse than the run's true best result.

The recovery path has been corrected to read `best.pt`. A reconstruction utility then selected the maximum `i2t_R@1` directly from every `metrics.csv`, removed the duplicate MiniLM alias, and enforced one row per encoder/setup key.

Within the 80 canonical configurations, 47 rows differed from their value in the old aggregate table. Consequently, all rankings, plots, and dissertation claims must be generated from `alignment_matrix_best_clean.csv`.

## Corrected main result

The strongest completed single-seed configuration is:

| Vision encoder | Text encoder | Setup | Best epoch | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 | Validation loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 ViT-S/14 | All-MiniLM-L6-v2 | Local BLF | 3 | **0.1514** | 0.1542 | 0.3872 | 0.4046 | 0.5561 |

The matching DINOv2 + MiniLM baseline reached i2t R@1 = 0.1444, giving a preliminary absolute local-BLF gain of +0.0070. This is a single-seed observation, not yet a statistically supported final claim.

## Highest-ranked configurations

| Rank | Vision encoder | Text encoder | Setup | Best epoch | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | DINOv2 ViT-S/14 | MiniLM | Local | 3 | **0.1514** | 0.1542 | 0.3872 | 0.4046 |
| 2 | DINOv2 ViT-S/14 | MiniLM | Local+global | 2 | 0.1478 | **0.1598** | 0.3808 | 0.3994 |
| 3 | DINOv2 ViT-S/14 | MiniLM | Global | 3 | 0.1454 | 0.1516 | 0.3746 | 0.3942 |
| 4 | DINOv2 ViT-S/14 | MiniLM | Baseline | 2 | 0.1444 | 0.1494 | 0.3806 | 0.3940 |
| 5 | ConvNeXtV2-Tiny | MiniLM | Local+global | 2 | 0.1438 | 0.1506 | 0.3718 | 0.3920 |
| 6 | DINOv2 ViT-S/14 | BGE-Small | Local | 2 | 0.1432 | 0.1510 | 0.3734 | 0.3928 |
| 7 | ConvNeXt-Tiny | MiniLM | Baseline | 2 | 0.1426 | 0.1510 | 0.3682 | 0.3966 |
| 8 | ConvNeXtV2-Tiny | MiniLM | Local | 3 | 0.1420 | 0.1508 | 0.3686 | 0.3992 |
| 9 | ConvNeXt-Tiny | MiniLM | Local+global | 2 | 0.1410 | 0.1512 | 0.3794 | 0.3944 |
| 10 | ConvNeXt-Tiny | MiniLM | Local | 2 | 0.1406 | 0.1548 | 0.3758 | 0.3992 |

The top four positions all use DINOv2 with MiniLM, indicating that this encoder pair is the strongest overall candidate in the corrected matrix. ConvNeXt-Tiny and ConvNeXtV2-Tiny with MiniLM form the next performance tier.

## Encoder-level trends

The following values average over all text encoders and all four setups. They are descriptive summaries rather than controlled pairwise tests.

### Vision encoders

| Vision encoder | Mean i2t R@1 | Mean t2i R@1 |
| --- | ---: | ---: |
| DINOv2 ViT-S/14 | **0.1285** | **0.1364** |
| ConvNeXt-Tiny | 0.1226 | 0.1315 |
| ConvNeXtV2-Tiny | 0.1204 | 0.1315 |
| Swin-Tiny | 0.0975 | 0.1075 |
| EfficientNet-B0 | 0.0799 | 0.0825 |

DINOv2 has the strongest mean retrieval performance. ConvNeXt-Tiny and ConvNeXtV2-Tiny are close behind, while Swin-Tiny and EfficientNet-B0 are materially weaker under this frozen-encoder protocol.

### Text encoders

| Text encoder | Mean i2t R@1 | Mean t2i R@1 |
| --- | ---: | ---: |
| All-MiniLM-L6-v2 | **0.1280** | **0.1369** |
| BGE-Small-en-v1.5 | 0.1156 | 0.1242 |
| E5-Small-v2 | 0.1093 | 0.1174 |
| DistilBERT | 0.0864 | 0.0930 |

MiniLM is the strongest text encoder on average. BGE and E5 provide competitive alternatives, whereas DistilBERT is consistently weaker in this shared-embedding setup.

## BLF results by encoder pair

The table compares each baseline with the strongest of its three BLF variants using i2t R@1.

| Vision encoder | Text encoder | Baseline | Best BLF setup | Best BLF | Absolute change |
| --- | --- | ---: | --- | ---: | ---: |
| EfficientNet-B0 | MiniLM | 0.0916 | Global | 0.1044 | **+0.0128** |
| EfficientNet-B0 | DistilBERT | 0.0520 | Local+global | 0.0644 | **+0.0124** |
| DINOv2 | E5 | 0.1202 | Local+global | 0.1312 | **+0.0110** |
| DINOv2 | BGE | 0.1328 | Local | 0.1432 | **+0.0104** |
| ConvNeXtV2 | MiniLM | 0.1336 | Local+global | 0.1438 | **+0.0102** |
| DINOv2 | MiniLM | 0.1444 | Local | 0.1514 | **+0.0070** |
| Swin | MiniLM | 0.1126 | Global | 0.1192 | **+0.0066** |
| ConvNeXtV2 | E5 | 0.1190 | Local | 0.1256 | **+0.0066** |
| ConvNeXt | E5 | 0.1234 | Local+global | 0.1294 | **+0.0060** |
| ConvNeXt | BGE | 0.1278 | Local | 0.1332 | **+0.0054** |
| Swin | BGE | 0.1020 | Local+global | 0.1074 | **+0.0054** |
| EfficientNet-B0 | BGE | 0.0866 | Local+global | 0.0904 | **+0.0038** |
| ConvNeXtV2 | DistilBERT | 0.0960 | Local | 0.0996 | **+0.0036** |
| ConvNeXt | DistilBERT | 0.0996 | Local+global | 0.1028 | **+0.0032** |
| ConvNeXtV2 | BGE | 0.1268 | Local | 0.1284 | **+0.0016** |
| Swin | E5 | 0.0994 | Local+global | 0.0998 | **+0.0004** |
| ConvNeXt | MiniLM | 0.1426 | Local+global | 0.1410 | **−0.0016** |
| EfficientNet-B0 | E5 | 0.0818 | Global | 0.0796 | **−0.0022** |
| Swin | DistilBERT | 0.0788 | Global | 0.0752 | **−0.0036** |
| DINOv2 | DistilBERT | 0.1104 | Local+global | 0.1062 | **−0.0042** |

The best available BLF variant exceeds baseline in 16 of 20 encoder pairs and is lower in four. The mean best-BLF change is +0.00474 i2t R@1 and the median change is +0.0054.

This comparison is optimistic because it selects the best of three BLF variants for every pair. It is useful for candidate selection but must not be interpreted as an unbiased estimate of the expected BLF effect.

## Average performance by setup

| Setup | Runs | Mean i2t R@1 | Mean t2i R@1 | Mean i2t R@5 | Mean t2i R@5 | Mean validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 20 | 0.1091 | 0.1164 | 0.3063 | 0.3230 | **0.6749** |
| Global | 20 | 0.1085 | 0.1172 | 0.3071 | 0.3207 | 0.7180 |
| Local | 20 | 0.1107 | 0.1183 | 0.3093 | **0.3258** | 0.7085 |
| Local+global | 20 | **0.1111** | **0.1196** | **0.3099** | 0.3252 | 0.7083 |

Local+global has the highest mean Recall@1, but the absolute mean difference from baseline is small: +0.0020 i2t R@1 and +0.0032 t2i R@1. Global-only BLF is slightly below baseline on mean i2t R@1. Validation loss is generally higher for BLF even when retrieval recall improves, so loss and retrieval quality should be discussed separately.

Across the 20 encoder pairs, the overall winning setup by i2t R@1 is local+global for eight pairs, local for six, global for two, and baseline for four. No single BLF design dominates every encoder pair.

## Parameter-efficiency analysis

The BLF branches add a small number of trainable parameters relative to the complete pretrained model.

| Setup | Additional trainable parameters vs baseline | Mean increase in trainable parameters | Mean increase in total parameters |
| --- | ---: | ---: | ---: |
| Local | 73,568 | 5.17% | 0.13% |
| Global | 102,304 | 7.19% | 0.18% |
| Local+global | 175,872 | 12.36% | 0.31% |

This supports the claim that BLF is lightweight in total-model terms. Whether the additional capacity yields a reliable retrieval improvement remains encoder-dependent and requires the pending repeated-seed test.

## Best-epoch and overfitting behaviour

Best checkpoints occur very early:

| Best epoch | Number of canonical runs | Percentage |
| ---: | ---: | ---: |
| 1 | 44 | 55.0% |
| 2 | 32 | 40.0% |
| 3 | 4 | 5.0% |
| 4–5 | 0 | 0.0% |

All 80 canonical runs reached their highest i2t R@1 by epoch 3. This is strong evidence that the trainable projection/fusion layers overfit quickly when the large pretrained encoders are frozen. Longer training is therefore not automatically beneficial.

The training pipeline has since been updated with configurable early stopping, best-checkpoint recovery, deterministic settings for repeated experiments, logit-scale clamping, modern AMP calls, and per-run configuration/summary files. The matched-seed sweep uses a maximum of five epochs and stops after two consecutive non-improving epochs.

## ConvNeXt finalist comparison

A longer 15-epoch check compared ConvNeXt-Tiny + MiniLM baseline against local+global BLF using the same seed and best-checkpoint selection.

| Setup | Best epoch | i2t R@1 | t2i R@1 | Mean R@1 | i2t R@5 | t2i R@5 | Validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2 | **0.1426** | 0.1510 | **0.1468** | 0.3682 | **0.3966** | 0.5676 |
| Local+global BLF | 2 | 0.1410 | **0.1512** | 0.1461 | **0.3794** | 0.3944 | **0.5627** |

The baseline is 0.0016 higher on the primary i2t R@1 metric, while BLF is 0.0002 higher on t2i R@1 and 0.0112 higher on i2t R@5. These small, mixed differences do not support a claim that local+global BLF clearly improves ConvNeXt + MiniLM.

Both models peak at epoch 2 and deteriorate substantially later, reinforcing the early-overfitting finding. The 15-epoch experiment should be presented as a best-checkpoint study, not as evidence that long training helps.

## Interpretation suitable for the dissertation

The completed matrix supports the following cautious findings:

1. Encoder choice has a larger effect than BLF choice. DINOv2 and MiniLM are the strongest vision and text components respectively.
2. DINOv2 + MiniLM + local BLF is the strongest observed single run.
3. At least one BLF variant outperforms baseline in 16 of 20 encoder pairs, suggesting BLF can be useful, but not universally.
4. The most effective BLF design depends on the encoder pair; local+global is best most often, but local and global variants also win, and baseline remains best for four pairs.
5. Average BLF gains are modest compared with the differences between encoders.
6. BLF adds only 0.13–0.31% to total model parameters, making it lightweight even when gains are small.
7. The ConvNeXt finalist does not reproduce the original apparent local+global advantage on the primary metric.
8. Rapid overfitting is a consistent property of the current frozen-encoder training regime.

## Claims that should not yet be made

The completed evidence does not justify saying that:

- BLF always improves image-text alignment.
- Local+global BLF is universally the best setup.
- The observed DINOv2 local-BLF gain is statistically significant.
- The completed one-caption retrieval numbers are directly comparable to published five-caption COCO results.
- Fifteen epochs improve model quality.
- Differences from rows trained with different batch sizes are solely caused by BLF.

## Limitations

1. Most matrix configurations were run with a single seed, so random training variation is not quantified.
2. The initial baseline and later BLF stages did not all use the same batch size.
3. The completed matrix uses one caption per validation image rather than the standard five-caption COCO protocol.
4. Selecting the best of three BLF variants inflates the apparent best-BLF improvement.
5. Eighty configurations create a multiple-comparisons risk: the best observed run may partly reflect search variance.
6. Vision and text encoders are frozen, so conclusions apply to lightweight adapter/projection training rather than full fine-tuning.
7. Validation loss is a batch-level contrastive quantity and is not perfectly aligned with full-dataset retrieval ranking.
8. The experiment currently covers COCO only; generalisation to another dataset has not been tested.

## Completed engineering and reproducibility work

The following supporting work is complete:

- Resume-capable multi-GPU matrix execution with one process per GPU.
- Best, latest, and final checkpoint handling.
- Correction of recovered result rows to use `best.pt`.
- Reconstruction of a duplicate-free best-epoch result table.
- Removal of MiniLM alias duplication from the scientific matrix.
- Multi-positive retrieval metrics for all-caption COCO evaluation.
- Preparation of `data/val_all_captions.csv`.
- Deterministic repeated-seed configuration.
- Early stopping and logit-scale clamping.
- Five-seed, four-GPU sweep controller and resumable job layout.
- Automatic result summaries, paired seed differences, uncertainty intervals, and plots.
- Notebook cells for launching the sweep on an allocated GPU node and documenting final results.
- Nine passing workflow and metric tests.
- Pinned direct dependency record and updated GPU runbook.

## Pending Step 3: matched-seed reliability experiment

The prepared next experiment repeats the two most important comparisons across seeds 42–46:

| Comparison | Candidate setups | Seeds | Runs |
| --- | --- | ---: | ---: |
| DINOv2 + MiniLM | Baseline vs local BLF | 5 | 10 |
| ConvNeXt-Tiny + MiniLM | Baseline vs local+global BLF | 5 | 10 |

The 20 jobs will run in five waves of four on four RTX PRO 6000 GPUs. Every run uses batch size 64, deterministic settings, best-checkpoint selection, and early stopping. Each best checkpoint will then be evaluated using the all-caption COCO protocol.

This stage will determine whether the single-run BLF differences are repeatable rather than random.

## Pending Step 4: final analysis and reporting

After the sweep, the notebook will:

- Display every seed and checkpoint.
- Calculate means, standard deviations, standard errors, and 95% t-intervals.
- Calculate seed-paired BLF-minus-baseline changes.
- Count BLF wins, ties, and losses.
- Produce a cautious written conclusion for each comparison.
- Save a matched-seed comparison figure and summary CSVs.

The final dissertation claim should be based on these repeated-seed and five-caption results. If the interval for a BLF-minus-baseline difference crosses zero, the outcome should be reported as inconclusive rather than beneficial.

## Key files for write-up

| File | Purpose |
| --- | --- |
| `results/alignment_matrix_best_clean.csv` | Authoritative 80-row corrected matrix |
| `results/finalist_convnext_tiny_minilm_l6_15ep.csv` | Matched ConvNeXt finalist comparison |
| `notebooks/01_experiment_workflow.ipynb` | Experiment execution, visualisation, and final reporting workflow |
| `data/train.csv` / `data/val.csv` | Completed one-caption matrix data |
| `data/val_all_captions.csv` | Standard multi-caption validation data for the next stage |
| `configs/seed_sweep.yaml` | Matched-seed experiment definition |
| `docs/gpu_runbook.md` | GPU allocation, monitoring, resume, and summary instructions |
