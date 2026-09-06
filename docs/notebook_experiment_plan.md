# BLF Vision-Language Notebook Experiment Plan

> **Superseded workflow note (2026-07-10):** The completion statuses and aggregate-table assumptions below predate the corrected best-epoch rebuild. Use `results/alignment_matrix_best_clean.csv`, `configs/seed_sweep.yaml`, and `docs/gpu_runbook.md` for current work.

Generated: 2026-07-03

This document describes what `alignment_vlm/notebooks/01_experiment_workflow.ipynb` is set up to do, what has already been completed, what is still pending, and which evaluation metrics will be recorded.

## Current Notebook Configuration

| Setting | Current value |
| --- | --- |
| Active config | `configs/coco_blf_rtxpro.yaml` |
| Dataset | COCO Captions prepared as `data/train.csv` and `data/val.csv` |
| Train CSV | `data/train.csv` |
| Validation CSV | `data/val.csv` |
| Vision encoders | `efficientnet_b0`, `convnext_tiny` |
| Text encoders | `minilm_l6`, `distilbert` |
| BLF variants | local only, global only, local + global |
| Fusion type | `concat_mlp` |
| Epochs per run | 5 |
| RTX PRO config batch size | 64 |
| RTX PRO config dataloader workers | 8 |
| GPU selection | `GPU_IDS = ["0"]` |
| Resume behavior | Enabled with `--resume` |
| Main results file | `results/alignment_matrix_results.csv` |
| Checkpoint root | `checkpoints/` |

Important comparison note: the baseline rows currently in `results/alignment_matrix_results.csv` were completed before the RTX PRO config change and used the earlier batch size of 32. The current BLF runs use `configs/coco_blf_rtxpro.yaml`, which sets batch size to 64. This makes the run faster, but final claims comparing baseline vs BLF should mention the batch-size difference or rerun baseline under the same RTX PRO config.

## What The Notebook Does

| Notebook section | Cell purpose | Status |
| --- | --- | --- |
| 0. Prepare COCO Captions | Checks for `data/train.csv` and `data/val.csv`; prepares them from COCO annotation JSON if missing. | Completed |
| 1. Train Baseline Encoder Pairs | Runs `scripts/run_encoder_matrix.py --mode baseline` over the 2 vision encoders and 2 text encoders with BLF disabled. | Completed on disk |
| 2. Evaluate Baseline Results | Reads `results/alignment_matrix_results.csv` and displays baseline rows. | Can be rerun anytime |
| 3. Add BLF Branches and Retry Pairings | Runs `scripts/run_encoder_matrix.py --mode blf` over the same encoder grid with local/global BLF variants. | In progress / partially complete |
| 4. Alignment Matrix | Builds a pivot table and heatmap, currently using `i2t_R@1`. | Pending final BLF results |
| Optional: Extract Embeddings | Extracts embeddings from `checkpoints/efficientnet_b0_minilm_l6_local_global/best.pt`. | Pending local+global checkpoint |

## Model Components

| Component | Choices / behavior |
| --- | --- |
| Vision encoder | `efficientnet_b0` or `convnext_tiny`; pretrained and frozen by config. |
| Text encoder | `minilm_l6` or `distilbert`; pretrained and frozen by config. |
| BLF local branch | Trainable neural network branch using small convolutional/detail features from the image. |
| BLF global branch | Trainable neural network branch using coarse/global image context. |
| Fusion module | Combines the main vision encoder feature with optional local/global BLF features. |
| Projection heads | Project image and text features into a shared embedding space. |
| Loss | CLIP-style contrastive loss over image-text pairs in the batch. |

## Evaluation Metric Definitions

| Metric | Direction | Meaning | Better value |
| --- | --- | --- | --- |
| `val_loss` | Validation | Average validation contrastive loss. | Lower |
| `i2t_R@1` | Image to text | Fraction of image queries where the correct caption is ranked first. | Higher |
| `i2t_R@5` | Image to text | Fraction of image queries where the correct caption is in the top 5. | Higher |
| `i2t_R@10` | Image to text | Fraction of image queries where the correct caption is in the top 10. | Higher |
| `t2i_R@1` | Text to image | Fraction of caption queries where the correct image is ranked first. | Higher |
| `t2i_R@5` | Text to image | Fraction of caption queries where the correct image is in the top 5. | Higher |
| `t2i_R@10` | Text to image | Fraction of caption queries where the correct image is in the top 10. | Higher |
| `mean_rank_i2t` | Image to text | Average rank of the correct caption for each image query. | Lower |
| `median_rank_i2t` | Image to text | Median rank of the correct caption for image queries. | Lower |
| `mean_rank_t2i` | Text to image | Average rank of the correct image for each caption query. | Lower |
| `median_rank_t2i` | Text to image | Median rank of the correct image for caption queries. | Lower |
| `params_total` | Model | Total parameter count. | Context only |
| `params_trainable` | Model | Parameters updated during training. | Context only |

## Completed Evaluation Metrics

| Status | Vision encoder | Text encoder | BLF variant | Val loss | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Mean rank I2T | Median rank I2T | Mean rank T2I | Median rank T2I |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Completed | efficientnet_b0 | minilm_l6 | baseline | 0.5057 | 0.0812 | 0.2616 | 0.3922 | 0.0890 | 0.2678 | 0.3968 | 57.0970 | 17.0000 | 54.1880 | 16.0000 |
| Completed | efficientnet_b0 | distilbert | baseline | 0.6554 | 0.0520 | 0.1762 | 0.2766 | 0.0542 | 0.1940 | 0.2962 | 77.4220 | 29.0000 | 70.0142 | 26.0000 |
| Completed | convnext_tiny | minilm_l6 | baseline | 0.3552 | 0.1210 | 0.3368 | 0.4878 | 0.1362 | 0.3602 | 0.5106 | 35.5696 | 11.0000 | 37.1742 | 10.0000 |
| Completed | convnext_tiny | distilbert | baseline | 0.4431 | 0.0996 | 0.2680 | 0.3984 | 0.0996 | 0.2908 | 0.4172 | 44.4098 | 16.0000 | 46.4394 | 15.0000 |
| Completed | efficientnet_b0 | minilm_l6 | local | 0.7853 | 0.0956 | 0.2870 | 0.4152 | 0.1034 | 0.2984 | 0.4238 | 54.9944 | 15.0000 | 51.2952 | 15.0000 |
| Completed | efficientnet_b0 | distilbert | local | 1.0197 | 0.0600 | 0.1936 | 0.2996 | 0.0620 | 0.2040 | 0.3052 | 76.4098 | 27.0000 | 72.0612 | 24.0000 |

## Pending Evaluation Metrics

These rows are expected from the BLF cell but were not yet present in `results/alignment_matrix_results.csv` when this document was generated.

| Status | Vision encoder | Text encoder | BLF variant | Val loss | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Mean rank I2T | Median rank I2T | Mean rank T2I | Median rank T2I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | convnext_tiny | minilm_l6 | local |  |  |  |  |  |  |  |  |  |  |  |
| Pending | convnext_tiny | distilbert | local |  |  |  |  |  |  |  |  |  |  |  |
| Pending | efficientnet_b0 | minilm_l6 | global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | efficientnet_b0 | distilbert | global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | convnext_tiny | minilm_l6 | global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | convnext_tiny | distilbert | global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | efficientnet_b0 | minilm_l6 | local+global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | efficientnet_b0 | distilbert | local+global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | convnext_tiny | minilm_l6 | local+global |  |  |  |  |  |  |  |  |  |  |  |
| Pending | convnext_tiny | distilbert | local+global |  |  |  |  |  |  |  |  |  |  |  |

## Parameter Counts For Completed Runs

| Status | Vision encoder | Text encoder | BLF variant | Total params | Trainable params |
| --- | --- | --- | --- | ---: | ---: |
| Completed | efficientnet_b0 | minilm_l6 | baseline | 28,362,749 | 1,641,985 |
| Completed | efficientnet_b0 | distilbert | baseline | 72,209,021 | 1,838,593 |
| Completed | convnext_tiny | minilm_l6 | baseline | 51,913,185 | 1,379,841 |
| Completed | convnext_tiny | distilbert | baseline | 95,759,457 | 1,576,449 |
| Completed | efficientnet_b0 | minilm_l6 | local | 28,436,317 | 1,715,553 |
| Completed | efficientnet_b0 | distilbert | local | 72,282,589 | 1,912,161 |

## Full BLF Job List

The BLF cell runs 12 jobs in this order:

| Order | Vision encoder | Text encoder | BLF variant | Checkpoint directory |
| ---: | --- | --- | --- | --- |
| 1 | efficientnet_b0 | minilm_l6 | local | `checkpoints/efficientnet_b0_minilm_l6_local` |
| 2 | efficientnet_b0 | distilbert | local | `checkpoints/efficientnet_b0_distilbert_local` |
| 3 | convnext_tiny | minilm_l6 | local | `checkpoints/convnext_tiny_minilm_l6_local` |
| 4 | convnext_tiny | distilbert | local | `checkpoints/convnext_tiny_distilbert_local` |
| 5 | efficientnet_b0 | minilm_l6 | global | `checkpoints/efficientnet_b0_minilm_l6_global` |
| 6 | efficientnet_b0 | distilbert | global | `checkpoints/efficientnet_b0_distilbert_global` |
| 7 | convnext_tiny | minilm_l6 | global | `checkpoints/convnext_tiny_minilm_l6_global` |
| 8 | convnext_tiny | distilbert | global | `checkpoints/convnext_tiny_distilbert_global` |
| 9 | efficientnet_b0 | minilm_l6 | local+global | `checkpoints/efficientnet_b0_minilm_l6_local_global` |
| 10 | efficientnet_b0 | distilbert | local+global | `checkpoints/efficientnet_b0_distilbert_local_global` |
| 11 | convnext_tiny | minilm_l6 | local+global | `checkpoints/convnext_tiny_minilm_l6_local_global` |
| 12 | convnext_tiny | distilbert | local+global | `checkpoints/convnext_tiny_distilbert_local_global` |

## Resume And Output Behavior

| Behavior | Detail |
| --- | --- |
| Resume flag | Notebook commands include `--resume`. |
| Completed job skip | If a job has `final.pt` and a result row already exists, it is skipped. |
| Completed row recovery | If `final.pt` exists but the result row is missing, the runner recovers the row from the checkpoint metrics. |
| Partial job resume | Training resumes from `latest.pt` where available, otherwise from `best.pt`. |
| Resume granularity | Epoch-level. If a job stops mid-epoch, it resumes from the previous saved epoch/checkpoint. |
| Per-run metrics | Each checkpoint directory contains `metrics.csv`. |
| Best checkpoint | `best.pt` is saved when validation `i2t_R@1` improves. |
| Final checkpoint | `final.pt` is saved after the configured final epoch. |
| Latest checkpoint | New runs save `latest.pt` after each completed epoch. |

## Planned Final Analysis

After all pending rows are filled, the notebook should support these comparisons:

1. Baseline vs local BLF for each vision/text pair.
2. Baseline vs global BLF for each vision/text pair.
3. Baseline vs local+global BLF for each vision/text pair.
4. Best vision encoder for each text encoder.
5. Best text encoder for each vision encoder.
6. Overall best configuration by `i2t_R@1`, with supporting checks from `t2i_R@1`, Recall@5/10, rank metrics, and validation loss.

## Files Used Or Produced

| File or directory | Purpose |
| --- | --- |
| `notebooks/01_experiment_workflow.ipynb` | Main interactive workflow. |
| `scripts/run_encoder_matrix.py` | Launches baseline/BLF experiment grid, now with GPU and resume support. |
| `configs/coco_blf_rtxpro.yaml` | Active fast RTX PRO config. |
| `configs/coco_blf.yaml` | COCO dataset config inherited by the RTX PRO config. |
| `configs/experiment_blf.yaml` | Base experiment config for BLF settings. |
| `results/alignment_matrix_results.csv` | Main aggregate results table. |
| `checkpoints/*/metrics.csv` | Per-epoch metrics for each run. |
| `checkpoints/*/best.pt` | Best validation checkpoint. |
| `checkpoints/*/final.pt` | Final checkpoint for completed runs. |
| `checkpoints/*/latest.pt` | Latest checkpoint for resumable new runs. |
