# Phase 1 multi-task findings

Results use best checkpoints and frozen pretrained backbones.

## Best configuration per task

| config_id                                   | vision_encoder   | text_encoder     | variant   | task              | metric        |     value |   best_epoch | checkpoint                                                                                                           | mode   |
|:--------------------------------------------|:-----------------|:-----------------|:----------|:------------------|:--------------|----------:|-------------:|:---------------------------------------------------------------------------------------------------------------------|:-------|
| convnextv2_tiny__all_minilm_l6_v2__baseline | convnextv2_tiny  | all_minilm_l6_v2 | baseline  | cifar100_zeroshot | top1_accuracy | 0.2951    |            2 | /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/checkpoints/convnextv2_tiny_all_minilm_l6_v2_baseline/best.pt | full   |
| dinov2_vits14__all_minilm_l6_v2__local      | dinov2_vits14    | all_minilm_l6_v2 | local     | coco_retrieval    | coco5_i2t_R@1 | 0.1952    |            3 | /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/checkpoints/dinov2_vits14_all_minilm_l6_v2_local/best.pt      | full   |
| dinov2_vits14__all_minilm_l6_v2__local      | dinov2_vits14    | all_minilm_l6_v2 | local     | coco_retrieval    | coco5_t2i_R@1 | 0.16047   |            3 | /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/checkpoints/dinov2_vits14_all_minilm_l6_v2_local/best.pt      | full   |
| dinov2_vits14__bge_small_en__baseline       | dinov2_vits14    | bge_small_en     | baseline  | eurosat_zeroshot  | top1_accuracy | 0.222889  |            2 | /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/checkpoints/dinov2_vits14_bge_small_en_baseline/best.pt       | full   |
| dinov2_vits14__bge_small_en__baseline       | dinov2_vits14    | bge_small_en     | baseline  | pets_zeroshot     | top1_accuracy | 0.0686836 |            2 | /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/checkpoints/dinov2_vits14_bge_small_en_baseline/best.pt       | full   |

## Matched BLF versus baseline

| comparison                     | metric_label             |   baseline_value |   blf_value |   delta_percentage_points | outcome   |
|:-------------------------------|:-------------------------|-----------------:|------------:|--------------------------:|:----------|
| dinov2_minilm_local            | COCO i2t R@1             |        0.1792    |   0.1952    |                  1.6      | blf_win   |
| dinov2_minilm_local            | COCO t2i R@1             |        0.154354  |   0.16047   |                  0.611658 | blf_win   |
| dinov2_minilm_local            | CIFAR-100 top-1 accuracy |        0.2549    |   0.2604    |                  0.549999 | blf_win   |
| dinov2_minilm_local            | Pets top-1 accuracy      |        0.0621423 |   0.0564186 |                 -0.572363 | blf_loss  |
| dinov2_minilm_local            | EuroSAT top-1 accuracy   |        0.200074  |   0.194519  |                 -0.555556 | blf_loss  |
| convnextv2_minilm_local_global | COCO i2t R@1             |        0.17      |   0.183     |                  1.3      | blf_win   |
| convnextv2_minilm_local_global | COCO t2i R@1             |        0.150236  |   0.154793  |                  0.455746 | blf_win   |
| convnextv2_minilm_local_global | CIFAR-100 top-1 accuracy |        0.2951    |   0.2901    |                 -0.5      | blf_loss  |
| convnextv2_minilm_local_global | Pets top-1 accuracy      |        0.0444263 |   0.037885  |                 -0.654129 | blf_loss  |
| convnextv2_minilm_local_global | EuroSAT top-1 accuracy   |        0.217889  |   0.192185  |                 -2.57037  | blf_loss  |
| dinov2_bge_local_exploratory   | COCO i2t R@1             |        0.1736    |   0.1824    |                  0.88     | blf_win   |
| dinov2_bge_local_exploratory   | COCO t2i R@1             |        0.150676  |   0.154114  |                  0.343807 | blf_win   |
| dinov2_bge_local_exploratory   | CIFAR-100 top-1 accuracy |        0.2645    |   0.2608    |                 -0.369999 | blf_loss  |
| dinov2_bge_local_exploratory   | Pets top-1 accuracy      |        0.0686836 |   0.0585991 |                 -1.00845  | blf_loss  |
| dinov2_bge_local_exploratory   | EuroSAT top-1 accuracy   |        0.222889  |   0.217704  |                 -0.518519 | blf_loss  |

### Win/tie/loss summary

| comparison                     | baseline_config_id                          | blf_config_id                                   |   metrics_compared |   wins |   ties |   losses |
|:-------------------------------|:--------------------------------------------|:------------------------------------------------|-------------------:|-------:|-------:|---------:|
| dinov2_minilm_local            | dinov2_vits14__all_minilm_l6_v2__baseline   | dinov2_vits14__all_minilm_l6_v2__local          |                  5 |      3 |      0 |        2 |
| convnextv2_minilm_local_global | convnextv2_tiny__all_minilm_l6_v2__baseline | convnextv2_tiny__all_minilm_l6_v2__local_global |                  5 |      2 |      0 |        3 |
| dinov2_bge_local_exploratory   | dinov2_vits14__bge_small_en__baseline       | dinov2_vits14__bge_small_en__local              |                  5 |      2 |      0 |        3 |

Detailed output: `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/phase1_multitask/blf_vs_baseline_multitask.csv`

## Limitations

- Tasks use different official metrics and are not raw-averaged.
- CIFAR-100 and Oxford-IIIT Pet use test splits; torchvision exposes EuroSAT only as the complete dataset.
- Zero-shot classes use a fixed three-template prompt ensemble with no task-specific fitting.
- Matched BLF changes are point estimates unless paired uncertainty is computed from sample-level predictions.
- Oracle and specialist claims require Phase 1.5 sample-level analysis.
