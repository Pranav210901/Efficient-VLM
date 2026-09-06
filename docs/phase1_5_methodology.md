# Phase 1.5 methodology

Phase 1.5 analyses whether frozen pretrained vision and text encoders make the same errors, where they specialise, and which small, diverse set is justified for later investigation. It does not train an ensemble and does not implement Phase 2.

## Complementarity

Retrieval complementarity compares whether each pair succeeds for the same query at the stated recall threshold. Classification complementarity is calculated independently for CIFAR-100, Oxford-IIIT Pets, and EuroSAT. It reports both-correct, only-A-correct, only-B-correct, both-wrong, correct-set Jaccard similarity, prediction agreement, and confidence/margin correlations. Per-class tables are retained because pooling unrelated classes or datasets would hide meaningful specialisation.

## Oracle interpretation

The classification oracle marks a sample correct when any expert in the subset is correct. Pair, triple, full-pool, selected-vision-pool, and selected-text-pool results are diagnostic upper bounds. They are not deployable results and are not evidence that a learned selector can attain the same score.

## Cross-task comparison

COCO image-to-text Recall@1, COCO text-to-image Recall@1, and the three classification accuracies remain available as raw metrics. They are never averaged directly. Cross-task scores default to `relative_to_task_best`, with within-task min-max and z-score alternatives also exported. Per-task ranks, correlations, near-best counts, task variation, and latency-adjusted scores describe breadth and specialisation.

## BLF reliability

Seed-sweep names are converted to canonical configuration IDs before joining. A BLF variant is `supported` only when its matched mean delta exceeds the configured minimum, the lower confidence bound exceeds zero, at least four of five matched seeds improve, and the minimum run count is met. Confidence intervals crossing zero are `inconclusive`; they are not treated as confirmed improvements.

## Diversity-constrained selection

The weighted score combines multi-task performance, unique wins, oracle marginal contribution, specialisation, latency, memory, seed evidence, architecture diversity, and token-interface readiness. Hard constraints are applied separately: no more than two primary paths per vision or text encoder, at least two vision families and two text encoders, a low-cost path, and a unique-error path. An inconclusive BLF cannot duplicate its matched baseline in the primary shortlist. The canonical selection artifact is `results/phase15/expert_selection/selected_experts.yaml`.

## Hard negatives

Existing COCO-validation and classification-test errors are diagnostic only and explicitly non-trainable. New Phase 2 training candidates must be mined from COCO train2017, deterministic partitions of CIFAR-100 train and Pets trainval, and the stored EuroSAT train split. Valid captions from the same image and normalised duplicate captions are excluded as retrieval negatives. Winoground and SugarCrepe remain evaluation-only unless a separate training source is configured.
