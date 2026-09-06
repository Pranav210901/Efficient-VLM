# Efficiency frontier

This phase evaluates the dissertation question under a fixed inference-budget
framing. Flickr30k Karpathy test is primary because it is the only available
evaluation set not used in project selection. COCO validation is supplementary
and contaminated by earlier model selection; the project has no untouched COCO
test split.

The roster contains the locked MiniLM student, the separately labelled E5
"best available configuration", the historical v4-recipe student, and locally
evaluated MobileCLIP2-S0, SigLIP2-B/32 and OpenCLIP ViT-B/32 references.

The phase is intentionally split:

1. Complete only the missing historical-v4 seeds 43 and 44.
2. Evaluate retrieval and profile efficiency in separate GPU allocations.
3. Aggregate only after both branches succeed.

All GPU work requires the teaching partition, RTX PRO 6000 Blackwell, and
native BF16. Each task requests one GPU, 12 CPUs and 96 GiB host memory.
Profiling uses 20 discarded warm-ups and 100 synchronized repeats at batch 64.
Each model retains its native image resolution and native text context.

The parameter axis is full-stack inference parameters. Query-side parameters
are secondary. Runtime checks assert that reported full-stack and query-side
counts equal the parameters in the loaded model. One multiply-add counts as two
FLOPs; projection and normalization are included, while preprocessing and
tokenization are excluded.

Results are written under `results/efficiency_frontier/`, with the rendered
view in `notebooks/05_efficiency_frontier_results.ipynb`.
