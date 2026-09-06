# Final zero-shot evaluation

This package evaluates the two best frozen-encoder systems and three locally
measured compact VLM references. It performs no learning and exposes no
optimizer path.

Primary retrieval evaluation: Flickr30k Karpathy test. Supplementary external
classification diagnostics: CIFAR-100 official test, Oxford-IIIT Pets official
test, and the fixed EuroSAT stratified test partition.

The roster, prompt templates, dataset roles, selected epochs, and checkpoint
paths are frozen in `configs/final_zero_shot/pipeline.yaml` before evaluation.
