# Efficiency frontier

Primary dataset: Flickr30k Karpathy test. COCO is supplementary and contaminated
by earlier project selection. The project has no untouched COCO test split.

Parameter Pareto frontier: locked_e5, locked_minilm, mobileclip2_s0_dfndr2b, siglip2_vit_b32_256_webli

IQR-aware latency frontier: openclip_vit_b32_quickgelu_openai, siglip2_vit_b32_256_webli

Raw-median latency frontier: openclip_vit_b32_quickgelu_openai, siglip2_vit_b32_256_webli

Caption FLOPs use each model's native context length and are therefore not
directly comparable without qualification. One multiply-add is counted as two
FLOPs; the alternative convention halves every reported FLOP value. COCO
captions average approximately 11 tokens, so padding policy affects FLOP
accounting more strongly than typical caption realism.

The students were selected on development data and are reported on Flickr30k
for the first time here. Upstream reference revisions were not recorded at
fetch time; prospective local checksums are supplied instead.
