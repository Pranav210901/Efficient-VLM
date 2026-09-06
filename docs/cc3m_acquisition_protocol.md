# CC3M Acquisition and Training Protocol

This protocol was frozen before acquisition on 2026-07-29.

The acquisition uses CC3M as distributed by the WebDataset mirror
`pixparse/cc3m-wds` at revision
`46f3d69f840e59d77d52e8decfe5baec97e94c7f`. It is not described as the
original CC3M distribution: 2,905,954 of 3,318,333 original training URLs
survived the mirror's December 2021 acquisition, and its img2dataset filtering
and shortest-edge-at-most-512 resizing are inherited. A deterministic shuffle
with seed `20260729` selects 298 of the 576 training tar shards.

The completed acquisition rejected 1,105 of 1,502,070 examined usable
candidates on the 256-pixel minimum-side filter (0.074%), far below the 10%
allowance; 29 downloaded shards were consequently unused under the
pre-registered smallest-sufficient-prefix definition. Exactly one sample
overlapped COCO or Flickr30k by the approved exact-hash checks across the
1,502,070 candidates examined. The 20-shard descriptive audit produced zero
heterogeneity flags under the pre-registered rules, so seeded shard clustering
was empirically benign here. The sampling is nevertheless described as a
deterministic random usable subset of the seed-ordered shard pool, not as a
uniform sample of the mirror.

The usable subset contains exactly 1,351,680 mirror-only image-caption pairs.
There is no semantic, similarity, aesthetic, language-model, teacher or concept
filter. Images are decoded with EXIF orientation, converted to RGB, limited to
a 256-pixel minimum short side, 100 megapixels maximum, and aspect ratio
0.5--2.0. Captions contain 3--40 whitespace tokens and fewer than 512
characters. Exact raw-byte, decoded-pixel and normalized-pair duplicates are
removed. Exact decoded-pixel hashes are checked against COCO and both Flickr30k
splits. This does not detect resized, cropped, recompressed or otherwise
transformed near-duplicates.

Arm A trains for exactly 1,320 optimizer updates (one pass) at batch 1,024 for
seeds 42, 43 and 44. It does not select on Flickr30k validation. Arm B uses the
same subset and settings, evaluates on Flickr30k validation, has patience three
evaluations and a 5,280-step ceiling. Arm B is therefore selection-influenced
on Flickr30k validation. Flickr30k test remains sealed until the design is
frozen. Both arms load images per epoch and never use a feature cache. The
pinned LR `0.002545584412271571` was not tuned for CC3M.

Seeds share the frozen selected key set, shard-local ordering algorithm and
configuration. The seed is intentionally an input to projector initialization,
sample ordering and deterministic per-sample augmentation. Seeds 42/43/44 are
therefore independent training replicates, not initialization-only replicates.

The weekly retention job explicitly refreshes atime for every downloaded tar
and every manifest, ledger and config. A stat-only scan is insufficient because
`stat(2)` does not normally update atime.

At least 20 deterministic shards receive a descriptive homogeneity audit.
Per-shard caption-token, width, height, short-side and rejection statistics are
always reported. A range across shard-level statistics above 10% of the pooled
mean, or a rejection-rate deviation above one percentage point, is flagged for
inspection. Coefficients of variation are reported. This is not a hypothesis
test and the flag is not a verdict.

Only after all six Arm A/B runs complete may a fixed-augmentation cache be
created. The cache uses DINOv3 ViT-S/16 and all-MiniLM-L6-v2 frozen features,
float16 storage, deterministic resize-short-side 256 plus 256x256 centre crop,
bicubic interpolation, ImageNet normalization, and the native MiniLM tokenizer
with dynamic batch padding and truncation. It is a durable rerun artifact and
is not an input to either reported arm.

## Closed outcome

The data-scaling branch was closed on 2026-07-29 as an accepted,
budget-bounded negative result. At matched compute, the CC3M-mirror arm
reached 35.49% three-seed mean Flickr30k validation R@1, 8.13pp below the
43.62% COCO-trained locked recipe. The extended-training arm, ceiling reached,
recovered to 37.73% but remained 5.89pp below COCO. Early stopping never
fired: seeds 42 and 43 selected pass four and were still improving at the
5,280-step ceiling; seed 44 selected pass three.

This does not establish that CC3M-mirror training cannot overtake COCO with
more compute. It establishes that it did not within the pre-registered budget
using an LR pinned from COCO and explicitly untuned for CC3M. The earlier
pilot's 15.03pp benefit from unique COCO coverage therefore does not generalize
to unique cross-distribution web pairs under this budget. Pair count alone was
insufficient; coupled caption and distribution properties remain a plausible,
but not causally isolated, explanation.

Flickr30k test remains sealed, neither CC3M arm enters the efficiency
frontier, and no CC3M optimization branch is authorized.
