# CC3M-Mirror Data-Scaling Branch — Closed

Status: **accepted negative result; branch closed**.

## Narrow result

Under a compute budget matched to the COCO control, with the learning rate
pinned from COCO and explicitly untuned for this CC3M mirror, the
CC3M-mirror subset underperformed COCO-trained Flickr30k validation transfer
by **8.13 percentage points at matched compute** and **5.89 percentage points
after four passes**.

The matched-compute arm produced 35.49% mean bidirectional R@1 (three-seed
SD 0.48), against 43.62% for the COCO-trained locked recipe. The
extended-training arm, ceiling reached, produced 37.73% (SD 0.26).

## Counterargument and limit

This is a budget-bounded negative result, not evidence that CC3M is
intrinsically worse. Seeds 42 and 43 were still improving at the 5,280-step
ceiling, early stopping never fired, and the extended-training arm recovered
2.23pp over the matched-compute arm. We did not establish that this
CC3M-mirror subset cannot overtake COCO with additional optimization; we
established that it did not within the pre-registered compute budget and
COCO-pinned learning-rate setting.

The extended-training arm must never be called “converged.” Its selected
passes were four for seeds 42 and 43 and three for seed 44. Seed 44 declined
at pass four.

## Cross-distribution finding

The data-scale pilot showed unique COCO coverage beating repeated COCO
coverage by 15.03pp at 1,320 steps. Arm A used the same 1,320-step budget
over 1,351,680 unique CC3M-mirror pairs and nevertheless lost to the
COCO-trained locked recipe by 8.13pp. Unique-pair coverage therefore helped
within distribution and hurt across distribution at this scale.

This partially qualifies the pilot’s implication: pair count alone did not
determine transfer. Caption and distribution properties—human-written,
retrieval-shaped COCO captions versus web alt-text—appear to matter more than
volume under this budget. Because those properties were not independently
manipulated, they are a plausible interpretation rather than an isolated
causal mechanism.

## Final disposition

Flickr30k test remains sealed. Neither CC3M arm enters the final efficiency
frontier. No learning-rate tuning, longer schedule, mixed-corpus experiment,
or other CC3M optimization is authorized. Any such work would constitute a
new research programme requiring fresh pre-registration.
