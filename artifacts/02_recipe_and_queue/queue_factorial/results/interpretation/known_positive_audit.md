# Known-positive and residual false-negative audit

**Status:** post-observation reanalysis of existing diagnostics; no retraining  
**Inputs:** `results/queue_factorial/ledger.jsonl`,
`src/training/losses.py`, and
`configs/queue_factorial/analysis_plan.yaml`

## Question

Could queue damage be explained by captions of the same COCO image reappearing
in the historical queue and being treated incorrectly as negatives?

## Result

No. The bidirectional loss constructs positive masks from `image_ids` and
`text_image_ids` *after* appending queued embeddings. A resident queue entry
sharing an image identity with a current query is therefore included as a
multi-positive, not used as a negative. The queue-positive diagnostic is the
fraction of resident entries that are positive for at least one current query.
Its required epoch-boundary check observed non-zero values, showing that the ID
matching path was exercised rather than vacuously absent.

Across the 60 queue-enabled factorial runs, the aggregate queue-positive
fraction ranged from 0.00122% to 0.25737%. At the age-one harm boundary, the
three-seed mean was 0.00147% for batch 512 and 0.00792% for batch 1024. These
known same-image recurrences were explicitly protected by the loss.

## Claim boundary

This rules out one narrow mechanism: *known same-image captions incorrectly
treated as negatives*. It does not rule out semantically valid cross-image
matches that COCO does not annotate, near-duplicate images with different IDs,
or other forms of false-negative structure. Those require semantic or human
relevance annotation and remain future work. It also does not identify the
remaining mechanism among representation geometry and gradient sensitivity.
