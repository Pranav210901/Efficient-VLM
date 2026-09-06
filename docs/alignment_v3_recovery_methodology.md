# Alignment v3 batch-calibration recovery methodology

## Status and scope

This is a separately labelled exploratory recovery track. The official
Alignment v3 screen remains a negative result: DINOv3 ViT-S/16 plus
all-MiniLM-L6-v2 achieved 17.02% mean bidirectional R@1, a 1.42 percentage
point improvement over Alignment v2, and did not meet the registered
continuation gate.

The recovery track does not edit that gate, configuration, or result. It
writes only to:

- `configs/alignment_v3_recovery`
- `results/alignment_v3_recovery`
- `checkpoints/alignment_v3_recovery`
- `logs/alignment_v3_recovery`
- `slurm/alignment_v3_recovery`

## Motivation

The v3 throughput probe selected batch 2048 for every pair. For DINOv3
ViT-S/16, measured throughput differed by less than one percent between batch
256 and batch 2048, while memory remained far below the 96 GB device limit.
The selection rule therefore defaulted to the largest eligible batch without
considering optimization quality.

With approximately 113k training images, batch 2048 provides roughly 55
optimizer updates per epoch, or about 660 updates over 12 epochs. Batch 512
provides approximately four times as many updates at nearly the same measured
epoch throughput. Recovery treats batch size as a development-tuned training
hyperparameter rather than a throughput-only decision.

## Pre-registered batch screen

The architecture is locked to:

- Vision encoder: DINOv3 ViT-S/16
- Text encoder: all-MiniLM-L6-v2
- Initial seed: 42
- Fixed batches: 512, 768, 1024, and 2048
- Epochs: 12

Batch 2048 is the control. The automatic batch probe is disabled. All jobs
inherit one base learning rate of 0.0003 at batch 128 and apply:

`lr = 0.0003 × min(sqrt(batch / 128), 3.0)`

No batch has a manually assigned learning rate.

The initial four results are ranked by mean bidirectional R@1. The best two
batch settings are repeated with seed 43. The winning batch is selected by
the two-seed mean and must meet the same registered pair gate as v3:

- mean bidirectional R@1 at least 20%; or
- improvement over the Alignment v2 value of 15.60% at least 3 percentage
  points.

If neither confirmed batch meets that gate, recovery stops and is reported as
incomplete.

## Pre-registered compact recipe study

Only a passing batch screen permits this phase. The selected batch propagates
to every downstream training job. The four matched recipes are:

1. projection-only baseline;
2. distillation only;
3. LoRA only;
4. distillation plus LoRA.

Each recipe uses seeds 42 and 43. The token adapter is deferred.

The distillation strength is fixed at 1.0 before the run. Recovery deliberately
does not use v3's 30% absolute distillation-sensitivity gate: that gate was
designed for the original seven-recipe path and would prevent the compact
comparison from answering its stated question. This is a new, explicit
recovery design decision, not a retroactive change to v3.

Eligible models must satisfy all of:

- no more than 80M inference parameters;
- no more than 5M trainable parameters;
- paired latency no greater than the locally measured OpenCLIP latency
  (fallback registered ceiling: 4.169 ms).

For promotion, the winning non-baseline recipe must improve over the matched
projection-only baseline by at least 2 percentage points in each of seeds 42
and 43. Requiring the threshold in both seeds is stricter than applying it only
to their mean.

## Final evaluation

Only a passing recipe gate permits final training. The selected recipe and
batch are retrained on full COCO train with seeds 42, 43, and 44 for 12 epochs.
The sealed COCO validation split remains the final evaluation set. Optional
Flickr30k transfer remains unavailable until its official dataset CSV is
provided.

## Honest terminal status

Gate-skipped intermediate jobs may exit zero so the dependency graph can reach
its sink. The terminal report exits with code 3 whenever its scientific status
is not `COMPLETE`. A genuinely complete three-seed final report exits zero.

## Commands

Omitting `--phase` submits the complete dependency graph. Downstream stages
perform gate-aware no-ops, and an incomplete terminal report exits 3:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh --dry-run
```

For tighter queue control, the same workflow can be submitted in three
explicit phases.

Dry-run and submit the batch screen:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase screen --mode parallel --max-total-gpus 8 --dry-run

bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase screen --mode parallel --max-total-gpus 8
```

If `results/alignment_v3_recovery/selection/pair.json` records
`"proceed": true`, submit recipes:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase recipes --mode parallel --max-total-gpus 8
```

If `results/alignment_v3_recovery/selection/recipe.json` has a status beginning
with `SELECTED`, submit final training:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase final --mode parallel --max-total-gpus 8
```

Status:

```bash
bash scripts/alignment_v3_recovery_status.sh
```

## Resume behavior

Recovery training writes `latest.pt` after completed epochs and validates its
configuration, dataset, model, preprocessing, cache, code, and seed
fingerprint before loading it. Resubmission:

- skips complete training exports and complete evaluations with matching
  fingerprints;
- resumes interrupted training from the next epoch in `latest.pt`;
- reruns inexpensive validation and deterministic selection stages;
- resubmits array indices, allowing complete indices to no-op while unfinished
  indices continue;
- rejects stale checkpoints rather than silently mixing configurations.

Resume the complete graph:

```bash
bash scripts/alignment_v3_recovery_resume.sh \
  --mode parallel --max-total-gpus 8
```

Resume only the currently authorized phase:

```bash
bash scripts/alignment_v3_recovery_resume.sh \
  --phase screen --mode parallel --max-total-gpus 8
```

The same command supports `--phase recipes` or `--phase final`; their existing
scientific gate preflight checks still apply.
