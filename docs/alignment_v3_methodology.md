# Alignment v3 methodology and execution contract

## Research question

Alignment v3 tests whether modern independently pretrained vision and text
encoders can approach paired-VLM retrieval quality through teacher-guided
alignment and tightly constrained parameter-efficient adaptation, without
jointly pretraining the towers from scratch or sacrificing the efficiency
advantage established by Alignment v2.

Alignment v2 is an immutable safety-net result. Alignment v3 writes only below
`configs/alignment_v3`, `src/alignment_v3`, `checkpoints/alignment_v3`,
`results/alignment_v3`, `logs/alignment_v3` and `slurm/alignment_v3`.

## Leakage-safe protocol

Validation deterministically partitions COCO train into:

- 113,287 training images;
- 5,000 development images;
- the existing 5,000 COCO-validation images as the sealed final set.

Pair, distillation-strength and recipe selection use only the development
split. After locking the recipe, all three final seeds are retrained for a
fixed 12 epochs on full COCO train. Only those locked runs are evaluated on
COCO validation. Split manifests store exact hashes and zero-overlap checks.

## Compute policy

The target device is the NVIDIA RTX PRO 6000 Blackwell Server Edition with
96 GB memory. Training uses BF16, TF32, fused AdamW when supported,
channels-last ConvNeXt inputs, pinned asynchronous loading, 12 workers,
prefetch factor 4 and a 16,384-entry detached negative queue.

The fallback batch is 768 images (1,536 captions). A GPU profiler tries
256, 512, 768, 1024, 1536 and 2048 images per step for every pair. It selects
the largest batch within 3% of maximum throughput and below 88% allocated GPU
memory, then uses the minimum recommendation as the common comparison batch.
Evaluation uses a separate batch of 256 because it retains all five captions.
Learning rate follows capped square-root scaling from the proven batch-128
setting. A matched two-seed batch-128 control on the selected pair measures
whether the high-throughput regime itself changes retrieval quality.

## Experimental stages and gates

1. Validate dependencies, create splits, lock Alignment v2 and write manifests.
2. Prefetch all exact pretrained checkpoints.
3. Evaluate OpenCLIP QuickGELU, MobileCLIP2-S0 and SigLIP2 using native
   checkpoint transforms.
4. Profile the common Blackwell batch.
5. Train and evaluate all six DINOv3 pairings for the full schedule, seed 42.
6. Select within a 0.5 percentage-point practical-equivalence margin, then
   prefer latency and parameters. Continue if mean R@1 is at least 20% or the
   gain over Alignment v2 is at least three percentage points.
7. Build a content-checksummed FP16 MobileCLIP2 teacher cache.
8. Compare a matched two-seed baseline with 0.5x, 1x and 2x distillation.
   Continue only if the best strength gains two points and reaches 30%.
9. Smoke-test adapter-only, LoRA-only and distillation-only on real data,
   including strict inference-export reconstruction.
10. Run seven conditions on seeds 42 and 43: tuned-batch baseline, batch-128
    baseline control, distillation, adapter, LoRA, adapter+distillation and
    adapter+distillation+LoRA.
11. Select under 80M inference parameters, 5M deployed trainable parameters
    and OpenCLIP latency. A budget failure is explicit and never silently
    relaxed.
12. Retrain the locked winner on full COCO train for seeds 42, 43 and 44.
13. Evaluate once on sealed COCO validation, calculate development-set expert
    oracle complementarity and generate the report.

Flickr30k is optional and never blocks COCO completion. Its absence is recorded
in validation until an official-distribution test CSV is supplied.

## Artifact integrity

Fingerprints cover configuration, exact train split, model checkpoint IDs,
preprocessing, cache schema, seed, relevant source files and direct software
versions. Training resume requires an exact fingerprint match. The teacher
cache additionally stores a SHA-256 content checksum, embedding dimensions,
row counts and one-to-one ID indices.

Full training checkpoints retain optimizer, scheduler, scaler, cross-batch
memory, RNG state and teacher-matching heads. `inference.pt` is a separate
strictly loadable export whose configuration reconstructs without
training-only teacher heads.

## Result reporting

[`notebooks/03_alignment_v3_results.ipynb`](../notebooks/03_alignment_v3_results.ipynb)
maps every Slurm stage. Each job executes only its tagged observational cell;
notebook failures are logged but cannot invalidate completed training.
Locally measured results and literature-only context are stored and displayed
separately.

## Commands

Preflight without submission:

```bash
bash scripts/submit_alignment_v3_pipeline.sh \
  --mode parallel \
  --max-total-gpus 8 \
  --dry-run
```

Submit:

```bash
bash scripts/submit_alignment_v3_pipeline.sh \
  --mode parallel \
  --max-total-gpus 8
```

Inspect or resume:

```bash
bash scripts/alignment_v3_status.sh
bash scripts/alignment_v3_resume.sh --mode parallel --max-total-gpus 8
```

## Isolated fixed-batch recovery

The failed official pair gate remains an immutable negative result. Recovery
uses `configs/alignment_v3_recovery/pipeline.yaml` and separate result,
checkpoint, log, split-manifest, and teacher-cache paths.

Run the adaptive screen:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase screen --mode parallel --max-total-gpus 8
```

The screen trains batch sizes 512, 768, 1024, and 2048 with seed 42. It ranks
those results, runs seed 43 only for the top two, and applies the unchanged
pair gate to the two-seed mean. The throughput probe is disabled, and the
base batch-128 learning rate is scaled automatically with the capped square
root rule for every job.

If and only if `results/alignment_v3_recovery/selection/pair.json` records
`"proceed": true`, submit the compact projection/distillation/LoRA study:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase recipes --mode parallel --max-total-gpus 8
```

The final phase is rejected unless the selected non-baseline recipe beats its
matched baseline by at least two percentage points in each seed and remains
within the configured parameter and measured OpenCLIP latency budgets:

```bash
bash scripts/submit_alignment_v3_recovery_pipeline.sh \
  --phase final --mode parallel --max-total-gpus 8
```

Inspect artifacts and scheduler state with:

```bash
bash scripts/alignment_v3_recovery_status.sh
```

See [`alignment_v3_recovery_methodology.md`](alignment_v3_recovery_methodology.md)
for the independently registered recovery hypothesis and gates.
