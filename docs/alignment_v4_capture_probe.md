# Alignment v4: distillation capture probe

## Fixed dissertation question

> Can modern frozen SSL vision encoders, combined with compatibility-based pair
> selection and sub-5M-parameter adaptation, approach compact jointly pretrained
> vision-language models under a fixed inference budget?

Alignment v4 does not replace this question. It is a bounded diagnostic that
decides whether the next experiment should continue pursuing absolute retrieval
performance through distillation or pivot to efficiency-normalized evidence.

## Pre-registered design

- Student pair: frozen DINOv3 ViT-S/16 plus frozen all-MiniLM-L6-v2.
- Teachers: MobileCLIP2-S0 and SigLIP2-B/32 in separate, matched branches.
- Conditions per teacher: projection-only baseline and distillation at strength
  1.0.
- Seeds per condition: 42 and 43.
- Training: 12 epochs, batch 512, bf16, identical capped-square-root LR rule.
- Evaluation: the disjoint 5,000-image COCO development split.
- Teacher cache: the complete COCO training-minus-development split, separately
  fingerprinted by teacher checkpoint, preprocessing, data, config, code, and
  cache version.

The only intentional between-branch changes are teacher identity, teacher
embedding width, and cache location/version.

## Decision rule

For each teacher:

`capture = (distilled R@1 - baseline R@1) / (teacher R@1 - baseline R@1)`

- Distilled mean R@1 at least 0.30: `VIABLE_ABSOLUTE_PATH`.
- Distilled mean R@1 at most 0.22: `PIVOT_EFFICIENCY_NORMALIZED`.
- Otherwise: `INCONCLUSIVE_MIDDLE`.

The absolute student threshold is the primary gate. Capture fraction is
directional because the teacher anchors are sealed COCO-validation measurements
whereas the student probe is evaluated on the disjoint COCO-development split.

## Execution and recovery

The default two-teacher run uses at most eight GPUs total:

```bash
bash scripts/submit_alignment_v4_pipeline.sh \
  --teacher both \
  --mode parallel \
  --max-total-gpus 8
```

Re-submit safely after cancellation or pre-emption:

```bash
bash scripts/resume_alignment_v4_pipeline.sh \
  --teacher both \
  --mode parallel \
  --max-total-gpus 8
```

Completed jobs are skipped only when their fingerprints match. Interrupted
training resumes from `latest.pt`; stale results or mismatched caches are not
silently reused. Each GPU job requests one GPU, 12 CPU cores, and 120 GiB RAM.

Inspect progress:

```bash
bash scripts/status_alignment_v4.sh
```

The terminal combine job writes:

- `results/alignment_v4/combined_probe_report.json`
- `results/alignment_v4/combined_probe_report.csv`
- `results/alignment_v4/combined_probe_report.md`
- `results/alignment_v4/notebooks/04_alignment_v4_results.ipynb`

Slurm stdout/stderr are under `logs/alignment_v4/slurm/`.

