# Alignment v2: strong reference and improved frozen-unimodal experts

## Purpose

Alignment v2 is a new, non-destructive experiment family. It does not overwrite
the Phase 1 or Phase 2 results. It answers two narrower questions:

1. How far are the independently pretrained frozen vision/text experts from a
   checkpoint-native paired VLM under exactly the same COCO five-caption
   retrieval protocol?
2. How much can the frozen-unimodal experts improve through better alignment
   training without increasing their inference backbones?

The paired reference is OpenCLIP ViT-B/32-QuickGELU with OpenAI weights. The
QuickGELU architecture exactly matches the checkpoint tag. It is evaluated as
one frozen native paired model. The older `openclip_vit_b32` plus
`openclip_text` hooks are not used for this reference because those hooks
instantiate separate models and add new alignment projections.

## Frozen-unimodal improvements

The v2 training configuration keeps both pretrained towers frozen and changes
only the alignment procedure:

- all COCO training captions are available;
- each image is decoded and encoded once while two captions are sampled per
  image per epoch;
- rectangular multi-positive contrastive loss treats every caption belonging
  to the image as valid;
- a residual projection head is used instead of relying only on a plain MLP;
- a short detached queue supplies more negatives without affecting inference;
- AdamW excludes biases, normalisation parameters and temperature from weight
  decay;
- cosine scheduling, warm-up, gradient clipping and early stopping are enabled;
- checkpoints are selected using mean bidirectional R@1;
- validation uses all five captions and the same retrieval implementation as
  the reference.

The queue and extra training captions exist only during training. They add no
online inference parameters or latency.

## Experiment matrix

The canonical Alignment v2 pipeline runs three seeds for:

- DINOv2-S/14 + MiniLM;
- ConvNeXt-Tiny + MiniLM;
- EfficientNet-B0 + BGE-Small.

The first is the semantic expert, the second is a faster architecture-diverse
expert, and the third is the cheapest candidate for a later cascade.

Each run has a 12-epoch ceiling with early stopping. The paired reference,
training configuration, checkpoints, metrics, reports and executed notebooks
under `results/alignment_v2` and `checkpoints/alignment_v2` are the sole
canonical Alignment v2 evidence.

## Slurm dependency graph

```text
validate
  └── prefetch unique pretrained checkpoints
        ├── paired OpenCLIP reference evaluation ─┐
        └── frozen-unimodal training array        │
          └── matched evaluation array ───────────┤
                                                   └── report
```

Parallel submission:

```bash
bash scripts/submit_alignment_v2_pipeline.sh \
  --mode parallel \
  --max-total-gpus 8
```

Sequential submission:

```bash
bash scripts/submit_alignment_v2_pipeline.sh --mode sequential
```

Inspect the commands without submitting:

```bash
bash scripts/submit_alignment_v2_pipeline.sh --mode parallel --dry-run
```

Check saved and scheduler status:

```bash
bash scripts/alignment_v2_status.sh
```

## Executable result map

`notebooks/02_alignment_v2_results.ipynb` maps the output from every pipeline
stage: validation, prefetch, paired-reference evaluation, frozen-unimodal
training, matched evaluation, completion status, and the aggregate report.

The cells carry stage tags. After a successful Slurm task, its `.sbatch` script
executes the common setup cell and the matching result cell. Array tasks write
separate notebooks, so parallel workers never edit the same file:

```text
results/alignment_v2/notebooks/<stage>/<job_id>_<array_index>.ipynb
```

The final report job executes all tagged cells and therefore produces the
complete experiment notebook under `results/alignment_v2/notebooks/report`.
Notebook rendering is observational: a rendering failure is logged as a
warning but does not invalidate completed training or block downstream Slurm
dependencies.

Run or inspect a stage manually:

```bash
python scripts/run_alignment_v2_notebook_cell.py \
  --stage evaluate \
  --index 0

python scripts/run_alignment_v2_notebook_cell.py \
  --stage all \
  --output results/alignment_v2/notebooks/manual_complete.ipynb

python scripts/run_alignment_v2_notebook_cell.py \
  --stage train \
  --list-cells
```

Every reference, training and evaluation array task requests exactly one GPU
with `--gres=gpu:1`. The submission command itself runs on a login/CPU node:
Slurm queues the array tasks and starts each script when it assigns a GPU. No
interactive `salloc` session is required. `--max-total-gpus` caps the number of
simultaneously runnable GPU tasks across the parallel reference/training
branches; evaluation starts after training and uses the full cap.

Interrupted or partially completed work can be resubmitted safely:

```bash
bash scripts/alignment_v2_resume.sh
```

Completed tasks exit immediately, while incomplete training resumes from
`latest.pt`.

The Python entry points can also be run individually:

```bash
python -m src.alignment_v2.runner validate
python -m src.alignment_v2.runner prefetch
python -m src.alignment_v2.runner reference --index 0
python -m src.alignment_v2.runner train --index 0
python -m src.alignment_v2.runner evaluate --index 0
python -m src.alignment_v2.runner report
```

## Outputs

- `results/alignment_v2/reference/*/metrics.json`
- `checkpoints/alignment_v2/*/{best,latest,final}.pt`
- `results/alignment_v2/unimodal/*/metrics.json`
- `results/alignment_v2/results_long.csv`
- `results/alignment_v2/results_summary.csv`
- `results/alignment_v2/report.md`
- `results/alignment_v2/notebooks/<stage>/*.ipynb`

Each evaluated run contains bidirectional R@1/5/10, rank metrics, batch-one
image/text latency, peak allocated memory, and total/trainable parameters.

## Interpretation constraint

OpenCLIP is a reference baseline, not a teacher used during v2 training.
Results must not be described as state of the art. A useful v2 outcome is a
material reduction in the gap to the paired reference while retaining the
heterogeneous experts' lower cost or complementary error patterns. Router work
should begin only after this comparison is complete.
