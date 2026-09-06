# Project state after the Alignment v2 cutover

Cutover date: 2026-07-23

## Current research state

Alignment v2 is the active experiment family. The original Phase 1, Phase 1.5,
and Phase 2 implementation remains in the repository for provenance, but its
large execution artifacts are quarantined.

The canonical Alignment v2 run completed on 2026-07-23. It uses the
checkpoint-matched `ViT-B-32-quickgelu` OpenAI reference, counts that
inference-only reference as frozen, caps evaluation data loading at the ten
allocated CPUs, and trains every frozen-unimodal configuration with a 12-epoch
ceiling and early stopping. All ten evaluation rows, checkpoints, metrics,
reports and executed notebooks are present.

The following evidence boundary remains important:

- the legacy numbers are completed historical observations;
- the results under `results/alignment_v2` are the sole Alignment v2 results;
- the historical Phase 2 dense-router evidence must not be presented as a
  valid final router result because its evaluation used label-derived routing;
- OpenCLIP ViT-B/32 with OpenAI weights is a paired reference, not a teacher
  and not a claim of state-of-the-art performance.

## What remains from the previous project

The lightweight, inspectable parts remain in the active tree:

- all legacy source code under `src`, including Phase 1.5 and Phase 2;
- legacy configurations, Slurm files, scripts, tests, notebooks, and research
  documentation;
- compact canonical evidence in `results/legacy_summary`;
- the cleaned Phase 1 alignment matrix, matched five-seed summaries, finalist
  comparison, Phase 1.5 audit/methodology/expert-selection records, and Phase 2
  report/statistics;
- COCO images and caption annotations, which are also needed by Alignment v2.

Artifact-dependent legacy integration tests are skipped while the corresponding
execution state is quarantined. Legacy algorithmic/unit tests and all Alignment
v2 tests remain active. Restoring the cutover re-enables those integration
checks automatically.

The main legacy observations still available for review are:

- the best single-run Phase 1 matrix result had mean bidirectional R@1 of
  0.1538 (`DINOv2-S/14 + all-MiniLM-L6-v2 + local/global BLF`);
- the matched five-seed study reported mean bidirectional R@1 of 0.18776 for
  the DINOv2/MiniLM baseline and 0.19324 with the local branch;
- the 15-epoch ConvNeXt/MiniLM finalist did not improve over its baseline:
  0.14610 versus 0.14680 mean bidirectional R@1;
- Phase 1.5 remains useful for exploratory expert selection and diagnostics;
- Phase 2 outputs are retained as historical exploratory evidence only, not as
  publishable validation of the router.

The full old execution state was initially moved intact to
`quarantine/alignment_v2_cutover_20260723`. The AISurrey environment was later
restored to `.venv-aisurrey` at the user's request:

| Quarantined group | Approximate size | Contents |
|---|---:|---|
| `legacy_checkpoints` | 76 GB | Phase 1, finalist, and seed-sweep checkpoints |
| `legacy_results` | 43 GB | predictions, embeddings, hard negatives, bridge/router runs, and worker shards |
| `legacy_data` | 22 GB | redundant COCO ZIPs, non-caption annotations, legacy CSVs, and multitask datasets |
| `legacy_environment` | restored | `.venv-aisurrey` is active at the project root again |
| `legacy_logs` | 101 MB | completed legacy Slurm logs |
| `scratch_and_caches` | 256 KB | notebook, pytest, and Python caches |

Nothing in those groups was deleted. See the quarantine manifest or run
`scripts/restore_alignment_v2_cutover.sh` for recovery instructions.

## What Alignment v2 adds

Alignment v2 replaces the immediate router experiment with a smaller,
methodologically defensible alignment study:

| Dimension | Previous pipeline | Alignment v2 |
|---|---|---|
| Main comparison | heterogeneous encoders and BLF variants | frozen unimodal experts versus one native paired VLM |
| Reference | no checkpoint-native paired reference | OpenCLIP ViT-B/32-QuickGELU, OpenAI weights |
| Training captions | legacy one-caption sampling | all COCO captions, two sampled captions per image per epoch |
| Loss | square one-positive contrastive loss | rectangular multi-positive contrastive loss |
| Alignment head | plain projection/BLF variants | residual projection head |
| Negatives | current minibatch | minibatch plus detached 4,096-entry training queue |
| Backbone policy | pretrained towers | explicitly frozen unimodal towers |
| Model selection | legacy runs often selected on i2t R@1 | mean bidirectional R@1 |
| Validation | historically mixed protocols | common five-caption COCO retrieval implementation |
| Repeats | uneven, then focused five-seed studies | three predetermined seeds for every unimodal setup |
| Efficiency | fragmented measurements | parameters, batch-one latency, peak memory, and retrieval metrics together |
| Scheduling | several ad hoc launch paths | dependency-aware Slurm graph, parallel/sequential modes, resume and status |
| Router claim | invalid dense-router evaluation exists | router work deferred until expert quality is established |

The active matrix contains nine frozen-unimodal runs:

- DINOv2-S/14 + MiniLM, seeds 42/43/44;
- ConvNeXt-Tiny + MiniLM, seeds 42/43/44;
- EfficientNet-B0 + BGE-Small, seeds 42/43/44.

Every GPU task requests one scheduler-assigned GPU. The dependency graph
validates data and configuration, prefetches unique pretrained weights, runs
the paired reference and unimodal training, evaluates completed checkpoints,
and generates the final report.

## Active paths

- pipeline configuration: `configs/alignment_v2/pipeline.yaml`
- implementation: `src/alignment_v2/runner.py`
- Slurm graph: `slurm/alignment_v2`
- launcher: `scripts/submit_alignment_v2_pipeline.sh`
- status/resume helpers: `scripts/alignment_v2_status.sh` and
  `scripts/alignment_v2_resume.sh`
- new checkpoints: `checkpoints/alignment_v2`
- new results: `results/alignment_v2`
- new logs: `logs/alignment_v2`
- retained legacy summaries: `results/legacy_summary`

## Recommended next action

Run a dry-run first, then submit the dependency graph:

```bash
bash scripts/submit_alignment_v2_pipeline.sh \
  --mode parallel \
  --max-total-gpus 8 \
  --dry-run

bash scripts/submit_alignment_v2_pipeline.sh \
  --mode parallel \
  --max-total-gpus 8
```

Do not begin a new learned-router claim until the v2 report establishes the
accuracy/efficiency gap between each frozen expert and the paired reference.
