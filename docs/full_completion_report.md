# Full Repository Completion and Results Audit

> **Superseded status notice (8 August 2026).** This document is a preserved
> 21 July readiness audit. Its statement that Phase 2 was not implemented is no
> longer current: Phase 2 completed, and the later frozen-alignment,
> FreezeShift, TokenShift, and latency-amendment tracks also closed. Use
> [`dissertation.md`](dissertation.md), [`completion_status.md`](completion_status.md),
> and the canonical [`experiments/README.md`](../experiments/README.md) for the
> current project state. The historical text below is retained for provenance.

**Project:** Adaptive alignment of lightweight pretrained vision and text encoders  
**Repository:** `alignment_vlm`  
**Audit date:** 21 July 2026  
**Audit mode:** Read-only inspection of the repository, saved notebook, datasets, source files, checkpoints, logs, and result artifacts  
**Current research state:** Phase 1 complete; Phase 1.5 complete; Phase 2 evidence gate passed; Phase 2 method not implemented  
**Primary notebook:** [`notebooks/01_experiment_workflow.ipynb`](../notebooks/01_experiment_workflow.ipynb)  
**Canonical Phase 2 input:** [`results/phase15/expert_selection/selected_experts.yaml`](../results/phase15/expert_selection/selected_experts.yaml)

## 1. Executive conclusion

The repository has completed the intended Phase 1 encoder/BLF screening and the Phase 1.5 evidence-building workflow. The final machine-readable readiness report is **`READY`**:

- `ready_to_implement_phase2 = true`;
- 23 of 23 readiness checks pass;
- required failures: 0;
- warnings: 0;
- all 115 expected development prediction exports are valid;
- all four selected primary paths have complete train/development hard-negative outputs;
- the training-negative leakage check reports zero overlap with protected test IDs;
- every selected checkpoint exists;
- all selected encoders are covered by the implementation registry;
- the notebook is committed in safe, results-only mode.

This does **not** mean Phase 2 has already been built. The readiness artifact explicitly records `phase2_implemented = false`. No learned router, controller, cross-attention module, adaptive multi-expert fusion method, or Phase 2 training loop exists in the current repository state.

### Overall status by research area

| Area | Completed evidence | Current status |
|---|---:|---|
| COCO preparation | 118,287 train images; 5,000 validation images; 591,753 train captions; 25,014 validation captions | Complete |
| Canonical Phase 1 grid | 5 vision encoders × 4 text encoders × 4 variants = 80 unique configurations | Complete |
| Historical labelled executions | 100 matrix executions, including 20 redundant MiniLM alias runs | Preserved |
| ConvNeXt 15-epoch finalist comparison | 2 runs | Complete |
| Matched-seed reliability | 20 runs across seeds 42–46 | Complete; both tested BLF gains inconclusive |
| Qualitative retrieval | 2 models × 3 fixed COCO queries | Complete; illustrative only |
| Historical multi-task evaluation | 23 configurations × 4 tasks = 92 evaluations | Complete; classification evidence is exploratory |
| Leakage-safe development evaluation | 23 configurations × 4 tasks = 92 evaluations | Complete |
| Strict prediction validation | 115/115 exports; 1,030,400 represented query/sample rows | Complete and valid |
| Complementarity | 253 retrieval pairs + 759 classification dataset-pairs | Complete |
| Oracle analysis | 253 retrieval pairs + 6,075 classification subsets | Complete; diagnostic upper bounds only |
| Cross-task analysis | 23 configurations × 5 headline metrics | Complete |
| Efficiency | 5 vision, 4 text, and 23 complete-pair profiles | Complete |
| Held-out hard negatives | 842,343 rows | Complete; diagnostic-only and non-trainable |
| Training hard negatives | 22,721,280 retrieval + 247,376 classification rows | Complete; train/development sources only |
| Expert selection | 3 primary vision, 3 primary text, 4 primary pair paths, 2 optional ablations | Complete |
| Readiness gate | 23/23 checks pass | **READY** |
| Optional compositional benchmarks | Winoground and SugarCrepe documented but not installed | Optional/missing |
| Untouched external retrieval evaluation | Flickr30k not installed | Not completed |
| Phase 2 method | No router/controller/adaptive fusion implementation | Not started |

## 2. Audit scope and integrity checks

This report was generated from a fresh read-only inspection. Existing notebooks, code, datasets, checkpoints, logs, and results were not modified. The only repository change made for this audit is this new Markdown report.

### Repository-scale inventory

| Directory | Files | Approximate size | Contents |
|---|---:|---:|---|
| `checkpoints/` | 497 | 76 GiB | Best/final/latest model states, metrics, seed configs and summaries |
| `data/` | 176,174 | 41 GiB | COCO, CIFAR-100, Pets, EuroSAT, prepared CSVs, split manifests and retained archives |
| `results/` | 1,640 | 8.9 GiB | Canonical metrics, embeddings/caches, predictions, analyses, plots and hard negatives |
| `logs/` | 313 | 101 MiB | Encoder-matrix, seed-sweep, multi-task and Phase 1.5 worker logs |
| `notebooks/` | 2 | 5.6 MiB | Main workflow notebook and notebook checkpoint copy |
| `src/` | 116 total; 58 Python source files | 960 KiB | Models, training, multi-task evaluators, Phase 1.5 analyses and utilities |
| `scripts/` | 37 total; 22 source/configuration scripts | 312 KiB | Training/evaluation launchers, rebuild utilities and Slurm jobs |
| `tests/` | 14 total; 6 Python test files | 208 KiB | Metric, workflow, multi-task and Phase 1.5 tests |
| `docs/` before this report | 7 | 72 KiB | Protocol, methodology, GPU runbook and prior reviews |

### Read-only structural validation performed in this audit

| Check | Result |
|---|---:|
| Python files parsed with `ast` | 77 |
| Python syntax errors | 0 |
| Result JSON files parsed | 28/28 |
| YAML files parsed, including `selected_experts.yaml` | 9/9 |
| CSV/CSV.GZ outputs with readable non-empty headers | 806/806 |
| Parquet outputs with readable metadata | 10/10 |
| Parquet metadata errors | 0 |
| Empty files under `results/` | 0 |
| Empty files under `checkpoints/` | 0 |
| Missing checkpoints referenced by historical multi-task manifest | 0/23 |
| Missing checkpoints referenced by development multi-task manifest | 0/23 |
| Missing checkpoints referenced by final six-path shortlist | 0/6 |

The only zero-byte file found under `data/` is the harmless extracted CIFAR-100 backup file `data/multitask/cifar100/cifar-100-python/file.txt~`. It is not an input to the workflow.

The ten Parquet files comprise two canonical combined outputs and eight per-configuration shards. Their metadata accounts for 45,937,312 rows when canonical files and their duplicate shard representation are both counted. The unique canonical Phase 2 training corpus contains 22,968,656 rows.

### Environment observed during this audit

| Component | Version/state |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |
| Torchvision | 0.23.0+cu128 |
| CUDA build | 12.8 |
| pandas | 2.3.3 |
| NumPy | 2.0.2 |
| PyArrow | 21.0.0 |
| timm | 1.0.27 |
| Transformers | 4.57.6 |
| GPU visible from the audit shell | No |

The absence of a visible GPU here does not invalidate the saved GPU results. The efficiency manifest records a single consistent device, `NVIDIA RTX PRO 6000 Blackwell Server Edition`, and the final Step 16 run was completed through the four-GPU Slurm workflow.

No relevant training/evaluation process was running in the local process table during the audit. The current shell did not expose the `squeue` command, so cluster scheduler state could not be independently queried from this environment.

### Test evidence caveat

The existing supervisor review records an earlier successful **59/59** test run. The current `.pytest_cache` lists 65 collected node IDs and contains one stale `lastfailed` entry dated 15 July for `test_notebook_defaults_integrity_and_phase_order`. The notebook and readiness outputs were subsequently repaired and the final readiness check passes, but a fresh test suite was intentionally not executed during this strictly read-only audit. Therefore:

- current source syntax is verified;
- current result schemas and readiness are verified;
- the report does not claim a newly executed 65/65 test run.

## 3. Implemented research system

### Core Phase 1 model

The implemented model keeps the pretrained vision and text backbones frozen. The trainable alignment system consists of:

1. a pretrained vision encoder producing the main visual representation;
2. optional lightweight local and/or global BLF visual branches;
3. a concatenation/MLP vision fusion module;
4. trainable vision and text projection heads into a shared 256-dimensional space;
5. L2-normalised image/text embeddings;
6. a learned similarity temperature;
7. symmetric CLIP-style contrastive loss.

Four vision-side variants are represented in the canonical result grid:

| Variant | Main vision backbone | Local BLF | Global BLF |
|---|---:|---:|---:|
| `baseline` | Yes | No | No |
| `local` | Yes | Yes | No |
| `global` | Yes | No | Yes |
| `local_global` | Yes | Yes | Yes |

### Canonical encoder search space

| Vision encoders | Text encoders |
|---|---|
| EfficientNet-B0 | All-MiniLM-L6-v2 |
| ConvNeXt-Tiny | BGE-Small-en |
| ConvNeXtV2-Tiny | E5-Small-v2 |
| DINOv2 ViT-S/14 | DistilBERT |
| Swin-Tiny | — |

The historical labels `minilm_l6` and `all_minilm_l6_v2` refer to the same underlying Sentence Transformers model. The authoritative cleaned grid drops the redundant alias rather than treating it as an independent expert.

### Multi-task evaluation system

The repository now implements:

- bidirectional COCO retrieval evaluation;
- image-to-text (`i2t`) and text-to-image (`t2i`) Recall@1/5/10;
- mean and median rank in each retrieval direction;
- zero-shot CIFAR-100, Oxford-IIIT Pets and EuroSAT classification;
- prompt ensembling and per-class outputs;
- top-1, top-5, balanced accuracy, macro-F1, confidence and signed margin metrics;
- embedding caches and prediction exports;
- one-configuration-per-GPU coordination;
- atomic output writes, locks, worker recovery and resume support;
- separate historical and development evidence roots;
- optional Winoground and SugarCrepe scoring functions.

### Phase 1.5 analysis system

The implemented Phase 1.5 pipeline includes:

- artifact auditing;
- strict prediction and sample coverage validation;
- pairwise retrieval and classification complementarity;
- per-class error analysis;
- pair, triple and full-pool oracle analysis;
- within-task normalisation and rank correlations;
- specialisation and cost-adjusted scores;
- GPU latency, memory, parameter and FLOP profiling;
- canonical seed-reliability mapping;
- diagnostic hard-negative mining from held-out evidence;
- leakage-safe training-negative mining from training partitions;
- diversity- and cost-constrained expert selection;
- a 23-check Phase 2 readiness gate.

## 4. Dataset completion and data-governance state

### Physical dataset inventory

| Dataset/artifact | Physical or tabular count | Status |
|---|---:|---|
| COCO train2017 images | 118,287 image files | Present |
| COCO val2017 images | 5,000 image files | Present |
| `data/train.csv` | 118,287 rows, one caption per train image | Present |
| `data/val.csv` | 5,000 rows, one caption per validation image | Present |
| `data/val_all_captions.csv` | 25,014 caption rows over 5,000 images | Present |
| `data/splits/coco_train_all_captions.csv` | 591,753 COCO train caption rows | Present |
| CIFAR-100 | Official 50,000 train and 10,000 test samples | Downloaded/extracted |
| Oxford-IIIT Pets | 3,680 trainval + 3,669 official test samples | Downloaded/extracted |
| Oxford-IIIT Pets image directory | 7,390 JPG files; protocol uses 7,349 labelled samples | Present |
| EuroSAT | 27,000 image files | Downloaded/extracted |
| EuroSAT split manifest | 27,000 rows | Present |
| Winoground | 0 files | Optional and not installed |
| SugarCrepe | 0 files | Optional and not installed |
| Flickr30k external retrieval | Not present | Untouched retrieval test unavailable |

### Frozen development/final-use policy

| Dataset | Training source | Development/selection source | Reserved final source | Current interpretation |
|---|---|---|---|---|
| COCO retrieval | train2017 | val2017 | Flickr30k if installed | COCO validation is development evidence, not untouched final evidence |
| CIFAR-100 | 40,000-sample deterministic train partition | 10,000-sample deterministic train development partition | Official 10,000 test | Historical test results retained as exploratory only |
| Oxford-IIIT Pets | 2,944-sample trainval train partition | 736-sample trainval development partition | Official 3,669 test | Historical test results retained as exploratory only |
| EuroSAT | 18,900 deterministic stratified train | 4,050 deterministic stratified development | 4,050 deterministic stratified test | Earlier complete-dataset result is exploratory |
| Winoground | None | Evaluation-only if installed | Evaluation-only | Cannot provide training negatives |
| SugarCrepe | None | Evaluation-only if installed | Evaluation-only | Cannot provide training negatives |

The frozen protocol is stored in [`configs/evaluation_protocol.yaml`](../configs/evaluation_protocol.yaml). The leakage audit explicitly prevents held-out COCO-validation or classification-test error rows from being used to train Phase 2.

## 5. Notebook-by-notebook-step completion report

## Step 0 — Project setup, controls and safe execution

**Status: Complete.**

The notebook contains 45 cells: 23 Markdown and 22 code cells. All 22 code cells retain output, and 20 retain non-null execution counts. The persisted safety controls are:

```python
RUN_PHASE1_TRAINING = False
RUN_SEED_SWEEP = False
RUN_QUALITATIVE_RETRIEVAL = False
RUN_MULTITASK_EVALUATION = False
RUN_PHASE15 = False
RUN_DEVELOPMENT_EVALUATION = False
RUN_EFFICIENCY_PROFILING = False
RUN_DIAGNOSTIC_HARD_NEGATIVE_MINING = False
RUN_TRAINING_HARD_NEGATIVE_MINING = False
RUN_MODE = "smoke"
RESULTS_ONLY = True
```

These settings mean a default notebook run displays saved results and does not launch training, evaluation, profiling or mining.

The notebook contains some older saved cell outputs from intermediate runs, including an earlier `ready_to_implement_phase2 = false`. The later Step 17 output and the standalone readiness files are newer and authoritative: they record `true` and `READY`.

## Step 1 — Prepare and validate COCO

**Status: Complete.**

| Prepared file | Rows | Purpose |
|---|---:|---|
| `data/train.csv` | 118,287 | One-caption-per-image contrastive training |
| `data/val.csv` | 5,000 | One-caption-per-image Phase 1 validation |
| `data/val_all_captions.csv` | 25,014 | Standard multi-positive COCO validation |
| `data/splits/coco_train_all_captions.csv` | 591,753 | Training-split bidirectional hard-negative mining |

Earlier integrity analysis records no train/validation image overlap, no nulls in the one-caption tables, and no exact duplicate one-caption rows. The all-caption validation file retains six duplicate source annotation rows as separate text queries, which is valid because grouped-positive retrieval treats every caption belonging to the same image as correct.

## Steps 2–5 — Baseline grid, cleanup, BLF grid and canonical retrieval evaluation

**Status: Complete.**

The raw aggregate contains 104 rows and mixes historical recovery states. It is preserved for provenance but is not authoritative. The cleaned table contains exactly:

```text
5 vision encoders × 4 canonical text encoders × 4 variants = 80 rows
```

It has zero duplicate canonical `(vision, text, variant)` identities and contains 20 rows for each variant.

### Best configuration within each variant

| Variant | Vision + text | i2t R@1 | t2i R@1 | Mean R@1 | Best epoch |
|---|---|---:|---:|---:|---:|
| Baseline | DINOv2 + MiniLM | 14.44% | 14.94% | 14.69% | 2 |
| Global BLF | DINOv2 + MiniLM | 14.54% | 15.16% | 14.85% | 3 |
| Local BLF | DINOv2 + MiniLM | **15.14%** | 15.42% | 15.28% | 3 |
| Local+global BLF | DINOv2 + MiniLM | 14.78% | **15.98%** | **15.38%** | 2 |

The primary Phase 1 selection metric was i2t R@1. By that criterion, DINOv2 + MiniLM + local BLF is the single-seed leader. By the descriptive mean of i2t and t2i R@1, the local+global variant is the leader.

### Top ten canonical configurations by mean R@1

| Rank | Vision | Text | Variant | i2t R@1 | t2i R@1 | Mean R@1 |
|---:|---|---|---|---:|---:|---:|
| 1 | DINOv2 | MiniLM | local_global | 14.78% | 15.98% | **15.38%** |
| 2 | DINOv2 | MiniLM | local | **15.14%** | 15.42% | 15.28% |
| 3 | DINOv2 | MiniLM | global | 14.54% | 15.16% | 14.85% |
| 4 | ConvNeXt-Tiny | MiniLM | local | 14.06% | 15.48% | 14.77% |
| 5 | ConvNeXtV2-Tiny | MiniLM | local_global | 14.38% | 15.06% | 14.72% |
| 6 | DINOv2 | BGE | local | 14.32% | 15.10% | 14.71% |
| 7 | DINOv2 | MiniLM | baseline | 14.44% | 14.94% | 14.69% |
| 8 | ConvNeXt-Tiny | MiniLM | baseline | 14.26% | 15.10% | 14.68% |
| 9 | ConvNeXtV2-Tiny | MiniLM | local | 14.20% | 15.08% | 14.64% |
| 10 | ConvNeXt-Tiny | MiniLM | local_global | 14.10% | 15.12% | 14.61% |

### Mean performance by variant

| Variant | Mean i2t R@1 | Mean t2i R@1 | Mean i2t R@5 | Mean t2i R@5 | Mean validation loss |
|---|---:|---:|---:|---:|---:|
| Baseline | 10.907% | 11.644% | 30.632% | 32.303% | **0.6749** |
| Global | 10.847% | 11.716% | 30.713% | 32.069% | 0.7180 |
| Local | 11.065% | 11.825% | 30.930% | **32.582%** | 0.7085 |
| Local+global | **11.106%** | **11.964%** | **30.985%** | 32.518% | 0.7083 |

Local+global BLF has the highest mean Recall@1, but its mean gain over baseline is small: +0.199 percentage points i2t and +0.320 percentage points t2i. BLF validation loss is not consistently lower, so retrieval quality and contrastive loss should be discussed separately.

### Encoder-level descriptive averages over all variants and partners

| Vision encoder | Mean i2t R@1 | Mean t2i R@1 |
|---|---:|---:|
| DINOv2 ViT-S/14 | **12.854%** | **13.641%** |
| ConvNeXt-Tiny | 12.261% | 13.154% |
| ConvNeXtV2-Tiny | 12.044% | 13.145% |
| Swin-Tiny | 9.754% | 10.749% |
| EfficientNet-B0 | 7.994% | 8.248% |

| Text encoder | Mean i2t R@1 | Mean t2i R@1 |
|---|---:|---:|
| All-MiniLM-L6-v2 | **12.795%** | **13.694%** |
| BGE-Small-en | 11.562% | 12.415% |
| E5-Small-v2 | 10.931% | 11.743% |
| DistilBERT | 8.637% | 9.297% |

### Best-of-three BLF change for every encoder pair

This is a screening analysis and is optimistic because the best of three BLF variants is selected separately for every baseline.

| Vision | Text | Baseline i2t R@1 | Best BLF | Best BLF i2t R@1 | Change |
|---|---|---:|---|---:|---:|
| EfficientNet-B0 | MiniLM | 9.16% | global | 10.44% | **+1.28 pp** |
| EfficientNet-B0 | DistilBERT | 5.20% | local_global | 6.44% | **+1.24 pp** |
| DINOv2 | E5 | 12.02% | local_global | 13.12% | **+1.10 pp** |
| DINOv2 | BGE | 13.28% | local | 14.32% | **+1.04 pp** |
| ConvNeXtV2 | MiniLM | 13.36% | local_global | 14.38% | **+1.02 pp** |
| DINOv2 | MiniLM | 14.44% | local | 15.14% | **+0.70 pp** |
| Swin | MiniLM | 11.26% | global | 11.92% | +0.66 pp |
| ConvNeXtV2 | E5 | 11.90% | local | 12.56% | +0.66 pp |
| ConvNeXt | E5 | 12.34% | local_global | 12.94% | +0.60 pp |
| ConvNeXt | BGE | 12.78% | local | 13.32% | +0.54 pp |
| Swin | BGE | 10.20% | local_global | 10.74% | +0.54 pp |
| EfficientNet-B0 | BGE | 8.66% | local_global | 9.04% | +0.38 pp |
| ConvNeXtV2 | DistilBERT | 9.60% | local | 9.96% | +0.36 pp |
| ConvNeXt | DistilBERT | 9.96% | local_global | 10.28% | +0.32 pp |
| ConvNeXtV2 | BGE | 12.68% | local | 12.84% | +0.16 pp |
| Swin | E5 | 9.94% | local_global | 9.98% | +0.04 pp |
| ConvNeXt | MiniLM | 14.26% | local_global | 14.10% | −0.16 pp |
| EfficientNet-B0 | E5 | 8.18% | global | 7.96% | −0.22 pp |
| Swin | DistilBERT | 7.88% | global | 7.52% | −0.36 pp |
| DINOv2 | DistilBERT | 11.04% | local_global | 10.62% | −0.42 pp |

The selected best BLF beats baseline in 16 of 20 pairs, loses in four, has a mean screened improvement of +0.474 percentage points, and a median improvement of +0.540 percentage points.

### Parameter overhead

| Variant vs baseline | Extra trainable parameters | Mean trainable increase | Mean total-model increase |
|---|---:|---:|---:|
| Local | 73,568 | 5.17% | 0.13% |
| Global | 102,304 | 7.19% | 0.18% |
| Local+global | 175,872 | 12.36% | 0.31% |

BLF is lightweight in total parameter terms. Its empirical value is nevertheless pair- and task-dependent.

## Step 6 — ConvNeXt 15-epoch finalist comparison

**Status: Complete.**

| Configuration | Best epoch | i2t R@1 | t2i R@1 | Mean R@1 | i2t R@5 | t2i R@5 | Validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny + MiniLM baseline | 2 | **14.26%** | 15.10% | **14.68%** | 36.82% | **39.66%** | 0.5676 |
| ConvNeXt-Tiny + MiniLM local+global | 2 | 14.10% | **15.12%** | 14.61% | **37.94%** | 39.44% | **0.5627** |

The longer requested horizon did not move the best checkpoint past epoch 2. The BLF finalist improved t2i R@1 by 0.02 percentage points, i2t R@5 by 1.12 points, validation loss and mean ranks, but the baseline retained the stronger primary i2t R@1 and mean R@1. This comparison does not establish a clear overall BLF winner.

## Steps 7–8 — Matched-seed sweep and reliability analysis

**Status: Complete.**

Twenty runs were completed:

```text
2 matched comparisons × 2 variants × 5 seeds = 20 runs
```

Seeds are 42, 43, 44, 45 and 46. All runs use the same five-caption COCO protocol.

### Per-variant seed summary

| Comparison | Variant | Runs | Mean i2t R@1 | Standard deviation | 95% interval |
|---|---|---:|---:|---:|---:|
| ConvNeXtV2 + MiniLM | baseline | 5 | 17.732% | 0.789 pp | 16.753%–18.711% |
| ConvNeXtV2 + MiniLM | local_global | 5 | 18.212% | 0.535 pp | 17.548%–18.876% |
| DINOv2 + MiniLM | baseline | 5 | 18.776% | 0.730 pp | 17.870%–19.682% |
| DINOv2 + MiniLM | local | 5 | 19.324% | 0.341 pp | 18.901%–19.747% |

### Matched BLF deltas

| Comparison | Mean matched delta | Wins/ties/losses | 95% matched interval | Reliability outcome |
|---|---:|---:|---:|---|
| ConvNeXtV2 + MiniLM local+global | +0.480 pp | 3/0/2 | −0.874 to +1.834 pp | **Inconclusive** |
| DINOv2 + MiniLM local | +0.548 pp | 4/0/1 | −0.296 to +1.392 pp | **Inconclusive** |

Both mean deltas are positive, but neither confidence interval excludes zero. The final expert selector therefore retains the matched baselines as reliable primary paths and prevents the inconclusive BLF variants from duplicating them in the primary shortlist.

## Step 8A — Qualitative retrieval

**Status: Complete; illustrative only.**

Two seed-42 BLF checkpoints were compared on the same three fixed caption queries against the full 5,000-image COCO validation gallery.

| Model | Queries | R@1 | R@5 | R@10 | Mean target rank |
|---|---:|---:|---:|---:|---:|
| DINOv2 + MiniLM + local BLF | 3 | 33.3% | 33.3% | 66.7% | 122 |
| ConvNeXtV2 + MiniLM + local+global BLF | 3 | 0.0% | 0.0% | 33.3% | 182 |

For the DINOv2 model, the three correct target ranks were 6, 359 and 1. For the ConvNeXtV2 model, they were 7, 453 and 86. With only three selected queries, these outputs demonstrate behaviour and support visual inspection; they are not statistical performance estimates.

## Step 9 — Prepare and validate multi-task datasets

**Status: Complete for retrieval and zero-shot classification.**

All four mandatory tasks report `ready`:

| Task | Group | Historical sample source | Development sample source | Classes/categories |
|---|---|---:|---:|---:|
| COCO retrieval | Retrieval | 25,014 captions / 5,000 images | Same COCO val development gallery | — |
| CIFAR-100 | Zero-shot classification | 10,000 official test | 10,000 train-derived development | 100 |
| Oxford-IIIT Pets | Zero-shot classification | 3,669 official test | 736 trainval-derived development | 37 |
| EuroSAT | Zero-shot classification | 27,000 complete dataset | 4,050 stratified development | 10 |

The datasets were loaded with automatic download disabled during evaluation, avoiding accidental web access or dataset mutation.

## Step 10 — Multi-task smoke test

**Status: Completed previously.**

Four smoke-mode cache files exist for the ConvNeXt-Tiny + MiniLM baseline: COCO retrieval, CIFAR-100, Pets and EuroSAT. The final canonical manifests contain full-mode results rather than smoke rows, which prevents a smoke cache from being mistaken for completed full evaluation.

## Step 11 — Full historical retrieval and zero-shot evaluation

**Status: Complete; classification results are exploratory.**

The historical root contains:

- 23 configurations;
- 4 tasks;
- 92 completed full configuration-task records;
- 759 long-form metric rows;
- 33 metrics per configuration;
- a 23-row × 37-column wide result table;
- 96 cache files, including four retained smoke caches;
- 168 prediction/per-class files.

The 23 configurations are 20 canonical baselines plus three targeted BLF variants:

- DINOv2 + MiniLM local;
- ConvNeXtV2 + MiniLM local+global;
- DINOv2 + BGE local.

### Historical task leaders

| Headline metric | Best configuration | Result |
|---|---|---:|
| COCO i2t R@1 | DINOv2 + MiniLM local | **19.52%** |
| COCO t2i R@1 | DINOv2 + MiniLM local | **16.047%** |
| CIFAR-100 top-1 | ConvNeXtV2 + MiniLM baseline | **29.51%** |
| Pets top-1 | DINOv2 + BGE baseline | **6.868%** |
| EuroSAT top-1 | DINOv2 + BGE baseline | **22.289%** |

The retrieval tasks use COCO validation as development evidence. The historical classification scores use official CIFAR/Pets test sources and the complete EuroSAT dataset, so they are retained for exploration but are not untouched confirmatory results.

## Step 11.5 — Selected BLF multi-task experiment

**Status: Complete.**

All three BLF configurations improved both COCO retrieval directions. Across the nine historical classification comparisons, only DINOv2 + MiniLM local improved CIFAR-100; the other eight classification comparisons declined.

### Historical BLF changes against matched baselines

| BLF configuration | COCO i2t | COCO t2i | CIFAR top-1 | Pets top-1 | EuroSAT top-1 |
|---|---:|---:|---:|---:|---:|
| DINOv2 + MiniLM local | +1.600 pp | +0.612 pp | +0.550 pp | −0.572 pp | −0.556 pp |
| ConvNeXtV2 + MiniLM local+global | +1.300 pp | +0.456 pp | −0.500 pp | −0.654 pp | −2.570 pp |
| DINOv2 + BGE local | +0.880 pp | +0.344 pp | −0.370 pp | −1.008 pp | −0.519 pp |

This supports a retrieval-focused BLF interpretation, not a claim of consistent cross-task transfer.

## Step 12 — Consolidated historical reporting

**Status: Complete.**

The reporting code recognises both baseline and newly named BLF prediction/result files. The saved matched comparison table contains 15 rows: five headline metrics for each of the three BLF-versus-baseline comparisons. The historical reporting artifact remains useful for provenance, while Phase 1.5 selection correctly uses the later development root.

## Step 13 — Leakage-safe development evaluation, audit and prediction coverage

**Status: Complete and strictly valid.**

The development root contains 92 completed full configuration-task evaluations, 759 long-form metric rows, 23 wide rows, 69 classification JSONL prediction files and 69 per-class CSV files. COCO development results are assembled from the existing validation evidence; the three classification tasks use their deterministic development partitions.

### Development task leaders

| Headline metric | Best configuration | Result |
|---|---|---:|
| COCO i2t R@1 | DINOv2 + MiniLM local | **19.52%** |
| COCO t2i R@1 | DINOv2 + MiniLM local | **16.047%** |
| CIFAR-100 development top-1 | ConvNeXtV2 + MiniLM baseline | **29.79%** |
| Pets development top-1 | DINOv2 + E5 baseline | **7.337%** |
| EuroSAT development top-1 | DINOv2 + BGE baseline | **22.272%** |

### Top five development configurations per task

| Task | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---|---|---|---|---|
| COCO i2t | DINOv2+Mini local 19.52% | ConvNeXt+Mini baseline 18.52% | ConvNeXtV2+Mini L+G 18.30% | DINOv2+BGE local 18.24% | DINOv2+Mini baseline 17.92% |
| COCO t2i | DINOv2+Mini local 16.047% | ConvNeXt+Mini baseline 15.763% | ConvNeXtV2+Mini L+G 15.479% | DINOv2+Mini baseline 15.435% | DINOv2+BGE local 15.411% |
| CIFAR-100 | ConvNeXtV2+Mini baseline 29.79% | ConvNeXtV2+Mini L+G 29.30% | ConvNeXtV2+E5 baseline 28.51% | ConvNeXtV2+BGE baseline 28.49% | ConvNeXt+BGE baseline 26.94% |
| Pets | DINOv2+E5 baseline 7.337% | DINOv2+Mini baseline 7.065% | DINOv2+BGE baseline 6.929% | Swin+Mini baseline 6.522% | Swin+BGE baseline 6.250% |
| EuroSAT | DINOv2+BGE baseline 22.272% | ConvNeXtV2+Mini baseline 21.531% | DINOv2+BGE local 21.383% | ConvNeXtV2+E5 baseline 20.370% | ConvNeXt+BGE baseline 19.901% |

### BLF changes on the leakage-safe development evidence

| BLF configuration | COCO i2t | COCO t2i | CIFAR top-1 | Pets top-1 | EuroSAT top-1 |
|---|---:|---:|---:|---:|---:|
| DINOv2 + MiniLM local | +1.600 pp | +0.612 pp | +0.360 pp | −2.853 pp | +0.370 pp |
| ConvNeXtV2 + MiniLM local+global | +1.300 pp | +0.456 pp | −0.490 pp | +0.679 pp | −3.210 pp |
| DINOv2 + BGE local | +0.880 pp | +0.344 pp | −0.390 pp | −1.902 pp | −0.889 pp |

All three BLFs again improve both retrieval directions, but six of nine classification deltas are negative.

### Strict prediction coverage

| Export group | Expected exports | Valid exports | Expected samples | Observed samples | Duplicates | Non-finite | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO i2t | 23 | 23 | 115,000 | 115,000 | 0 | 0 | 0 |
| COCO t2i | 23 | 23 | 575,322 | 575,322 | 0 | 0 | 0 |
| CIFAR-100 classification | 23 | 23 | 230,000 | 230,000 | 0 | 0 | 0 |
| Pets classification | 23 | 23 | 16,928 | 16,928 | 0 | 0 | 0 |
| EuroSAT classification | 23 | 23 | 93,150 | 93,150 | 0 | 0 | 0 |
| **Total** | **115** | **115** | **1,030,400** | **1,030,400** | **0** | **0** | **0** |

The missing-predictions table has zero rows. COCO t2i files were successfully materialised from saved embeddings without retraining the models.

## Step 14 — Complementarity, oracle and cross-task analysis

**Status: Complete.**

### Retrieval complementarity

All `23 choose 2 = 253` model pairs were compared on COCO image-query success sets. Across all pairs:

- mean successful-set Jaccard: 0.176;
- mean pair oracle R@1: 24.821%;
- the strongest pair oracle is 31.20% R@1.

The best retrieval oracle pair combines:

- ConvNeXtV2 + MiniLM local+global;
- DINOv2 + MiniLM local.

Their pair oracle is 31.20% i2t R@1, compared with the 19.52% best individual result. This is evidence that their successful-query sets differ, not evidence that a deployable selector already achieves 31.20%.

### Classification complementarity

The classification analysis contains:

- 759 pair rows: 253 model pairs × 3 datasets;
- 37,191 per-class rows;
- 1,518 directional unique-win rows;
- 12 saved heatmaps.

| Dataset | Pair rows | Mean correct-set Jaccard | Mean prediction agreement | Mean margin Spearman | Largest directional unique fraction |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 development | 253 | 0.433 | 0.240 | 0.388 | 18.14% |
| Pets development | 253 | 0.065 | 0.108 | 0.155 | 7.20% |
| EuroSAT development | 253 | 0.239 | 0.305 | 0.245 | 18.69% |

Pets has very low agreement and correct-set overlap, indicating diverse mistakes, but the absolute single-model accuracy is also weak. Complementarity should therefore not be confused with task competence.

### Classification oracle analysis

The classification oracle export contains:

- every pair subset: 759 rows;
- every triple subset: 5,313 rows;
- one full 23-model pool per dataset: 3 rows;
- total subsets: 6,075;
- per-class oracle rows: 297,675;
- marginal contribution rows: 17,526.

| Dataset | Best individual | Best pair oracle | Best triple oracle | Full 23-model oracle | Full-pool gain |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 development | 29.79% | 38.17% | 43.18% | **69.08%** | +39.29 pp |
| Pets development | 7.337% | 13.723% | 18.750% | **43.478%** | +36.141 pp |
| EuroSAT development | 22.272% | 36.790% | 47.037% | **73.556%** | +51.284 pp |

The best pair/triple pools differ by task, which supports conditional expert selection. Every oracle number is explicitly labelled a non-deployable diagnostic upper bound.

### Cross-task rankings and specialisation

The five headline metrics are:

- COCO i2t R@1;
- COCO t2i R@1;
- CIFAR-100 development top-1;
- Pets development top-1;
- EuroSAT development top-1.

Raw recall and accuracy are not averaged. Each task is normalised relative to its best result before the mean multi-task score is calculated.

| Cross-task rank | Configuration | Mean normalised score | Tasks won | Strongest task | Specialisation index |
|---:|---|---:|---:|---|---:|
| 1 | DINOv2 + BGE baseline | **0.933** | 1 | EuroSAT | 0.041 |
| 2 | DINOv2 + MiniLM baseline | 0.917 | 0 | Pets | 0.041 |
| 3 | ConvNeXtV2 + MiniLM local+global | 0.901 | 0 | CIFAR-100 | 0.077 |
| 4 | ConvNeXtV2 + MiniLM baseline | 0.896 | 1 | CIFAR-100 | 0.105 |
| 5 | DINOv2 + E5 baseline | 0.887 | 1 | Pets | 0.061 |
| 6 | DINOv2 + BGE local | 0.884 | 0 | COCO t2i | 0.104 |
| 7 | DINOv2 + MiniLM local | 0.869 | 2 | COCO i2t/t2i | 0.156 |
| 8 | ConvNeXtV2 + BGE baseline | 0.865 | 0 | CIFAR-100 | 0.068 |
| 9 | ConvNeXt + BGE baseline | 0.864 | 0 | COCO t2i | 0.090 |
| 10 | ConvNeXt + MiniLM baseline | 0.854 | 0 | COCO t2i | 0.115 |

The DINOv2 + BGE baseline is the broadest high-performing development pair. DINOv2 + MiniLM local is the retrieval specialist, winning both retrieval directions but showing the largest task variation among the top seven.

### Rank correlations between tasks

| Task pair | Spearman | Kendall |
|---|---:|---:|
| COCO i2t vs COCO t2i | **0.988** | **0.945** |
| COCO i2t vs CIFAR-100 | 0.780 | 0.581 |
| COCO t2i vs CIFAR-100 | 0.796 | 0.605 |
| COCO i2t vs Pets | 0.449 | 0.316 |
| COCO t2i vs Pets | 0.437 | 0.308 |
| COCO i2t vs EuroSAT | 0.472 | 0.312 |
| COCO t2i vs EuroSAT | 0.467 | 0.304 |
| CIFAR-100 vs Pets | 0.283 | 0.196 |
| CIFAR-100 vs EuroSAT | 0.495 | 0.352 |
| Pets vs EuroSAT | 0.228 | 0.148 |

The retrieval directions are nearly redundant as rankings, while Pets and EuroSAT rank models differently from retrieval and from each other. This is meaningful evidence for maintaining complementary paths rather than selecting only the single best COCO model.

## Step 15 — Efficiency profiling

**Status: Complete.**

All profiles share:

- device: NVIDIA RTX PRO 6000 Blackwell Server Edition;
- batch size: 8;
- 2 warmup passes;
- 5 measured repeats;
- FLOP measurement enabled;
- one common profile signature;
- 0 profiling errors.

### Vision encoder profiles

| Vision encoder | Latency | Peak allocated memory | Parameters | FLOPs/batch |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | **1.899 ms** | 213.7 MiB | 4.01M | 6.15G |
| ConvNeXt-Tiny | 4.074 ms | 313.2 MiB | 27.82M | 71.28G |
| Swin-Tiny | 4.637 ms | 329.0 MiB | 27.52M | 71.84G |
| ConvNeXtV2-Tiny | 4.839 ms | 350.3 MiB | 27.87M | 71.28G |
| DINOv2 ViT-S/14 | 6.097 ms | 231.1 MiB | 22.06M | 97.98G |

### Text encoder profiles

| Text encoder | Latency | Peak allocated memory | Parameters | FLOPs/batch |
|---|---:|---:|---:|---:|
| DistilBERT | **1.572 ms** | 384.4 MiB | 66.36M | 7.49G |
| MiniLM | 1.704 ms | **214.2 MiB** | **22.71M** | **1.88G** |
| BGE-Small | 2.475 ms | 253.7 MiB | 33.36M | 3.76G |
| E5-Small | 2.520 ms | 255.5 MiB | 33.36M | 4.44G |

DistilBERT has the lowest measured latency but is much larger and uses more peak memory than MiniLM. MiniLM is the stronger balanced efficiency choice.

### Final shortlisted pair efficiency

| Role | Pair | Latency | Peak memory | Parameters | FLOPs/batch |
|---|---|---:|---:|---:|---:|
| Primary | EfficientNet-B0 + BGE | **4.756 ms** | 254.8 MiB | 39.01M | **9.94G** |
| Primary | ConvNeXt-Tiny + MiniLM | 5.557 ms | 313.2 MiB | 51.91M | 73.18G |
| Optional | ConvNeXt-Tiny + E5 | 6.364 ms | 354.4 MiB | 62.56M | 75.74G |
| Optional | Swin-Tiny + E5 | 6.913 ms | 369.6 MiB | 62.26M | 76.30G |
| Primary | DINOv2 + MiniLM | 7.526 ms | **231.1 MiB** | 45.95M | 99.88G |
| Primary | DINOv2 + BGE | 8.343 ms | 271.7 MiB | 56.60M | 101.75G |

All 23 multi-task configurations, including the three explicit BLF variants, have complete pair profiles.

## Step 16 — Diagnostic and training-split hard negatives

**Status: Complete.**

Step 16 now correctly separates two different uses of hard negatives.

### A. Historical held-out diagnostic negatives

| Output | Rows | Source | Training allowed |
|---|---:|---|---:|
| Bidirectional retrieval hard negatives | 720,336 | COCO val2017 | No |
| Classification hard negatives | 122,007 | CIFAR test, Pets test, complete EuroSAT | No |
| **Total** | **842,343** | Validation/test evidence | **No** |

The current combined diagnostic files use three selected configurations and eight retrieval negatives per query. The manifest preserves two additional older completed shard pairs, so five shard identities remain on disk while the current combined outputs contain the three currently selected configurations. Nothing was overwritten or deleted.

The sidecar declares:

```text
usage = diagnostic_only
allowed_for_training = false
source_split = validation_or_test
```

### B. Leakage-safe Phase 2 training candidates

Four primary pair paths were mined:

| Configuration | Retrieval rows | Classification rows | Status |
|---|---:|---:|---|
| DINOv2 + MiniLM baseline | 5,680,320 | 61,844 | Complete |
| DINOv2 + BGE baseline | 5,680,320 | 61,844 | Complete |
| ConvNeXt-Tiny + MiniLM baseline | 5,680,320 | 61,844 | Complete |
| EfficientNet-B0 + BGE baseline | 5,680,320 | 61,844 | Complete |
| **Total** | **22,721,280** | **247,376** | **Complete** |

Per configuration, retrieval rows are explained exactly by:

```text
118,287 COCO image queries × 8 negatives =   946,296 i2t rows
591,753 COCO caption queries × 8 negatives = 4,734,024 t2i rows
                                                  ---------
                                                  5,680,320
```

Per configuration, classification rows are:

```text
40,000 CIFAR training-partition samples
 2,944 Pets training-partition samples
18,900 EuroSAT training samples
------
61,844 rows, with one mined negative class per sample
```

The training schema records:

- `usage = phase2_training`;
- `allowed_for_training = true`;
- eight retrieval negatives per positive;
- four configuration fingerprints;
- `protected_test_overlap = 0`;
- status `complete`.

The canonical output SHA-256 hashes were independently recomputed during this audit and match the committed schema:

| File | Size | SHA-256 |
|---|---:|---|
| `retrieval_train_hard_negatives.parquet` | 407,916,890 bytes | `44b7c71acaad98cac762e7e5dfe2f726ac2f8128d861dc53355e12011f6637c9` |
| `classification_train_hard_negatives.parquet` | 7,289,273 bytes | `75be6a254111fa1e3d67b90163191fc1c8be70616b9b175693795342ebc115f2` |

The final coordinator state is `complete`. The four-GPU, resumable Slurm script is stored at [`scripts/run_step16_hard_negatives.slurm`](../scripts/run_step16_hard_negatives.slurm). It handles termination signals, preserves completed shards, and resumes without treating partial output as a completed commit.

No compositional training negatives were produced because Winoground and SugarCrepe are evaluation-only and no separate compositional training dataset is configured.

## Step 17 — Expert selection and final readiness

**Status: Complete; final gate is READY.**

The final selector joins development performance, unique wins, oracle marginal contribution, cross-task specialisation, measured latency/memory, seed evidence, architecture diversity and token-interface readiness.

### Selection weights

| Component | Weight |
|---|---:|
| Multi-task performance | 0.25 |
| Unique wins | 0.17 |
| Oracle contribution | 0.13 |
| Cross-task specialisation | 0.12 |
| Latency efficiency | 0.10 |
| Seed reliability | 0.07 |
| Memory efficiency | 0.06 |
| Architectural diversity | 0.05 |
| Token-interface readiness | 0.05 |

### Primary vision experts

| Rank | Vision expert | Selection score | Architecture family | Latency | Intended evidence role |
|---:|---|---:|---|---:|---|
| 1 | DINOv2 ViT-S/14 | **0.603** | Plain ViT | 6.097 ms | Strong semantic vision expert |
| 2 | ConvNeXt-Tiny | 0.578 | ConvNeXt CNN | 4.074 ms | Local/hierarchical vision expert |
| 3 | EfficientNet-B0 | 0.571 | Efficient CNN | **1.899 ms** | Complementary/low-cost vision expert |

Cheap fallback vision encoder: **Swin-Tiny**, measured at 4.637 ms.

### Primary text experts

| Rank | Text expert | Selection score | Architecture family | Latency | Intended evidence role |
|---:|---|---:|---|---:|---|
| 1 | MiniLM | **0.682** | MiniLM sentence transformer | 1.704 ms | Strong general text expert |
| 2 | BGE-Small | 0.584 | BGE embedding model | 2.475 ms | Complementary text expert |
| 3 | E5-Small | 0.469 | E5 embedding model | 2.520 ms | Complementary text expert |

Cheap fallback text encoder: **DistilBERT**, measured at 1.572 ms.

### Final pair shortlist

| Rank | Pair | Role | Selection score | Development multi-task score | Efficiency score | Seed status |
|---:|---|---|---:|---:|---:|---|
| 1 | DINOv2 + MiniLM baseline | Primary | **0.7324** | 0.7652 | 0.6168 | Supported baseline, 5 runs |
| 2 | DINOv2 + BGE baseline | Primary | 0.7250 | **0.8348** | 0.4531 | Not seed-tested |
| 3 | ConvNeXt-Tiny + MiniLM baseline | Primary | 0.6448 | 0.6783 | 0.7439 | Not seed-tested |
| 4 | EfficientNet-B0 + BGE baseline | Primary | 0.4985 | 0.3087 | **0.8770** | Not seed-tested |
| 5 | ConvNeXt-Tiny + E5 baseline | Optional ablation | 0.5505 | 0.5565 | 0.4891 | Not seed-tested |
| 6 | Swin-Tiny + E5 baseline | Optional ablation | 0.5157 | 0.4522 | 0.4219 | Not seed-tested |

The optional rows can have a higher numerical selection score than the fourth primary row because the primary set must satisfy hard diversity, low-cost and unique-error constraints as a group.

### Explicit BLF exclusions

| Excluded BLF | Reason |
|---|---|
| DINOv2 + MiniLM local | Positive mean retrieval delta, but matched confidence interval crosses zero; baseline retained |
| ConvNeXtV2 + MiniLM local+global | Positive mean retrieval delta, but matched confidence interval crosses zero; baseline preferred and duplicate prohibited |
| DINOv2 + BGE local | No matched-seed reliability evidence; baseline preferred and redundant primary path avoided |

This does not mean the BLF models performed badly. They are the top retrieval specialists in several analyses. It means the evidence does not justify spending scarce primary Phase 2 slots on a baseline and an unconfirmed near-duplicate BLF from the same encoder pair.

### Diversity constraints

All final constraints pass:

- at most two primary paths per vision encoder;
- at most two primary paths per text encoder;
- three represented vision architecture families;
- two distinct primary text encoders across primary pair paths;
- at least one low-cost path;
- at least one unique-error path;
- no inconclusive BLF duplicate in the primary set.

### All 23 readiness checks

| # | Check | Severity | Result |
|---:|---|---|---|
| 1 | Safe notebook defaults | Required | Pass |
| 2 | Canonical retrieval results | Required | Pass |
| 3 | Seed-sweep results | Required | Pass |
| 4 | Multi-task development outputs | Required | Pass |
| 5 | Strict prediction coverage | Required | Pass: 115/115 |
| 6 | Retrieval complementarity | Required | Pass |
| 7 | Classification complementarity | Required | Pass |
| 8 | Retrieval oracle | Required | Pass |
| 9 | Classification oracle | Required | Pass |
| 10 | Cross-task matrix | Required | Pass |
| 11 | Cross-task rank analysis | Required | Pass |
| 12 | Efficiency profiles | Required | Pass |
| 13 | Seed reliability mapping | Required | Pass |
| 14 | Canonical `selected_experts.yaml` | Required | Pass: schema v1 |
| 15 | Pair-shortlist diversity constraints | Required | Pass |
| 16 | No inconclusive BLF duplicate | Required | Pass |
| 17 | Training hard negatives | Required | Pass |
| 18 | Training-negative leakage validation | Required | Pass: zero overlap |
| 19 | Diagnostic negatives non-trainable | Required | Pass |
| 20 | Development/final evaluation protocol | Required | Pass |
| 21 | Selected checkpoints | Required | Pass |
| 22 | Selected encoder registry coverage | Required | Pass |
| 23 | Optional compositional benchmark status | Warning | Pass: status documented |

The current final report is [`results/phase15/phase2_readiness/readiness_report.json`](../results/phase15/phase2_readiness/readiness_report.json), last updated 21 July 2026 at 15:40 local time.

## 6. Major artifact inventory

### Phase 1 root outputs

| Artifact | Shape/content | Authority and use |
|---|---:|---|
| `results/alignment_matrix_results.csv` | 104 rows × 19 columns | Historical raw aggregate; do not use for final ranking |
| `results/alignment_matrix_best_clean.csv` | 80 × 27 | Authoritative canonical Phase 1 grid |
| `results/alignment_matrix_baseline_dedup_best.csv` | 15 × 20 | Historical non-MiniLM baseline subset |
| `results/finalist_convnext_tiny_minilm_l6_15ep.csv` | 2 × 23 | Finalist baseline/BLF comparison |
| `results/seed_sweep_results.csv` | 20 runs | Raw matched-seed output |
| `results/seed_sweep_summary.csv` | 4 rows | Per-comparison variant summary |
| `results/seed_sweep_paired_deltas.csv` | 2 rows | Matched BLF deltas and intervals |
| `results/seed_sweep_matched_comparison.png` | 1 plot | Visual seed comparison |

### Multi-task outputs

| Root/artifact | Content | Status |
|---|---:|---|
| `results/phase1_multitask/evaluation_manifest.csv` | 92 complete configuration-task rows | Historical/exploratory |
| `results/phase1_multitask/task_results_long.csv` | 759 metric rows | Historical/exploratory |
| `results/phase1_multitask/task_results_wide.csv` | 23 × 37 | Historical/exploratory |
| `results/phase1_multitask/blf_vs_baseline_multitask.csv` | 15 matched rows | Historical BLF comparison |
| `results/phase1_multitask_development/evaluation_manifest.csv` | 92 complete configuration-task rows | Frozen development evidence |
| `results/phase1_multitask_development/task_results_long.csv` | 759 metric rows | Frozen development evidence |
| `results/phase1_multitask_development/task_results_wide.csv` | 23 × 37 | Phase 1.5 selection input |
| `results/phase15/predictions/` | 47 files; 46 canonical full retrieval exports + 1 retained legacy duplicate | Retrieval analysis input |
| `results/phase1_multitask_development/predictions/` | 69 JSONL + 69 per-class CSV | Development classification analysis input |

### Phase 1.5 outputs

| Area | Main artifacts | Rows/files | Status |
|---|---|---:|---|
| Audit | `phase15_artifact_audit.csv/json` | 12 audited areas | Valid |
| Prediction validation | coverage CSV/JSON + empty missing table | 115 exports | Valid |
| Retrieval complementarity | `retrieval_pairwise.csv` | 253 rows | Complete |
| Classification complementarity | pairwise, per-class, unique-wins | 759 + 37,191 + 1,518 rows | Complete |
| Complementarity plots | Heatmaps | 12 PNGs | Complete |
| Retrieval oracle | `retrieval_oracle.csv` | 253 rows | Complete |
| Classification oracle | oracle, per-class, marginal | 6,075 + 297,675 + 17,526 rows | Complete |
| Cross-task | raw/normalised matrices, ranks, correlations, specialisation | 13 files | Complete |
| Efficiency | vision, text, pair CSV + manifest | 5 + 4 + 23 profiles | Complete |
| Methodology | protocol JSON, usage table, leakage audit | 3 files | Complete |
| Compositional status | CSV + JSON | 2 optional tasks | Documented missing |
| Diagnostic negatives | 2 combined files, manifest, sidecars, shards | 842,343 canonical rows | Complete/non-trainable |
| Training negatives | 2 canonical Parquet, 8 shards, caches and metadata | 22,968,656 canonical rows | Complete/trainable |
| Expert selection | YAML, ranked CSVs, JSON and Markdown | 14 files | Complete |
| Readiness | CSV, JSON and Markdown | 23 checks | **READY** |

### Results directory by file type

| Type | Count |
|---|---:|
| CSV | 794 |
| JSON | 28 |
| JSONL | 374 |
| PyTorch `.pt` caches under results | 336 |
| Parquet | 10 |
| PNG | 18 |
| Other | 80 |
| **Total** | **1,640** |

The 806 readable delimited outputs counted during validation are the 794 plain CSV files plus 12 compressed CSV.GZ files.

## 7. Checkpoint and log completion

### Checkpoints

The checkpoint root contains 103 top-level experiment directories:

- 100 historical labelled matrix directories: 5 vision × 5 text labels × 4 variants;
- 2 ConvNeXt finalist directories;
- 1 seed-sweep container with 20 nested runs.

Across all levels, the repository contains:

| Checkpoint artifact | Count |
|---|---:|
| `best.pt` | 122 |
| `final.pt` | 122 |
| `latest.pt` | 91 |
| `metrics.csv` | 122 |
| Seed `config.yaml` | 20 |
| Seed `run_summary.json` | 20 |

The 122 best checkpoints correspond to 100 labelled grid runs, two finalists and 20 seed-sweep runs. All 23 checkpoints referenced by each multi-task manifest exist, and all six checkpoints in the final shortlist exist.

### Logs

| Log area | Run directories | Files | Interpretation |
|---|---:|---:|---|
| Encoder matrix | 6 | 202 | Multiple resumable waves retained |
| Multi-task | 6 | 58 | Historical, BLF-only and development waves retained |
| Seed sweep | 2 | 39 | Earlier and resumed waves retained |
| Phase 1.5 hard-negative mining | 4 directories including parent | 12 | Interrupted/resumed and final Step 16 runs retained |
| Step 16 Slurm stdout/stderr | — | 2 | Final batch output; stderr is empty |
| **Total** | — | **313** | Historical logs preserved |

Multiple timestamped log directories do not mean completed experiments were overwritten. The coordinators intentionally create a new run directory for each launch while reading canonical manifests/shards to skip or resume completed work. The final Step 16 coordinator run ID is `1784641226_2411140` and its status is `complete`.

## 8. Most important scientific findings

### Finding 1 — Encoder choice dominates the average result

DINOv2 and MiniLM are the strongest vision and text encoders on average in the canonical retrieval grid. ConvNeXt/ConvNeXtV2 form the next vision tier. EfficientNet is substantially weaker in raw accuracy but provides the lowest-cost path.

### Finding 2 — BLF is lightweight and often helps screening retrieval, but is not universally beneficial

The best BLF option improves 16 of 20 encoder pairs in the single-seed screen. The average variant improvement is small, BLF is worse for four pairs, and global-only BLF has slightly lower average i2t R@1 than baseline. BLF should be described as conditional rather than universally beneficial.

### Finding 3 — The two targeted repeated-seed BLF gains remain inconclusive

DINOv2 local and ConvNeXtV2 local+global both have positive mean matched gains, but both confidence intervals include zero. This is why the final primary shortlist uses the supported DINOv2 + MiniLM baseline and avoids redundant BLF duplicates.

### Finding 4 — BLF is more consistently retrieval-focused than classification-transferable

All three Step 11.5 BLFs improve both COCO directions. On frozen development classification splits, six of nine BLF changes are negative, including large Pets and EuroSAT declines for some variants. BLF retrieval strength should not be generalised to zero-shot classification.

### Finding 5 — Error diversity is large enough to motivate multiple experts

Full-pool oracle accuracy is far above the best individual on every classification dataset, and the best retrieval pair oracle reaches 31.20% vs 19.52% for the best individual. These are upper bounds, but they demonstrate non-identical success sets and justify investigating a learned selector.

### Finding 6 — Different downstream tasks prefer different paths

Retrieval rankings correlate strongly with CIFAR but only moderately/weakly with Pets and EuroSAT. Pets versus EuroSAT Spearman correlation is only 0.228. The best retrieval, CIFAR, Pets and EuroSAT configurations are not the same.

### Finding 7 — A mixed performance/efficiency pool is justified

The final primaries deliberately combine:

- two high-performance DINOv2 semantic paths;
- a faster ConvNeXt + MiniLM path;
- a very low-cost EfficientNet + BGE path.

This pool covers three vision families and two text encoders while preserving measured error diversity.

### Finding 8 — The repository is methodologically ready for Phase 2

The development protocol is frozen, predictions are complete, selected checkpoints exist, efficiency costs are measured, training negatives are leakage-safe and hash-verified, and the final selection file is canonical. The evidence gate is genuinely ready even though the Phase 2 method itself remains future work.

## 9. Remaining limitations and work not yet completed

1. **Phase 2 is not implemented.** `READY` means prerequisites pass, not that a router or adaptive fusion system exists.
2. **No untouched external retrieval benchmark is installed.** COCO val is development evidence. Flickr30k remains the intended external option if added later.
3. **Winoground and SugarCrepe are missing.** Scorers and optional task registry entries exist, but no dataset or prediction result exists.
4. **BLF reliability is unresolved.** Both repeated-seed BLF comparisons are inconclusive; the DINOv2+BGE BLF was not repeated across matched seeds.
5. **Historical classification results are exploratory.** The newer development splits repair expert-selection leakage, but future final reporting must use reserved test splits only after the Phase 2 method is frozen.
6. **Oracle scores are not deployable.** They assume perfect hindsight and must not be presented as expected learned-router performance.
7. **Absolute Pets performance is weak.** High complementarity on Pets coexists with low individual accuracy.
8. **Current test suite was not freshly executed during this audit.** Source syntax and saved output integrity pass, but a clean test run should be part of the first Phase 2 implementation change.
9. **The notebook retains stale intermediate outputs.** Readers should use the newest standalone result artifacts, especially the final readiness JSON, rather than an earlier displayed `NOT_READY` cell output.

## 10. Authoritative starting point for future Phase 2 work

The correct inputs are:

1. [`results/phase15/expert_selection/selected_experts.yaml`](../results/phase15/expert_selection/selected_experts.yaml) for the selected experts, pair paths, roles, constraints and checkpoint fingerprints;
2. [`results/phase15/training_hard_negatives/schema.json`](../results/phase15/training_hard_negatives/schema.json) for training-negative usage and integrity metadata;
3. `results/phase15/training_hard_negatives/retrieval_train_hard_negatives.parquet` for retrieval candidates;
4. `results/phase15/training_hard_negatives/classification_train_hard_negatives.parquet` for classification candidates;
5. [`configs/evaluation_protocol.yaml`](../configs/evaluation_protocol.yaml) for split governance;
6. [`results/phase15/phase2_readiness/readiness_report.json`](../results/phase15/phase2_readiness/readiness_report.json) for the final gate.

The four primary training paths are:

```text
dinov2_vits14__all_minilm_l6_v2__baseline
dinov2_vits14__bge_small_en__baseline
convnext_tiny__all_minilm_l6_v2__baseline
efficientnet_b0__bge_small_en__baseline
```

The optional ablations are:

```text
convnext_tiny__e5_small_v2__baseline
swin_tiny__e5_small_v2__baseline
```

Any Phase 2 experiment should preserve the current safe-default pattern, train only from allowed training/development rows, freeze the method before reserved final evaluation, and avoid representing the diagnostic or oracle outputs as trainable or deployable performance.

## 11. Final audit verdict

Phase 1 and Phase 1.5 are materially complete. The saved artifacts support the following defensible dissertation statement:

> Lightweight BLF branches can improve retrieval for selected frozen encoder pairs at small parameter cost, but the observed gains are encoder-dependent, do not consistently transfer to zero-shot classification, and are not statistically confirmed in the two matched-seed comparisons. The completed complementarity, efficiency and leakage-safe mining analyses justify a small, diverse baseline expert pool for a future adaptive Phase 2 method.

The repository’s final operational state is:

```text
Phase 1 experiments:       COMPLETE
Phase 1 reliability:       COMPLETE, BLF evidence INCONCLUSIVE
Multi-task development:    COMPLETE
Phase 1.5 analysis:        COMPLETE
Training negatives:        COMPLETE, ZERO PROTECTED-TEST OVERLAP
Expert selection:          COMPLETE
Readiness gate:            READY (23/23)
Phase 2 implementation:    NOT STARTED
```
