# Efficient Frozen Vision-Language Alignment

This repository studies the dissertation question:

> **Can modern frozen SSL vision encoders, combined with compatibility-based
> pair selection and sub-5M-parameter adaptation, approach compact jointly
> pretrained vision-language models under a fixed inference budget?**

## Start here

1. [`EXPERIMENTS.md`](EXPERIMENTS.md) — one-page entry point.
2. [`docs/dissertation.md`](docs/dissertation.md) — integrated dissertation manuscript.
3. [`docs/experiment_story.md`](docs/experiment_story.md) — how each experiment
   follows from the previous finding.
4. [`docs/completion_status.md`](docs/completion_status.md) — submission-readiness checklist.
5. [`docs/literature_gap_closure_20260810.md`](docs/literature_gap_closure_20260810.md)
   — 25-area comparison, claim boundaries, and irreducible gaps.
6. [`experiments/README.md`](experiments/README.md) — the complete experiment
   story, stable IDs, status, findings, and evidence links.
7. [`artifacts/README.md`](artifacts/README.md) — the organized physical layout
   for compact completed results.
8. [`experiments/registry.yaml`](experiments/registry.yaml) — machine-readable
   experiment register.

The repository-level `results/` directory is a compatibility view into the
canonical compact reports under `artifacts/`. Heavyweight checkpoints and raw
logs are excluded from the GitHub package and remain recoverable from the local
quarantine described in [`docs/github_submission_package.md`](docs/github_submission_package.md).

## Repository layout

```text
alignment_vlm/
├── experiments/   # canonical experiment index, registry, and layout tooling
├── artifacts/     # completed outputs grouped by research question
│   ├── 01_foundations/
│   ├── 02_recipe_and_queue/
│   ├── 03_data_and_transfer/
│   ├── 04_efficiency_and_distillation/
│   ├── 05_token_architecture/
│   └── 06_diagnostics/
├── configs/       # stable executable inputs; deliberately not renamed
├── scripts/       # stable launch, reporting, and maintenance entry points
├── slurm/         # generated Slurm templates grouped by implementation name
├── src/           # model, training, evaluation, and analysis code
├── tests/         # regression and integrity tests
├── docs/          # frozen methodology and protocol documents
├── notebooks/     # result notebooks
├── data/          # acquisition guide; local dataset payloads are excluded
├── research/      # literature and positioning work
├── reports/       # dissertation-facing reports and exports
├── results/       # compatibility links into artifacts/
├── quarantine/    # ignored local recovery store; never submit to GitHub
├── freezeshift/   # completed bounded encoder-adaptation study
├── tokenshift/    # completed profile-only token-merging study
├── latency_amendment/ # completed same-allocation timing repair
└── tokenshift_training/ # completed frozen-vs-LoRA accuracy study
```

The bounded follow-up packages remain outside `artifacts/` to preserve their
already-frozen paths. FreezeShift, TokenShift profiling, the latency amendment,
and TokenShift accuracy training all have closed reports. TokenShift reduced
latency but lost 5.69--15.95pp against its matched no-merge parents, so neither
merge depth was promoted.

## Current completed-study headline

The strongest fully frozen-backbone configuration is M_T1:

- frozen DINOv3 ViT-S/16 vision encoder at 224 px;
- two-block, 256-d learned vision-token transformer aggregation (`C4`);
- frozen all-MiniLM-L6-v2 text encoder;
- 128-d, four-head learned-query text aggregation;
- residual projections into a shared normalized 384-d embedding;
- MobileCLIP2-S0 distillation during training only;
- queue-free InfoNCE with all COCO captions and batch 1024;
- 2,896,389 inference-trainable parameters;
- 54.487% three-seed Flickr30k-validation mean bidirectional R@1;
- 9.148 ms same-allocation mean full-stack Q3 latency on an RTX PRO 6000
  Blackwell, 0.119 ms faster than the local OpenCLIP reference.

The strongest development-selected bounded-adaptation model is dual-tower
FreezeShift: 63.416% validation mean R@1, 4,862,469 trainable inference
parameters, and 9.166 ms same-allocation mean Q3 latency. It is 0.101 ms faster
than OpenCLIP (paired 95% bootstrap CI -0.145 to -0.016 ms) but remains 6.525pp
below OpenCLIP's 69.941% validation mean R@1. The later no-training sealed
evaluation measured 52.90 ± 0.87% for M_T1 and 62.20 ± 0.45% for FreezeShift
on Flickr30k test across three seeds, versus 68.22% for fixed OpenCLIP and
78.25% for fixed MobileCLIP2-S0 references.
Its 49.163M total deployed parameters are 32.5% of OpenCLIP ViT-B/32's 151.277M,
which supports an “about one third” size comparison only against that field
anchor. MobileCLIP2-S0 is the primary compact reference; FreezeShift is 65.7% of
its total size and retains 79.49% of its split-matched test retrieval score.

The completed TokenShift follow-up confirms that the profile-only speed gain
does not improve the accuracy--latency frontier in this implementation. The
late block-8 merge was the least damaging arm but still lost 12.581pp for the
strictly frozen parent and 5.690pp for the dual-LoRA parent.

## Integrity policy

- Existing config, script, result, and report names remain stable API.
- Compact result compatibility links must resolve without the ignored local
  quarantine.
- Heavyweight runtime material is moved recoverably and recorded in a manifest,
  never silently deleted.
- No stopped or negative branch is removed or relabelled.

Validate the experiment register and artifact layout with:

```bash
python experiments/check_registry.py
python -m pytest
```

The original artifact migration receipt remains at
[`artifacts/layout_receipt.json`](artifacts/layout_receipt.json); the later
GitHub packaging policy is documented separately.

## Environment

The cluster runtime is resolved by the repository's AISurrey Python wrapper.
The direct dependency specifications are retained in `requirements.txt` and
`requirements-lock.txt`. Historical setup and execution instructions remain
in the versioned methodology files under `docs/`; they are not the primary
navigation layer for the current project.
