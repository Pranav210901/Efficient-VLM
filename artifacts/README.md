# Canonical artifact store

This is the organized physical home of completed experiment outputs. The
canonical hierarchy is:

```text
artifacts/
├── 01_foundations/
├── 02_recipe_and_queue/
├── 03_data_and_transfer/
├── 04_efficiency_and_distillation/
├── 05_token_architecture/
└── 06_diagnostics/
```

Within each experiment family, `results/` retains compact metrics, reports,
ledgers, manifests, and evaluation exports. Model weights, tensor caches, and
raw scheduler logs are intentionally absent from the GitHub package. They were
moved into a dated local quarantine rather than deleted.

The repository-level `results/` directory supplies compatibility symlinks to
these compact results. Those links resolve entirely within the submitted tree.
Historical config fields that name `checkpoints/` or `logs/` remain provenance,
not promises that the heavyweight runtime products are redistributed.

Editable experiment inputs remain in `configs/`; runnable entry points remain
in `scripts/`. Their filenames have not been changed.

See [`../experiments/README.md`](../experiments/README.md) for the research
headlines and [`../experiments/layout.yaml`](../experiments/layout.yaml) for the
legacy-to-canonical mapping.

## Bounded-study exclusion

`freezeshift/`, `tokenshift/`, `latency_amendment/`, and
`tokenshift_training/` are intentionally outside this store. They were created
after or protected during the artifact reorganization, and all four now have
completed compact reports. Their paths remain unchanged to preserve frozen
manifests and provenance.

The restored `results/phase15`, `results/phase2`,
`results/phase1_multitask`, and `results/phase1_multitask_development`
compatibility links expose compact copies of the legacy summaries. They no
longer depend on an ignored quarantine path.

## Integrity and rollback

The migration receipt is `layout_receipt.json`. It records directory inodes,
file counts, byte counts, and hashes for configs, fingerprints, metrics,
run summaries, and reports. To inspect the layout:

```bash
python experiments/reorganize_artifacts.py
```

The later GitHub cleanup and its recovery manifest are documented in
[`../docs/github_submission_package.md`](../docs/github_submission_package.md).
