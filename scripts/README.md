# Script namespace

These are stable execution entry points retained under their historical names.
They are intentionally not moved into the research-facing artifact hierarchy:
existing documentation, Slurm dependencies, and operator commands refer to
their literal paths.

Use [`../experiments/README.md`](../experiments/README.md) to identify the
experiment first, then follow that experiment's config and script references.
Completed outputs belong under [`../artifacts/`](../artifacts/), not here.

Submission scripts only print or submit the explicitly selected experiment.
Project cleanup must never execute them.
