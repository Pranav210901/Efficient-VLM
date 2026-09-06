# Operating-point control: config verification receipt

Verified before submission, on CPU, without training.

## What was checked

`build_config` for the queue-free arm (`mt1__b1024__age_0__none__seed_42`) was
diffed key-by-key against the config the runner produces for the shipped M-T1
model itself, `build_job_config(pipeline_m_t1, sensitivity_jobs[0],
"sensitivity")`.

Result: **124 resolved keys, 17 differing.** Every one of the 17 is an
identifier, an output path, the added drift instrumentation, or the queue axis:

| Category | Keys |
|---|---|
| Identifiers | `experiment_id`, `run_id` |
| Output paths | `training.save_dir`, `queue_diagnostics.probe_manifest` |
| Queue axis | `training.queue_mode` |
| Added instrumentation | `queue_diagnostics.*` (6 keys, matching `configs/queue_factorial/base.yaml`) |
| Added provenance | `provenance.*` (6 keys) |

No substantive training key differs. Confirmed identical: `training.lr`
(0.002545584412271571, i.e. the x3 sweep multiplier resolved by the project's
own code rather than restated), `training.applied_lr_scale`, `training.epochs`
(24), `training.batch_size` (1024), `recipe.distillation` (True) and
`distillation.strength` (1.0), `data.image_size` (224), the C4 vision
aggregator (`transformer_128`, 256/8/2/512), the text learned-query aggregator
(128d, 4 heads), the split CSVs, dropout and every projection dimension.

The queue-free arm of this study is therefore M-T1 itself, retrained under a
different seed-to-path mapping, which is what makes the three-arm decomposition
attributable to the queue rather than to the operating point.

## Design notes

- The learning rate is **not** pinned to the Wave 0 value by this study's own
  config. It is resolved by `build_job_config` from the M-T1 pipeline. It
  happens to equal the factorial's pinned value because both use batch 1024
  under the locked recipe, but it is derived, not asserted.
- 15 cells, not 18: the queue-free control does not depend on the age axis, so
  it is trained once per seed and read against both ages.
- The matched stale and queue-free arms from Q02/Q04 are **not** reused. They
  exist only at the Wave 0 operating point and reusing them is precisely the
  confound this study exists to remove.

## Before submitting

`predictions/queue_operating_point.json` is status `DRAFT_AWAITING_AUTHOR_REVIEW`.
The `prediction_gate()` called by the training entry point only checks that
statements are non-empty; it cannot tell a reviewed prediction from an
unreviewed one. Review, edit, set status to `PREREGISTERED`, then submit.
