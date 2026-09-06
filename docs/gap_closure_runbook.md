# Dissertation gap-closure runbook

This package closes three bounded weaknesses without changing the selected
student architecture or reopening model selection:

1. residual queue-mechanism identification (15 new training runs);
2. one-shot compositional transfer (SugarCrepe primary, Winoground secondary
   when licensed before freezing);
3. paired-bootstrap uncertainty from already frozen outputs (no training).

All commands start in the repository root:

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
```

For the complete non-interactive SugarCrepe-primary workflow, all stages below
can instead be submitted as one dependency graph:

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm && bash scripts/submit_gap_closure_all.sh
```

This one-command path includes Winoground only when its licensed parquet was
already acquired before the command. Otherwise it freezes SugarCrepe alone;
it never accepts gated terms on the user's behalf.

## 0. Inspect the current state

```bash
bash scripts/gap_closure_status.sh
```

The retrospective bootstrap is already complete. Reproduce it if required:

```bash
bash scripts/run_final_bootstrap.sh
```

Expected report:
`artifacts/07_final_evaluation/uncertainty/report.md`.

## 1. Decide whether Winoground is included, then freeze once

SugarCrepe has already been acquired from the official repository and all
7,511 examples resolve against the installed COCO validation images.

Winoground is gated. To include it, first read and accept its licence at
<https://huggingface.co/datasets/facebook/winoground>, authenticate without
placing a token in a shell command, and acquire the frozen parquet:

```bash
hf auth login
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final \
  acquire-winoground --acknowledge-winoground-license
```

If Winoground access is unavailable, omit that command. SugarCrepe remains a
valid untouched primary evaluation. Once the available task set is final,
freeze it exactly once:

```bash
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final freeze
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final validate
```

Do **not** run `freeze` before deciding about Winoground. The freeze gate
deliberately refuses to add a task after any model score exists.

## 2. Submit the one-shot compositional evaluation

```bash
bash scripts/submit_compositional_final.sh
```

This submits nine inference-only jobs: six student checkpoints and three fixed
references. It contains no optimizer or training command. Monitor with:

```bash
squeue -u "${USER}" | rg 'comp_final|comp_report'
```

Expected report:
`artifacts/07_final_evaluation/compositional/report/report.md`.

## 3. Submit the fresh-capacity closure first

Validate the frozen queue design and inspect the 15-cell roster:

```bash
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism validate
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism jobs
```

Submit only the six missing fresh-capacity cells (1,024 and 4,096 entries,
three seeds each):

```bash
bash scripts/submit_queue_mechanism.sh capacity
```

Monitor and inspect the partial curve:

```bash
squeue -u "${USER}" | rg 'queue_mech'
sed -n '1,220p' results/queue_mechanism/report/report.md
```

The report reuses the existing queue-free, 16,384-entry and 65,536-entry
three-seed controls. It does not retrain them.

## 4. Submit the fixed-capacity mechanism cells

After the capacity array finishes, submit nine cells at capacity 16,384:

```bash
bash scripts/submit_queue_mechanism.sh mechanism
```

The three preregistered arms are:

- frozen-MobileCLIP semantic filtering;
- matched-random filtering with the exact same per-query removal count;
- queue denominator-mass normalization.

All use three seeds and the same fresh-reprojection recipe. The semantic
threshold (`0.222935`) and teacher-cache SHA-256 are checked before submission.
No threshold, weight, capacity, or seed may be changed after results appear.

When complete:

```bash
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism report
sed -n '1,260p' results/queue_mechanism/report/report.md
```

## 5. Verify the package

Run the focused regression suite:

```bash
"${ALIGNMENT_V3_PYTHON}" -m pytest -q \
  tests/test_queue_identification.py \
  tests/test_queue_mechanism.py \
  tests/test_compositional_final.py \
  tests/test_final_bootstrap.py \
  tests/test_final_zero_shot.py
```

Then run the repository-wide suite:

```bash
"${ALIGNMENT_V3_PYTHON}" -m pytest -q
```

In the current cleaned local package this gives 233 passes and two unrelated
legacy Phase 2 failures: the compatibility paths for the quarantined Phase 1.5
frozen snapshot and one Phase 1 multi-task cache are absent. The focused suite
above gives 22 passes. Do not copy those large quarantined assets back merely
to make the historical tests green.

## 6. Dissertation integration rule

Do not add pages speculatively. Once both reports are complete, replace the
current residual-mechanism paragraph and part of the limitations discussion
with:

- the five-point fresh-capacity table;
- the three paired mechanism effects and their preregistered 2pp decision
  threshold;
- the SugarCrepe primary table and optional Winoground secondary result;
- the compact paired-bootstrap interval table.

Keep implementation details and category-level tables in the appendix. The
main text should grow by no more than approximately one page; at the existing
70-page ceiling, remove an equal amount of speculative discussion.
