# Phase 2 execution guide

Phase 2 continues in `notebooks/01_experiment_workflow.ipynb`. The notebook contains bounded smoke tests and results-only reporting. Complete experiments are Slurm-only.

## Interactive smoke check (maximum two hours)

```bash
srun --partition=teaching --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --constraint=fs_weka --pty bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
source .venv/bin/activate
python -m src.phase2.cli.validate_prerequisites --config configs/phase2/prerequisites.yaml
python -m src.phase2.cli.validate_token_interfaces --config configs/phase2/token_interfaces.yaml
python -m src.phase2.cli.smoke_test --config configs/phase2/smoke.yaml
pytest -q tests/phase2
```

In the notebook, leave both Phase 2 controls `False` for results-only viewing. Set `RUN_PHASE2_SMOKE_TESTS=True` only inside the two-hour allocation. The notebook never submits jobs.

The two-GPU DDP smoke test is dependency-gated in the complete pipeline. It can also be submitted separately:

```bash
sbatch slurm/phase2/03b_smoke_ddp.sbatch
```

## Pipeline commands

Validate the submission plan without submitting:

```bash
bash scripts/submit_phase2_pipeline.sh --dry-run --max-total-gpus 8
```

Submit the complete dependency chain:

```bash
bash scripts/submit_phase2_pipeline.sh --max-total-gpus 8
```

Inspect status, diagnose failures, and resume:

```bash
bash scripts/phase2_status.sh
bash scripts/phase2_diagnose_failure.sh results/phase2/slurm/submission_manifest.json
bash scripts/phase2_resume.sh
```

The submission script uses `afterok` for required stages, `afternotok` for per-stage diagnostics, and `afterany` for the final status collector. It enforces the `teaching` partition, at most eight GPUs concurrently, and a maximum wall time of `2-23:59:00`.

## Leakage boundary

Bridge and dense-teacher training use only the Phase 1.5 training hard negatives derived from COCO train2017 and classification training/development partitions. COCO validation, classification development evaluation rows, Winoground, and SugarCrepe are not used as training examples. Compositional training remains `evaluation_only` because no separate compositional training split is registered.
