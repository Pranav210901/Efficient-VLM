# RTX PRO Seed-Sweep Runbook

## What this run validates

The sweep contains 20 matched jobs: five seeds for baseline vs local BLF on DINOv2 + MiniLM, and the same five seeds for baseline vs local+global BLF on ConvNeXt-Tiny + MiniLM. Each job trains for at most five epochs with batch size 64. Early stopping waits for two non-improving epochs, and model selection always uses image-to-text Recall@1 from `best.pt`.

The worker also evaluates every best checkpoint with all COCO validation captions. Training and checkpoint selection retain the existing one-caption protocol for comparability; the `coco5_*` result columns use 5,000 unique image queries and roughly 25,000 caption queries with multiple positives.

## Preflight on a login or data-mover node

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/rebuild_results.py
```

Do not start training on the data-mover node. The Slurm script prepares `data/val_all_captions.csv` inside the allocation if it is absent.

## Submit four GPUs on one node

The checked-in script requests one node, one controlling Slurm task, 32 CPU cores, 256 GiB RAM, four GPUs, and six hours. The controller starts one Python subprocess per visible GPU and fills GPUs from a 20-job queue, giving five waves of four jobs.

Submit the checked-in batch script to the `teaching` partition with:

```bash
sbatch --partition=teaching scripts/train_seed_sweep_rtxpro.slurm
```

The script already contains `--gres=gpu:4`, but the equivalent fully explicit submission is:

```bash
sbatch --partition=teaching --gres=gpu:4 scripts/train_seed_sweep_rtxpro.slurm
```

For an interactive notebook allocation instead of `sbatch`, request the same resources with:

```bash
srun -p teaching \
  --gres=gpu:4 \
  --cpus-per-task=32 \
  --mem=256G \
  --time=06:00:00 \
  --pty bash
```

Once inside the interactive allocation, start Jupyter there and use the notebook's Step 3 launch cell. Do not issue a nested `srun` from the notebook.

## Monitor and resume

```bash
squeue -u "$USER"
tail -f logs/blf_seed_sweep_<job-id>.out
find logs/seed_sweep -type f -name '*.log' -printf '%T@ %p\n' | sort -nr | head
```

Per-run logs are stored below `logs/seed_sweep/<run-id>/`. Checkpoints are below `checkpoints/seed_sweep/`. The consolidated output is `results/seed_sweep_results.csv`.

The launcher uses `--resume`. If the allocation expires or a node fails, submit the same command again. Completed result rows are skipped, completed checkpoints with a missing row are recovered from `best.pt`, and partial runs continue from `latest.pt`.

## Summarize after completion

```bash
.venv/bin/python scripts/summarize_seed_sweep.py
```

This writes per-variant means/standard deviations and seed-paired BLF-minus-baseline deltas. The default metric is five-caption image-to-text Recall@1. To summarize the original one-caption selection metric:

```bash
.venv/bin/python scripts/summarize_seed_sweep.py \
  --metric i2t_R@1 \
  --summary_output results/seed_sweep_summary_one_caption.csv \
  --paired_output results/seed_sweep_paired_deltas_one_caption.csv
```

Treat BLF as supported only if the paired effect is consistently positive across seeds and the uncertainty interval is practically meaningful. The five-seed interval measures run-to-run variation; a final query-level bootstrap should be added for the selected model before publication.

## Record the results in the experiment notebook

When Jupyter itself is running inside a four-GPU allocation, the notebook's **Step 3: Launch the Matched-Seed Sweep** cell can launch the controller directly without a nested `srun`; change its explicit `RUN_SEED_SWEEP` guard to `True`. After it finishes, run **Step 4: Matched-Seed Batch Results**. That cell reads `results/seed_sweep_results.csv` directly, displays every seed, writes the same summary and paired-delta CSVs, generates a cautious interval-based conclusion, and saves `results/seed_sweep_matched_comparison.png`. No result values need to be copied into the notebook manually.
