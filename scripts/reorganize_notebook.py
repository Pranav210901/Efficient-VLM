"""One-time, atomic reorganisation of the primary experiment notebook.

Uses the notebook JSON schema directly because nbformat is intentionally not a
runtime project dependency. Existing metadata and the historically important
cell IDs are preserved.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/01_experiment_workflow.ipynb"


def cell(kind: str, source: str, identifier: str) -> dict:
    value = {"cell_type": kind, "id": identifier, "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}
    previous = by_id.get(identifier)
    if previous is not None:
        value["metadata"] = previous.get("metadata", {})
    if kind == "code":
        unchanged = previous is not None and previous.get("cell_type") == kind and previous.get("source") == value["source"]
        value.update(
            {
                "execution_count": previous.get("execution_count") if unchanged else None,
                "outputs": previous.get("outputs", []) if unchanged else [],
            }
        )
    return value


old = json.loads(PATH.read_text())
old_cells = old["cells"]
by_id = {value.get("id"): value for value in old_cells}

setup = '''from pathlib import Path
import os, subprocess, sys
import torch

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks": PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
PYTHON = sys.executable
CONFIG = "configs/coco_blf_rtxpro.yaml"
RESULTS_CSV = "results/alignment_matrix_results.csv"
VISION_ENCODERS = ["efficientnet_b0", "convnext_tiny"]
TEXT_ENCODERS = ["minilm_l6", "distilbert"]
EXTENDED_VISION_ENCODERS = ["convnextv2_tiny", "dinov2_vits14", "swin_tiny"]
EXTENDED_TEXT_ENCODERS = ["all_minilm_l6_v2", "bge_small_en", "e5_small_v2"]
FULL_VISION_ENCODERS = ["efficientnet_b0", "convnext_tiny", "convnextv2_tiny", "dinov2_vits14", "swin_tiny"]
FULL_TEXT_ENCODERS = ["minilm_l6", "all_minilm_l6_v2", "bge_small_en", "e5_small_v2", "distilbert"]
GPU_IDS = [str(i) for i in range(torch.cuda.device_count())]
RUN_PHASE1_TRAINING = False
RUN_SEED_SWEEP = False
RUN_QUALITATIVE_RETRIEVAL = False
RUN_MULTITASK_EVALUATION = False
RUN_PHASE15 = False
RUN_MODE = "smoke"  # "smoke" or "full"
RESULTS_ONLY = True  # Default: never launch training/evaluation; only display saved outputs.

print("Project root:", PROJECT_ROOT)
print("Python:", PYTHON)
print("Configuration:", PROJECT_ROOT / CONFIG)
print("Visible CUDA devices:", torch.cuda.device_count())
print("GPU names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("GPU IDs:", GPU_IDS)
print("Switches:", {"results_only": RESULTS_ONLY, "training": RUN_PHASE1_TRAINING, "seed_sweep": RUN_SEED_SWEEP, "qualitative_retrieval": RUN_QUALITATIVE_RETRIEVAL, "multitask": RUN_MULTITASK_EVALUATION, "phase15": RUN_PHASE15})
print("Run mode:", RUN_MODE)'''

# Reorganising the notebook must not silently change the user's active run
# controls. Preserve only these explicit assignments; all other setup source
# continues to come from the canonical template above.
previous_setup = "".join(by_id.get("step-0-controls", {}).get("source", []))
for control in ("RUN_PHASE1_TRAINING", "RUN_SEED_SWEEP", "RUN_QUALITATIVE_RETRIEVAL", "RUN_MULTITASK_EVALUATION", "RUN_PHASE15", "RUN_MODE", "RESULTS_ONLY"):
    match = re.search(rf"^{control}\s*=.*$", previous_setup, flags=re.MULTILINE)
    if match:
        setup = re.sub(rf"^{control}\s*=.*$", match.group(0), setup, flags=re.MULTILINE)

train_baseline = '''from src.multitask.config import canonical_text_encoder
canonical_texts = list(dict.fromkeys(canonical_text_encoder(name) for name in FULL_TEXT_ENCODERS))
expected = len(FULL_VISION_ENCODERS) * len(canonical_texts)
print(f"Canonical baseline experiments: {expected} (MiniLM aliases deduplicated)")
cmd = [PYTHON, "scripts/run_encoder_matrix.py", "--config", CONFIG, "--mode", "baseline", "--results_path", RESULTS_CSV, "--vision_encoders", *FULL_VISION_ENCODERS, "--text_encoders", *canonical_texts, "--resume"]
if GPU_IDS: cmd += ["--gpus", *GPU_IDS]
if RUN_PHASE1_TRAINING and not RESULTS_ONLY:
    if not GPU_IDS: raise RuntimeError("No visible GPU; baseline training was not launched")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
else: print("Results-only: existing baseline experiments will not be rerun.")'''

baseline_eval = '''import pandas as pd
clean_path = PROJECT_ROOT / "results/alignment_matrix_baseline_dedup_best.csv"
canonical_path = PROJECT_ROOT / "results/alignment_matrix_best_clean.csv"
source_path = canonical_path if canonical_path.exists() else clean_path
if not source_path.exists(): raise FileNotFoundError("No saved clean result table exists. Rebuild it explicitly with scripts/rebuild_results.py.")
clean = pd.read_csv(source_path)
baseline = clean[(clean.use_local_blf == False) & (clean.use_global_blf == False)].copy()
baseline = baseline[baseline.text_encoder != "minilm_l6"].drop_duplicates(["vision_encoder", "text_encoder"])
display(baseline)
display(baseline.loc[baseline.groupby("vision_encoder")["mean_R@1"].idxmax()])
display(baseline.loc[baseline.groupby("text_encoder")["mean_R@1"].idxmax()])
display(baseline.sort_values("mean_R@1", ascending=False))
print("Loaded baseline results from:", source_path)'''

train_blf = '''canonical_texts = list(dict.fromkeys(canonical_text_encoder(name) for name in FULL_TEXT_ENCODERS))
expected = len(FULL_VISION_ENCODERS) * len(canonical_texts) * 3
print(f"Canonical BLF experiments: {expected}")
cmd = [PYTHON, "scripts/run_encoder_matrix.py", "--config", CONFIG, "--mode", "blf", "--results_path", RESULTS_CSV, "--vision_encoders", *FULL_VISION_ENCODERS, "--text_encoders", *canonical_texts, "--resume"]
if GPU_IDS: cmd += ["--gpus", *GPU_IDS]
if RUN_PHASE1_TRAINING and not RESULTS_ONLY:
    if not GPU_IDS: raise RuntimeError("No visible GPU; BLF training was not launched")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
else: print("Results-only: existing BLF experiments will not be rerun.")'''

finalist_code = '''import pandas as pd

FINALIST_RESULTS_CSV = PROJECT_ROOT / "results/finalist_convnext_tiny_minilm_l6_15ep.csv"
finalist_cmd = [
    PYTHON, "scripts/run_finalist_comparison.py",
    "--config", CONFIG,
    "--epochs", "15",
    "--results_path", str(FINALIST_RESULTS_CSV.relative_to(PROJECT_ROOT)),
    "--gpus", *GPU_IDS,
    "--resume",
]
print("Finalist jobs: 2; visible GPUs:", len(GPU_IDS))
print("Command:", " ".join(finalist_cmd))
if RUN_PHASE1_TRAINING and not RESULTS_ONLY:
    if not GPU_IDS: raise RuntimeError("No visible GPU; finalist training was not launched")
    subprocess.run(finalist_cmd, cwd=PROJECT_ROOT, check=True)
else:
    print("Results-only: historical finalist jobs will not be rerun.")

if FINALIST_RESULTS_CSV.exists():
    finalist_summary = pd.read_csv(FINALIST_RESULTS_CSV)
    display(finalist_summary)
    print("Loaded finalist results:", FINALIST_RESULTS_CSV)
else:
    print("No saved finalist result table exists yet:", FINALIST_RESULTS_CSV)'''

cells = [
cell("markdown", "# Phase 1 and Phase 1.5 Vision–Language Experiment Workflow\n\nResearch question: **Can multiple lightweight pretrained vision and text encoders be adaptively combined across multiple vision-language tasks to achieve a better performance–efficiency trade-off than a single fixed encoder pair?**\n\nPhase 1/1.5 keeps pretrained VE and TE backbones frozen. Cross-attention, routing, adaptive fusion, and unified controllers are out of scope.", "phase1-title"),
cell("markdown", "## Step 0 — Project Setup and Execution Controls\n\nThis step defines the project paths, encoder lists, visible GPUs, and central execution switches. Expensive work is disabled by default so **Run All** safely displays existing results without launching experiments.", "step-0-md"), cell("code", setup, "step-0-controls"),
cell("markdown", "## Step 1 — Prepare and Validate COCO Captions\n\nThis step checks the COCO image directories, caption annotations, and prepared train/validation CSV files. It reports sample counts, duplicates, null values, missing files, and train–validation overlap without redownloading valid data.", "prepare-coco-md"),
cell("code", '''from src.multitask.datasets import validate_coco
report = validate_coco(PROJECT_ROOT)
display(report)
if not report["ready"]:
    print("Missing:", report["missing"])
    print("If archives already exist, run the next command from this notebook cell only.")
    if RUN_PHASE1_TRAINING: subprocess.run([PYTHON, "scripts/prepare_coco.py"], cwd=PROJECT_ROOT, check=True)
else: print("COCO is valid; no download or rebuild was performed.")''', "prepare-coco-code"),
cell("markdown", """## Step 2 — Train Baseline Pretrained VE–TE Pairs

This step defines the complete baseline grid of frozen pretrained vision and text encoders. It resumes incomplete jobs and skips completed configurations, and only launches training when the Phase 1 training switch is explicitly enabled. Multi-GPU execution uses **one independent configuration per GPU**; it increases experiment throughput but does not shorten an individual model run.

| Visible GPUs | Concurrent jobs | Expected trade-off |
|---:|---:|---|
| 1 | 1 | Lowest resource use and simplest debugging, but the longest total matrix wall time. |
| 2 | 2 | Roughly doubles throughput when at least two jobs are pending, with twice the GPU allocation. |
| 4 | 4 | Strong balance for the 20-job baseline grid; possible shared data and model-cache I/O contention. |
| 8 | 8 | Highest matrix throughput when eight jobs are pending, but requires a larger allocation and can increase storage or download contention. |""", "step-2-md"), cell("code", train_baseline, "step-2-train-baseline"),
cell("markdown", "## Step 3 — Evaluate and Clean Baseline Results\n\nThis step loads the authoritative best-epoch baseline results, removes MiniLM aliases and repeated rows, and displays the full ranking plus the best pair for each vision and text encoder.", "step-3-md"), cell("code", baseline_eval, "step-3-baseline-eval"),
cell("markdown", """## Step 4 — Train BLF Variants

This step applies the local, global, and local+global BLF variants across the canonical encoder grid. Completed runs are skipped, interrupted runs can resume, and no training starts unless explicitly enabled. Each GPU receives one independent BLF configuration, so more GPUs reduce total grid time rather than the epoch time of one configuration.

| Visible GPUs | Concurrent jobs | Expected trade-off |
|---:|---:|---|
| 1 | 1 | Lowest allocation cost, but all BLF variants run sequentially. |
| 2 | 2 | Approximately twice the job throughput while keeping scheduling and I/O pressure modest. |
| 4 | 4 | Good throughput for the large BLF grid, with moderate concurrent data loading and checkpoint writes. |
| 8 | 8 | Fastest completion when enough jobs remain, but highest allocation cost and greatest shared-filesystem contention. |""", "step-4-md"), cell("code", train_blf, "step-4-train-blf"),
cell("markdown", "## Step 5 — Build the Canonical Phase 1 Retrieval Table\n\nThis step loads one best-epoch row per genuine configuration and verifies that no duplicate encoder–variant combinations remain. It then displays the clean retrieval table and alignment matrices for the available COCO metrics.", "step-5-md"),
cell("code", '''canonical = PROJECT_ROOT / "results/alignment_matrix_best_clean.csv"
if not canonical.exists(): raise FileNotFoundError(f"Saved canonical table missing: {canonical}")
matrix = pd.read_csv(canonical)
keys = ["vision_encoder", "text_encoder", "use_local_blf", "use_global_blf"]
assert not matrix.duplicated(keys).any(), "Duplicate genuine configurations remain"
display(matrix)
for metric in [value for value in ("coco5_i2t_R@1", "coco5_t2i_R@1", "i2t_R@1", "t2i_R@1") if value in matrix]:
    display(matrix.pivot_table(index="vision_encoder", columns=["text_encoder", "variant"], values=metric))
print("Canonical retrieval table:", canonical)''', "step-5-retrieval-table"),
cell("markdown", """## Step 6 — Historical Finalist Check: ConvNeXt Baseline vs Local+Global BLF

This step reports the historical 15-epoch ConvNeXt-Tiny and MiniLM baseline-versus-BLF comparison using best checkpoints. It is a single-seed diagnostic only; the matched-seed study in Steps 7–8 supersedes its conclusion. There are only two independent jobs, so this section can use at most two GPUs effectively.

| Visible GPUs | Concurrent jobs | Expected trade-off |
|---:|---:|---|
| 1 | 1 | Runs baseline and BLF sequentially; lowest allocation cost but roughly twice the total wall time. |
| 2 | 2 | Runs both finalists concurrently and gives the best wall-time/resource balance for this section. |
| 4 | 2 | No speedup beyond two GPUs because only two jobs exist; two GPUs remain idle. |
| 8 | 2 | No additional speedup; six GPUs remain idle, so this is inefficient for the finalist check alone. |""", "step-6-md"),
cell("code", finalist_code, "c25224ba-02f9-43c8-9496-81b0d2a2e0d1"),
by_id["launch-seed-sweep-on-allocated-node"],
by_id["launch-seed-sweep-directly"],
by_id["batch-seed-sweep-results"],
by_id["batch-seed-sweep-analysis"],
cell("markdown", """### Step 8A — Qualitative COCO Retrieval for the Two Candidate Models

This subsection compares the two BLF candidate encoder pairs from Step 8 at the same predeclared seed 42: **DINOv2 + MiniLM + local BLF** and **ConvNeXtV2 + MiniLM + local+global BLF**. It uses three captions selected deterministically with seed 42 and ranks each caption against all 5,000 COCO validation images, then displays the ground-truth image and each model's top-five retrieved images side by side.

COCO's public captioned validation split is disjoint from the training images and acts as the held-out gallery here; it is not the official hidden COCO test server. The montage is qualitative and does not establish superiority, particularly because the Step 8 paired confidence intervals cross zero. The two models run sequentially on one GPU and reuse metadata-validated gallery caches on later runs. Checkpoint weights are local, but MiniLM's tokenizer/configuration must already be cached on the node or be fetched once.""", "step-8a-qualitative-md"),
cell("code", '''from src.multitask.qualitative_retrieval import (
    QualitativeModel,
    display_qualitative_retrieval,
    load_qualitative_retrieval,
    run_qualitative_retrieval,
)

QUALITATIVE_OUTPUT = PROJECT_ROOT / "results/qualitative_retrieval"
QUALITATIVE_MODELS = [
    QualitativeModel(
        label="DINOv2 + MiniLM + local BLF (seed 42)",
        vision_encoder="dinov2_vits14",
        text_encoder="all_minilm_l6_v2",
        variant="local",
        checkpoint="checkpoints/seed_sweep/dinov2_minilm_local_local_seed42/best.pt",
    ),
    QualitativeModel(
        label="ConvNeXtV2 + MiniLM + local+global BLF (seed 42)",
        vision_encoder="convnextv2_tiny",
        text_encoder="all_minilm_l6_v2",
        variant="local_global",
        checkpoint="checkpoints/seed_sweep/convnextv2_minilm_local_global_local_global_seed42/best.pt",
    ),
]

if RUN_QUALITATIVE_RETRIEVAL and not RESULTS_ONLY:
    if not GPU_IDS:
        raise RuntimeError("No visible GPU; qualitative retrieval was not launched")
    qualitative_retrieval = run_qualitative_retrieval(
        PROJECT_ROOT,
        QUALITATIVE_MODELS,
        device="cuda:0",
        num_queries=3,
        topk=5,
        query_seed=42,
        max_images=None,
        output_dir=QUALITATIVE_OUTPUT,
    )
else:
    qualitative_retrieval = load_qualitative_retrieval(QUALITATIVE_OUTPUT)

if qualitative_retrieval is None:
    print("No saved qualitative retrieval exists. Set RUN_QUALITATIVE_RETRIEVAL=True and RESULTS_ONLY=False, then rerun this cell.")
else:
    display(qualitative_retrieval["summary"])
    display_qualitative_retrieval(qualitative_retrieval["rows"], PROJECT_ROOT)
    print("Qualitative outputs:", QUALITATIVE_OUTPUT)
    print("These three deterministic queries are illustrative, not a replacement for the full Step 8 statistics.")''', "step-8a-qualitative-code"),
cell("markdown", "## Step 9 — Validate Retrieval and Zero-Shot Evaluation Datasets\n\nThis step validates the selected evaluation scope: COCO retrieval plus CIFAR-100, Oxford-IIIT Pet, and EuroSAT zero-shot classification. It opens the three torchvision datasets with `download=False`, reports exact sample and class counts, and does not contact the web. CIFAR-100 and Pets use their test splits; torchvision exposes EuroSAT as one complete 27,000-image dataset with no official test split.", "step-9-md"),
cell("code", '''import pandas as pd
from src.multitask.datasets import validate_datasets

MULTITASK_DATASETS = ["coco_retrieval", "cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"]
dataset_status = pd.DataFrame(validate_datasets(PROJECT_ROOT, MULTITASK_DATASETS))
display(dataset_status)
if not dataset_status.status.eq("ready").all():
    print("Not ready:")
    display(dataset_status.loc[dataset_status.status != "ready"])
else:
    print("All selected datasets are ready; validation used download=False.")''', "step-9-datasets"),
cell("markdown", "## Step 10 — Retrieval and Zero-Shot Smoke Test\n\nThis step runs the focused evaluator tests and, in smoke mode, evaluates one best checkpoint on 8 COCO images and 32 examples from each zero-shot dataset. It verifies offline dataset loading, preprocessing, prompt ensembling, frozen checkpoint loading, task metrics, result writing, and cache reuse before the full run.", "step-10-md"),
cell("code", '''if not RESULTS_ONLY:
    subprocess.run([PYTHON, "-m", "pytest", "-q", "tests/multitask", "tests/phase15"], cwd=PROJECT_ROOT, check=True)
from src.multitask.runner import run_multitask_smoke_test
if RUN_MULTITASK_EVALUATION and RUN_MODE == "smoke" and not RESULTS_ONLY:
    smoke = run_multitask_smoke_test(PROJECT_ROOT, MULTITASK_DATASETS, device="cuda" if GPU_IDS else "cpu")
    display(smoke["manifest"]); display(smoke["results"])
else:
    saved = PROJECT_ROOT / "results/phase1_multitask/task_results_long.csv"
    if saved.exists(): display(pd.read_csv(saved)); print("Loaded saved smoke/full evaluation results:", saved)
    else: print("Results-only: no saved retrieval or zero-shot evaluation exists yet; no model was run.")''', "step-10-smoke"),
cell("markdown", """## Step 11 — Full Retrieval and Zero-Shot Evaluation

This step evaluates every selected baseline checkpoint on COCO retrieval and the three zero-shot classification datasets while keeping both pretrained backbones frozen. One independent model configuration runs per GPU; partial runs reuse valid caches, completed configurations are skipped, and every task keeps its own metrics, per-class table, and sample-level predictions. Checkpoint weights are loaded locally; each Hugging Face text encoder still requires its tokenizer and model configuration in the node's cache, or one-time web access if those small files are absent.

| Visible GPUs | Concurrent configurations | Expected trade-off |
|---:|---:|---|
| 1 | 1 | Lowest allocation and simplest diagnosis, but all model pairs run sequentially. |
| 2 | 2 | Roughly doubles configuration throughput with modest shared-filesystem pressure. |
| 4 | 4 | Good balance for the 20 baseline pairs; four datasets are evaluated sequentially inside each worker. |
| 8 | 8 | Highest throughput while at least eight pairs remain, but also the greatest data-loading and cache-write contention. |""", "step-11-md"),
cell("code", '''import pandas as pd

MULTITASK_CONFIG_SCOPE = "baseline_all_pairs"
preflight = pd.DataFrame(validate_datasets(PROJECT_ROOT, MULTITASK_DATASETS))
if RUN_MULTITASK_EVALUATION and RUN_MODE == "full" and not RESULTS_ONLY:
    if not GPU_IDS: raise RuntimeError("No visible GPU; multi-task evaluation was not launched")
    if not preflight.status.eq("ready").all():
        display(preflight.loc[preflight.status != "ready"])
        raise RuntimeError("Selected evaluation datasets are not all ready")
    multitask_cmd = [PYTHON, "scripts/run_multitask_jobs.py", "--mode", "full", "--config_scope", MULTITASK_CONFIG_SCOPE, "--datasets", *MULTITASK_DATASETS, "--gpus", *GPU_IDS, "--resume"]
    print("Multi-GPU evaluation command:", " ".join(multitask_cmd))
    try:
        subprocess.run(multitask_cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError:
        logs = sorted((PROJECT_ROOT / "logs/multitask").glob("*/*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        print("Multi-task evaluation failed. Latest worker logs:")
        for log in logs[:len(GPU_IDS)]:
            print(" -", log)
            tail = log.read_text(errors="replace").splitlines()[-20:]
            if tail: print("\\n".join(tail))
        raise
    saved = PROJECT_ROOT / "results/phase1_multitask/task_results_long.csv"
    display(pd.read_csv(saved)); print("Outputs:", saved.parent)
else:
    saved = PROJECT_ROOT / "results/phase1_multitask/task_results_long.csv"
    if saved.exists(): display(pd.read_csv(saved)); print("Loaded saved multi-task results:", saved)
    else: print("No saved retrieval or zero-shot results yet. Training will not rerun; one evaluation pass is required to create:", saved)''', "step-11-evaluate"),
cell("markdown", "## Step 12 — Retrieval and Zero-Shot Findings\n\nThis step reads the saved outputs without training or evaluating a model. It displays the best configuration for each task's headline metric and the complete task-specific results. Retrieval recall and classification accuracy are not averaged together because they measure different outcomes.", "step-12-md"),
cell("code", '''from src.multitask.reporting import write_multitask_summary
results_long = PROJECT_ROOT / "results/phase1_multitask/task_results_long.csv"
if results_long.exists():
    if not RESULTS_ONLY:
        summary_paths = write_multitask_summary(results_long.parent); print("Summary:", summary_paths)
    task_results = pd.read_csv(results_long)
    headline_metrics = ["coco5_i2t_R@1", "top1_accuracy"]
    headline = task_results[task_results.metric.isin(headline_metrics)]
    if not headline.empty:
        best = headline.loc[headline.groupby(["task", "metric"])["value"].idxmax()]
        display(best.sort_values(["task", "metric"]))
    display(task_results.sort_values(["task", "metric", "value"], ascending=[True, True, False]))
    print("Loaded saved findings from:", results_long)
else: print("Step 11 output missing:", results_long)''', "step-12-findings"),
cell("markdown", "## Step 13 — Phase 1.5: Export Sample-Level Predictions\n\nThis step collects query-level ranks, scores, margins, predictions, and correctness indicators from every evaluated task. The cache-safe exports support later complementarity, oracle, hard-negative, and routing research without recomputing valid results.", "step-13-md"),
cell("code", '''prediction_root = PROJECT_ROOT / "results/phase1_multitask/predictions"
if RUN_PHASE15 and not RESULTS_ONLY:
    files = sorted(prediction_root.glob("*.jsonl")); print(f"Found {len(files)} cache-safe sample-level exports", prediction_root)
    if not files: raise FileNotFoundError("Run Step 11 first; its evaluator writes resumable sample-level exports.")
else: print("Skipped: set RUN_PHASE15=True after Step 11.")''', "step-13-export"),
cell("markdown", "## Step 14 — Phase 1.5: Complementarity and Oracle Analysis\n\nThis step compares experts at the sample level to measure agreement, error overlap, unique wins, margin or rank correlations, and oracle gains. Oracle scores are reported only as non-deployable diagnostic upper bounds.", "step-14-md"),
cell("code", '''from src.phase15.runner import run_phase15
VISION_ANCHOR_TEXT = "all_minilm_l6_v2"
TEXT_ANCHOR_VISION = "dinov2_vits14"
if RUN_PHASE15 and not RESULTS_ONLY:
    phase15 = run_phase15(PROJECT_ROOT, VISION_ANCHOR_TEXT, TEXT_ANCHOR_VISION); display(phase15)
else: print("Skipped: oracle values will be labelled non-deployable diagnostic upper bounds.")''', "step-14-complementarity"),
cell("markdown", "## Step 15 — Phase 1.5: Encoder Efficiency Profiling\n\nThis step profiles every frozen vision and text encoder on the same visible hardware. It records parameters, latency, throughput, GPU memory, dimensions, token counts, and FLOPs or MACs when supported.", "step-15-md"),
cell("code", '''from src.phase15.efficiency_profile import profile_callable
efficiency_root = PROJECT_ROOT / "results/phase15/efficiency"
if RUN_PHASE15 and not RESULTS_ONLY:
    efficiency_root.mkdir(parents=True, exist_ok=True)
    print("Profile frozen encoders on the same visible hardware; unsupported FLOPs are recorded as unavailable.")
    print("Outputs:", efficiency_root / "vision_encoders.csv", efficiency_root / "text_encoders.csv", efficiency_root / "full_pairs.csv")
else: print("Skipped: set RUN_PHASE15=True.")''', "step-15-efficiency"),
cell("markdown", "## Step 16 — Phase 1.5: Mine Hard Negatives for Phase 2\n\nThis step extracts the strongest incorrect retrieval candidates, class prompts, and compositional alternatives from reliable dual encoders. Valid captions belonging to the same COCO image are excluded so they are not treated as false negatives.", "step-16-md"),
cell("code", '''HARD_NEGATIVES_PER_POSITIVE = 8
hard_root = PROJECT_ROOT / "results/phase15/hard_negatives"
if RUN_PHASE15 and not RESULTS_ONLY:
    hard_root.mkdir(parents=True, exist_ok=True)
    print("Mining uses Step 13 predictions and excludes every caption owned by the query image.")
    print("Outputs:", hard_root)
else: print("Skipped: set RUN_PHASE15=True.")''', "step-16-hard-negatives"),
cell("markdown", "## Step 17 — Phase 1.5: Select Pretrained Experts for Phase 2\n\nThis step ranks a compact expert pool using task performance, unique contribution, error diversity, seed reliability, efficiency, and architectural coverage. It produces the recommended vision experts, text experts, fallbacks, and VE–TE shortlist without implementing Phase 2.", "step-17-md"),
cell("code", '''selection_root = PROJECT_ROOT / "results/phase15/expert_selection"
required = [PROJECT_ROOT / "results/phase15/phase15_summary.json", PROJECT_ROOT / "results/phase15/efficiency/full_pairs.csv"]
if RUN_PHASE15 and not RESULTS_ONLY and all(path.exists() for path in required):
    print("Ready to rank up to three primary VEs/TEs plus optional cheap fallbacks using performance, complementarity, reliability, and cost.")
    print("Outputs:", selection_root)
else: print("Expert selection waits for:", [str(path) for path in required if not path.exists()])
print("Phase 2 cross-attention and routing are intentionally not implemented.")''', "step-17-selection"),
cell("markdown", "# Utility Appendix\n\nOptional and debugging utilities live here so they do not interrupt the numbered workflow.", "utility-appendix"),
cell("markdown", "## Extract Embeddings for One Configuration", "utility-extract-md"), by_id.get("2b2d0ece", cell("code", "print('No legacy one-off extraction cell was available.')", "utility-extract-code")),
]

# Make the historical cells safe and correctly numbered while preserving their IDs.
cells[15]["source"] = [line + "\n" for line in """## Step 7 — Launch Matched-Seed Reliability Experiments

This step runs matched baseline and BLF configurations across multiple random seeds to test whether observed retrieval gains are reliable. It assigns one independent seed/configuration job to each visible GPU, resumes interrupted jobs, skips completed runs, and remains disabled by default.

| Visible GPUs | Concurrent jobs | Expected trade-off |
|---:|---:|---|
| 1 | 1 | Lowest allocation cost and easiest failure diagnosis, but all 20 jobs run sequentially. |
| 2 | 2 | Approximately halves ideal sweep wall time with modest resource and I/O pressure. |
| 4 | 4 | Good balance for 20 jobs: approximately five scheduling waves when runtimes are similar. |
| 8 | 8 | Highest sweep throughput, but jobs complete in uneven waves and shared data/checkpoint I/O may become the bottleneck. |""".splitlines()]
seed_source = "".join(cells[16]["source"]).replace("RUN_SEED_SWEEP = True", "RUN_SEED_SWEEP = globals().get('RUN_SEED_SWEEP', False)")
seed_source = seed_source.replace("if RUN_SEED_SWEEP:", "if RUN_SEED_SWEEP and not RESULTS_ONLY:")
seed_source = seed_source.replace("SEED_SWEEP_GPU_IDS = [str(index) for index in range(2)]", "SEED_SWEEP_GPU_IDS = list(GPU_IDS) if 'GPU_IDS' in globals() else [str(index) for index in range(torch.cuda.device_count())]")
seed_source = seed_source.replace("inside the 4-GPU allocation", "inside the GPU allocation")
seed_source = seed_source.replace("show eight devices", "show the intended visible devices")
cells[16]["source"] = [line + "\n" for line in seed_source.splitlines()]
cells[17]["source"] = ["## Step 8 — Analyse Matched-Seed Reliability\n", "\n", "This step summarises the matched-seed results with means, uncertainty intervals, paired BLF-minus-baseline differences, and wins, ties, and losses. It closes the retrieval reliability study before multi-task evaluation.\n"]

new = {**old, "cells": cells}
with tempfile.NamedTemporaryFile("w", dir=PATH.parent, delete=False, suffix=".ipynb") as handle:
    json.dump(new, handle, indent=1, ensure_ascii=False)
    handle.write("\n")
    temporary = Path(handle.name)
json.loads(temporary.read_text())
temporary.replace(PATH)
print(f"Wrote {len(cells)} cells atomically to {PATH}")
