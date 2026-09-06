# Alignment v2 → v3 → v3-recovery: Project Summary

_Last updated: 2026-07-24_

This document summarizes what was accomplished in **Alignment v2** and
**Alignment v3**, and what the **v3 recovery branch** is designed to do. Each
track writes to its own isolated tree (`results/alignment_v2`,
`results/alignment_v3`, `results/alignment_v3_recovery`) and never overwrites an
earlier track's canonical evidence.

The unifying research question across all three: **can independently pretrained,
frozen vision and text encoders be aligned into a competitive image–text
retrieval model — without jointly pretraining the towers — while staying far
cheaper than a paired VLM?** All results use the same grouped five-caption COCO
retrieval protocol; the headline metric is **mean bidirectional R@1**.

---

## Alignment v2 — COMPLETE ✅

**Goal.** Establish (1) how far frozen unimodal experts sit below a native
paired VLM under an identical protocol, and (2) how much better alignment
training alone can push frozen towers without touching their inference
backbones.

**Method.** Both towers kept frozen; only the alignment procedure changes —
all COCO training captions available, two captions sampled per image per epoch,
rectangular multi-positive contrastive loss, a residual projection head, a short
detached negative queue (train-only, zero inference cost), decoupled-weight-decay
AdamW, cosine schedule with warmup, gradient clipping, early stopping, and
checkpoint selection on mean bidirectional R@1. Three seeds (42/43/44), 12-epoch
ceiling.

**Results (mean R@1, 3-seed).**

| Model | mean R@1 | Latency (ms) | Total params | Trainable |
|---|---|---|---|---|
| **ConvNeXt-Tiny + MiniLM** (residual, multi-positive) | **≈15.60%** (best) | ~2.9 | 52.8M | 2.28M |
| DINOv2-S/14 + MiniLM | ≈13.68% | ~4.8 | 46.9M | 2.09M |
| EfficientNet-B0 + BGE-Small | ≈8.22% | ~4.1 | 39.9M | 2.55M |
| _Paired reference: OpenCLIP ViT-B/32-QuickGELU (OpenAI)_ | _40.19%_ | _4.17_ | _151.3M_ | _0_ |

**Takeaways.**
- The best frozen-expert configuration (**ConvNeXt-Tiny + MiniLM at 15.60%**)
  became the **baseline-to-beat** for all later work.
- A large gap remained to the paired reference (15.6% vs 40.2%), motivating v3.
- Architecture-diverse ConvNeXt beat the semantic DINOv2 expert; EfficientNet+BGE
  was clearly the weakest and effectively ruled out.
- v2 is treated as an **immutable safety-net result** — v3 locks and never
  modifies it.

---

## Alignment v3 — INCOMPLETE (stopped at pair-screening gate) ⛔

**Goal.** Test whether *modern* frozen encoders (DINOv3-family vision towers +
small sentence encoders), plus teacher-guided distillation and tightly
constrained parameter-efficient adaptation (adapter / LoRA), can approach paired-VLM
retrieval quality while preserving the efficiency advantage from v2.

**Leakage-safe protocol.** COCO train deterministically partitioned into 113,287
train / 5,000 dev images, with the existing 5,000 COCO-val images as a sealed
final set. Pair, distillation-strength and recipe selection use **dev only**; the
locked recipe is retrained on full COCO train and evaluated **once** on sealed
COCO val.

**What ran successfully.**
- Validation, split/leakage locking, checkpoint prefetch.
- Three paired references (native transforms):

  | Reference | mean R@1 |
  |---|---|
  | SigLIP2 ViT-B/32-256 (webli) | 56.88% |
  | MobileCLIP2-S0 (dfndr2b) | 52.98% |
  | OpenCLIP ViT-B/32-QuickGELU (openai) | 40.19% |

- Blackwell batch profiling.
- **All six DINOv3 pairs trained + evaluated** (seed 42):

  | Pair | mean R@1 |
  |---|---|
  | **DINOv3 ViT-S/16 + all-MiniLM-L6-v2** | **17.02%** ← selected |
  | DINOv3 ViT-S/16 + bge-small-en | 16.96% |
  | DINOv3 ViT-S/16 + e5-small-v2 | 16.18% |
  | DINOv3 ConvNeXt-Tiny + all-MiniLM-L6-v2 | 16.09% |
  | DINOv3 ConvNeXt-Tiny + e5-small-v2 | 14.24% |
  | DINOv3 ConvNeXt-Tiny + bge-small-en | 14.20% |

- Pair selection, oracle diagnostic, and report generation.

**Why it stopped.** The pre-registered continuation gate required **mean R@1 ≥ 20%
OR ≥ +3pp over v2 (0.1560)**. The winner reached **17.02%**, only **+1.42pp** over
v2 → `STOPPED_AT_GATE`. All downstream stages (teacher cache, distillation
sensitivity, adapter/LoRA/distillation ablations, 3-seed final, sealed
evaluation) were correctly **skipped**, and the report honestly reads
`INCOMPLETE`.

**This is a valid negative screening result and is preserved unaltered.** v3's
gate thresholds are not retroactively lowered.

**Signals discovered in the v3 data.**
- The vision encoder, not the text encoder, drives the difference: the three
  text encoders differ by <1pp within each vision tower (likely single-seed
  noise), while ViT-S/16 beats ConvNeXt-Tiny by ~2pp.
- **The batch probe's selection was degenerate.** It chose batch **2048** for
  every pair, but throughput is *flat* across all candidate batches (ViT-S/16:
  ~4294 img/s @256 vs ~4328 @2048, <1% spread) and memory never exceeds ~7% of
  the 96 GB card. Its rule ("largest batch within 3% of peak throughput") had no
  discriminating power and defaulted to the maximum. Optimization quality was
  never an input.
- At batch 2048 with ~112k images, that is only **~55 optimizer updates/epoch
  (~660 over 12 epochs)** — likely **under-optimized**. Because throughput is
  flat, a smaller batch gives ~4× more updates at ~the same wall-clock.
- A secondary Slurm-reporting weakness was found: gate-skipped stages exit 0, so
  Slurm labelled the whole graph COMPLETED even though the science stopped early.

---

## Alignment v3 Recovery — designed / implemented, not yet submitted 🚧

**Hypothesis.** v3's 17.02% failure reflects an **under-optimized training
schedule (oversized batch)**, not an architectural ceiling. If the winning pair
is trained properly, it may clear the same gate — and distillation/LoRA may then
push it further.

**Scope & integrity.** A **separately labelled exploratory branch** writing only
under `*_recovery`. v3's screening result stays frozen. New gates are
**pre-registered before running** — this is *not* a retroactive relaxation of
v3's gate.

### What it does, stage by stage

1. **Batch calibration.** Retrain the locked pair (**DINOv3 ViT-S/16 +
   all-MiniLM**) at **batch 512, 768, 1024, and 2048** (2048 = faithful control
   reproducing the failed run), seed 42. The auto-probe is disabled so pinned
   batches govern. LR auto-scales per batch (`0.0003 × min(√(batch/128), 3.0)`),
   so only the optimization schedule varies.

2. **Confirmation.** Shortlist the **best 2 batch sizes** and re-run them on
   **seed 43**, so the winner is not a single-seed fluke.

3. **Recalibrated pair gate.** Apply the **same** bar (≥20% mean R@1 OR ≥+3pp
   over v2) to the **two-seed mean** of the best setting. If nothing clears it,
   **stop** — a legitimate result ("even properly optimized, this pair falls
   short").

4. **Compact 4-recipe study** _(only if the gate passes)._ Build the MobileCLIP2
   teacher cache, then compare, at the winning batch and two seeds:
   **projection-only baseline · distillation-only · LoRA-only ·
   distillation+LoRA**. The token adapter is deferred. A `direct_recipe_study`
   flag routes straight here, deliberately bypassing v3's 30% distillation-
   sensitivity gate (which was calibrated for a stronger baseline and would
   otherwise block the comparison).

5. **Final validation** _(only if a recipe wins)._ Promote a recipe only if it
   beats the matched recalibrated baseline by **≥+2pp** within budget
   (**≤80M inference params, ≤5M trainable, ≤4.17 ms latency**). Retrain the
   winner on **full COCO train, seeds 42/43/44**, evaluate **once** on sealed
   COCO val, then run the oracle diagnostic and report.

### Implementation status

- **Runner, configs, splits, Slurm graph, submit script, methodology doc:
  implemented.** The pipeline additions (per-batch pair jobs, two-stage
  confirmation, `_selected_batch` propagation, `select-recovery-*` stages,
  `direct_recipe_study`) are in place and backward-compatible (v3 behaviour
  unchanged).
- **Slurm-reporting fix: applied.** The terminal `report` stage now exits code 3
  on any non-`COMPLETE` run, so a gated stop shows as FAILED instead of a false
  COMPLETED — scoped to the sink node so it doesn't cancel upstream gate-skipped
  stages.
- **Known gap — notebook observability not wired.** Recovery results land as
  JSON/CSV and a `report.md` under `results/alignment_v3_recovery/`, but **no
  notebook renders them**: `notebooks/03_alignment_v3_results.ipynb` is hardcoded
  to the v3 directory, no recovery notebook exists, and the recovery Slurm wrapper
  omits the notebook-rendering step. This is a reporting/visibility gap, not a
  correctness gap.

### Intended narrative (if it succeeds)

> Very large batches maximize hardware throughput but can under-optimize
> heterogeneous frozen-encoder alignment; a calibrated training schedule plus
> lightweight distillation/LoRA recovers retrieval performance under strict
> parameter and latency budgets.

---

## Baseline chain at a glance

| Milestone | Best config | mean R@1 |
|---|---|---|
| Paired reference (efficiency-class) | OpenCLIP ViT-B/32-QuickGELU | 40.19% |
| **v2** frozen-unimodal best | ConvNeXt-Tiny + MiniLM | **15.60%** |
| **v3** modern-encoder screen (stopped) | DINOv3 ViT-S/16 + MiniLM @ batch 2048 | **17.02%** (+1.42pp, gate not met) |
| **v3-recovery** target | Same pair, recalibrated batch (+ optional distill/LoRA) | ≥20% or ≥+3pp over v2 _(TBD)_ |
